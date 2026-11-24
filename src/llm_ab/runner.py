"""Run the arms against an OpenAI-compatible chat endpoint and record every reply.

Replies are appended to one JSONL file per arm. The analysis reads only those files,
so it can be rerun, and reviewed, without calling a model again.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llm_ab.experiment import ARMS, PROMPTS, Arm
from llm_ab.items import Item, load_items
from llm_ab.records import read_records, run_path

logger = logging.getLogger(__name__)

Sleep = Callable[[float], None]


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    #: Seconds to wait between calls, to stay under a tokens-per-minute limit.
    pause: float = 0.0
    max_retries: int = 5

    @classmethod
    def from_env(cls) -> Settings:
        missing = [name for name in ("LLM_BASE_URL", "LLM_MODEL") if not os.environ.get(name)]
        if missing:
            raise SystemExit(f"set {', '.join(missing)} in the environment or in .env")
        return cls(
            base_url=os.environ["LLM_BASE_URL"].rstrip("/"),
            api_key=os.environ.get("LLM_API_KEY", ""),
            model=os.environ["LLM_MODEL"],
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2048")),
            pause=float(os.environ.get("LLM_PAUSE_SECONDS", "0")),
        )


def recorded_ids(path: Path) -> set[str]:
    return {record["item_id"] for record in read_records(path)}


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """What the server asked for, or exponential backoff if it did not say."""
    header = response.headers.get("retry-after") if response is not None else None
    try:
        return float(header) if header else 2.0**attempt
    except ValueError:
        return 2.0**attempt


def ask(
    client: httpx.Client,
    settings: Settings,
    system: str,
    user: str,
    sleep: Sleep = time.sleep,
) -> tuple[dict[str, Any], int]:
    """One chat completion and the number of attempts it took.

    Rate limits, server errors and network failures are retried; any other error
    stops the run, because repeating a bad request does not fix it.
    """
    payload = {
        "model": settings.model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
    url = f"{settings.base_url}/chat/completions"
    attempt = 0
    while True:
        response: httpx.Response | None = None
        try:
            response = client.post(url, json=payload, headers=headers)
            problem = f"HTTP {response.status_code}"
            retryable = response.status_code == 429 or response.status_code >= 500
        except httpx.TransportError as error:
            if attempt >= settings.max_retries:
                raise
            problem, retryable = type(error).__name__, True
        if response is not None and not (retryable and attempt < settings.max_retries):
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            return body, attempt + 1
        delay = _retry_delay(response, attempt)
        logger.info("%s, retrying in %.0f s", problem, delay)
        sleep(delay)
        attempt += 1


def run_arms(
    client: httpx.Client,
    settings: Settings,
    arms: Sequence[Arm],
    items: Sequence[Item],
    runs_dir: Path,
    sleep: Sleep = time.sleep,
) -> dict[str, int]:
    """Answer every item not yet recorded. Returns how many replies each arm gained.

    The arms take turns on each item instead of running one after another, so
    anything that changes over the hours a run takes (the provider's load, a silent
    model update) reaches every arm alike and is not mistaken for a prompt effect.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    done = {arm.name: recorded_ids(run_path(runs_dir, arm.name)) for arm in arms}
    added = {arm.name: 0 for arm in arms}
    with ExitStack() as stack:
        files = {
            arm.name: stack.enter_context(run_path(runs_dir, arm.name).open("a", encoding="utf-8"))
            for arm in arms
        }
        for position, item in enumerate(items, start=1):
            for arm in arms:
                if item.id in done[arm.name]:
                    continue
                started = time.monotonic()
                body, attempts = ask(client, settings, PROMPTS[arm.prompt], item.text, sleep)
                choice = body["choices"][0]
                record = {
                    "item_id": item.id,
                    "arm": arm.name,
                    "prompt": arm.prompt,
                    "model": settings.model,
                    "temperature": settings.temperature,
                    "max_tokens": settings.max_tokens,
                    "reply": choice["message"].get("content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": body.get("usage"),
                    "attempts": attempts,
                    # Includes the time spent waiting out rate limits.
                    "seconds": round(time.monotonic() - started, 3),
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                file = files[arm.name]
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                added[arm.name] += 1
                logger.info("%s %d/%d %s", arm.name, position, len(items), item.id)
                if settings.pause:
                    sleep(settings.pause)
    return added


def run_arm(
    client: httpx.Client,
    settings: Settings,
    arm: Arm,
    items: Sequence[Item],
    runs_dir: Path,
    sleep: Sleep = time.sleep,
) -> int:
    """Answer every item not yet recorded for one arm. Returns how many were added."""
    return run_arms(client, settings, [arm], items, runs_dir, sleep)[arm.name]


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Run experiment arms and record replies.")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--items", type=Path, default=Path("data/items.jsonl"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--limit", type=int, help="only the first N items")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    # httpx logs every request at INFO, which would bury the progress lines.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()
    settings = Settings.from_env()
    items = load_items(args.items)[: args.limit]
    with httpx.Client(timeout=120) as client:
        added = run_arms(client, settings, [ARMS[name] for name in args.arms], items, args.runs)
    for name, count in added.items():
        print(f"{name}: {count} new replies, {len(items) - count} already recorded")


if __name__ == "__main__":
    main()
