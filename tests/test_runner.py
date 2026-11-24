from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from llm_ab.experiment import ARMS, PROMPTS
from llm_ab.items import generate_items
from llm_ab.records import run_path
from llm_ab.runner import Settings, recorded_ids, run_arm, run_arms

SETTINGS = Settings(base_url="https://llm.test/v1", api_key="test-key", model="test-model")


def _reply(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_every_item_is_sent_with_the_arms_prompt_and_recorded(tmp_path: Path) -> None:
    items = generate_items(3)
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        return _reply("ATSAKYMAS: 1")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        added = run_arm(client, SETTINGS, ARMS["decompose"], items, tmp_path)

    assert added == 3
    assert [r["messages"] for r in requests] == [
        [
            {"role": "system", "content": PROMPTS["decompose"]},
            {"role": "user", "content": item.text},
        ]
        for item in items
    ]
    records = _records(run_path(tmp_path, "decompose"))
    assert [r["item_id"] for r in records] == [item.id for item in items]
    assert {r["reply"] for r in records} == {"ATSAKYMAS: 1"}
    assert {r["model"] for r in records} == {"test-model"}


def test_a_second_run_only_asks_for_what_is_missing(tmp_path: Path) -> None:
    items = generate_items(5)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _reply("ATSAKYMAS: 1")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run_arm(client, SETTINGS, ARMS["baseline"], items[:2], tmp_path)
        added = run_arm(client, SETTINGS, ARMS["baseline"], items, tmp_path)

    assert added == 3
    assert calls == 5
    assert recorded_ids(run_path(tmp_path, "baseline")) == {item.id for item in items}


def test_rate_limit_is_waited_out_and_retried(tmp_path: Path) -> None:
    responses = iter([httpx.Response(429, headers={"retry-after": "7"}), _reply("ATSAKYMAS: 1")])
    waits: list[float] = []

    with httpx.Client(transport=httpx.MockTransport(lambda _: next(responses))) as client:
        added = run_arm(
            client, SETTINGS, ARMS["baseline"], generate_items(1), tmp_path, sleep=waits.append
        )

    assert added == 1
    assert waits == [7.0]
    assert _records(run_path(tmp_path, "baseline"))[0]["attempts"] == 2


def test_a_dropped_connection_is_retried(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connection reset")
        return _reply("ATSAKYMAS: 1")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        added = run_arm(
            client, SETTINGS, ARMS["baseline"], generate_items(1), tmp_path, sleep=lambda _: None
        )

    assert (added, calls) == (1, 2)


def test_a_rate_limit_that_never_lifts_stops_the_run(tmp_path: Path) -> None:
    settings = Settings(base_url="https://llm.test/v1", api_key="", model="m", max_retries=2)
    waits: list[float] = []

    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(429))) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        run_arm(client, settings, ARMS["baseline"], generate_items(1), tmp_path, sleep=waits.append)

    assert waits == [1.0, 2.0]


def test_arms_take_turns_on_each_item(tmp_path: Path) -> None:
    items = generate_items(2)
    order: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        system, user = (m["content"] for m in json.loads(request.content)["messages"])
        order.append((system, user))
        return _reply("ATSAKYMAS: 1")

    arms = [ARMS["baseline"], ARMS["decompose"]]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        added = run_arms(client, SETTINGS, arms, items, tmp_path)

    assert added == {"baseline": 2, "decompose": 2}
    assert order == [(PROMPTS[arm.prompt], item.text) for item in items for arm in arms]


def test_a_client_error_stops_the_run_and_keeps_earlier_replies(tmp_path: Path) -> None:
    responses = iter([_reply("ATSAKYMAS: 1"), httpx.Response(404, json={"error": "no model"})])

    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: next(responses))) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        run_arm(client, SETTINGS, ARMS["baseline"], generate_items(2), tmp_path)

    assert len(_records(run_path(tmp_path, "baseline"))) == 1


def test_empty_content_is_recorded_as_an_empty_reply(tmp_path: Path) -> None:
    response = httpx.Response(
        200, json={"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
    )

    with httpx.Client(transport=httpx.MockTransport(lambda _: response)) as client:
        run_arm(client, SETTINGS, ARMS["baseline"], generate_items(1), tmp_path)

    record = _records(run_path(tmp_path, "baseline"))[0]
    assert record["reply"] == ""
    assert record["finish_reason"] == "length"
