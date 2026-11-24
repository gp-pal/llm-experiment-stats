"""The eval set: sentences with a sum written in words, and the sum in digits."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

from llm_ab.numerals import Case, euros_in_words

Magnitude = Literal["thousands", "millions"]

# Every sentence carries a year in digits: a number that is not the answer.
_TEMPLATES: dict[Case, list[str]] = {
    "nominative": [
        "{year} metų įmonės pelnas buvo {sum}.",
        "Bendra {year} metų sąskaitų suma – {sum}.",
    ],
    "accusative": [
        "{year} metais įmonė uždirbo {sum}.",
        "Per {year} metus klientai sumokėjo {sum}.",
    ],
}


@dataclass(frozen=True)
class Item:
    id: str
    text: str
    answer: int
    case: Case
    magnitude: Magnitude
    #: A whole group is absent from the words, as in 2 000 017 or 5 000 300.
    skipped_group: bool


def _draw_number(rng: random.Random, magnitude: Magnitude, skip_group: bool) -> int:
    millions = rng.randint(1, 999) if magnitude == "millions" else 0
    thousands = rng.randint(1, 999)
    units = rng.randint(1, 999)
    if skip_group:
        if magnitude == "millions" and rng.random() < 0.5:
            thousands = 0
        else:
            units = 0
    return millions * 1_000_000 + thousands * 1_000 + units


def generate_items(n: int, seed: int = 0) -> list[Item]:
    """Generate `n` items, balanced over case and magnitude."""
    rng = random.Random(seed)
    cases: tuple[Case, ...] = get_args(Case)
    magnitudes: tuple[Magnitude, ...] = get_args(Magnitude)
    items: list[Item] = []
    for index in range(n):
        case = cases[index % 2]
        magnitude = magnitudes[(index // 2) % 2]
        skip_group = rng.random() < 0.3
        answer = _draw_number(rng, magnitude, skip_group)
        template = rng.choice(_TEMPLATES[case])
        text = template.format(year=rng.randint(2015, 2024), sum=euros_in_words(answer, case))
        items.append(
            Item(
                id=f"item-{index:04d}",
                text=text,
                answer=answer,
                case=case,
                magnitude=magnitude,
                skipped_group=skip_group,
            )
        )
    return items


def save_items(items: Iterable[Item], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def load_items(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as file:
        return [Item(**json.loads(line)) for line in file if line.strip()]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the eval set.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/items.jsonl"))
    args = parser.parse_args()
    save_items(generate_items(args.n, args.seed), args.out)
    print(f"wrote {args.n} items to {args.out}")


if __name__ == "__main__":
    main()
