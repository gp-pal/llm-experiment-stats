"""Turn a model's reply into right or wrong.

The scoring rule is fixed before any model is called and is the same for every
variant. A reply that ignores the requested format is wrong, and is counted
separately so a format problem is not mistaken for a reading problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ANSWER_MARKER = "ATSAKYMAS:"

Status = Literal["correct", "wrong_number", "not_a_number", "no_marker"]

_MARKER_LINE = re.compile(rf"^\W*{re.escape(ANSWER_MARKER)}(.*)$", re.MULTILINE | re.IGNORECASE)
# Thousands separators people and models actually use: any Unicode whitespace
# (\s covers non-breaking spaces), dots, commas, apostrophes. A currency may follow.
_SEPARATORS = re.compile(r"[\s.,'\u2019]")
_CURRENCY = re.compile(r"(eur\w*|€)$", re.IGNORECASE)


@dataclass(frozen=True)
class Score:
    status: Status
    parsed: int | None

    @property
    def correct(self) -> bool:
        return self.status == "correct"


def parse_answer(reply: str) -> int | None:
    """Read the number from the last `ATSAKYMAS:` line, or None if there is none."""
    matches = _MARKER_LINE.findall(reply)
    if not matches:
        return None
    value = _SEPARATORS.sub("", matches[-1].strip().strip("*_`"))
    value = _CURRENCY.sub("", value)
    return int(value) if value.isdecimal() else None


def score(reply: str, answer: int) -> Score:
    if not _MARKER_LINE.search(reply):
        return Score("no_marker", None)
    parsed = parse_answer(reply)
    if parsed is None:
        return Score("not_a_number", None)
    return Score("correct" if parsed == answer else "wrong_number", parsed)
