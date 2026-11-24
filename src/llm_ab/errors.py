"""Sort wrong answers by what went wrong.

This is description, written after looking at the replies. It does not feed any
test: it explains a difference the planned comparison has already established.
"""

from __future__ import annotations

from typing import Literal

from llm_ab.scoring import Score

ErrorKind = Literal["truncated", "no_answer", "added_groups", "misread"]

ERROR_KINDS: dict[ErrorKind, str] = {
    "misread": "misread the number",
    "added_groups": "added group counts, not place values",
    "truncated": "cut off at the token limit",
    "no_answer": "no usable answer line",
}


def group_counts(n: int) -> tuple[int, int, int]:
    """Millions, thousands and the remainder: 257 121 -> (0, 257, 121)."""
    millions, rest = divmod(n, 1_000_000)
    thousands, units = divmod(rest, 1_000)
    return millions, thousands, units


def classify(answer: int, result: Score, truncated: bool) -> ErrorKind | None:
    """What kind of error a reply is, or None if it is correct.

    `added_groups` is the answer 378 to 257 121: the groups were read correctly and
    then added as bare counts (257 + 121) instead of as 257 000 + 121.
    """
    if result.correct:
        return None
    if truncated:
        return "truncated"
    if result.parsed is None:
        return "no_answer"
    if result.parsed == sum(group_counts(answer)):
        return "added_groups"
    return "misread"
