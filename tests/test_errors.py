from __future__ import annotations

import pytest

from llm_ab.errors import classify, group_counts
from llm_ab.scoring import score


def test_group_counts_split_a_number_into_millions_thousands_and_the_rest() -> None:
    assert group_counts(2_345_617) == (2, 345, 617)
    assert group_counts(257_121) == (0, 257, 121)


def test_a_correct_reply_is_not_an_error() -> None:
    assert classify(257_121, score("ATSAKYMAS: 257121", 257_121), truncated=False) is None


@pytest.mark.parametrize(
    ("answer", "reply"),
    [
        (257_121, "ATSAKYMAS: 378"),  # 257 + 121
        (524_000, "ATSAKYMAS: 524"),  # 524 + 0
        (2_345_617, "ATSAKYMAS: 964"),  # 2 + 345 + 617
    ],
)
def test_adding_the_group_counts_is_recognised(answer: int, reply: str) -> None:
    assert classify(answer, score(reply, answer), truncated=False) == "added_groups"


def test_any_other_wrong_number_is_a_misreading() -> None:
    assert classify(119_499, score("ATSAKYMAS: 191499", 119_499), truncated=False) == "misread"


def test_a_truncated_reply_is_counted_as_truncated_whatever_it_says() -> None:
    assert classify(5_300, score("", 5_300), truncated=True) == "truncated"
    assert classify(5_300, score("ATSAKYMAS: 53", 5_300), truncated=True) == "truncated"


def test_a_complete_reply_without_a_number_has_no_answer() -> None:
    assert classify(5_300, score("Nežinau.", 5_300), truncated=False) == "no_answer"
