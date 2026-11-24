from __future__ import annotations

import pytest

from llm_ab.scoring import parse_answer, score


@pytest.mark.parametrize(
    "reply",
    [
        "ATSAKYMAS: 2345617",
        "ATSAKYMAS: 2 345 617",
        "ATSAKYMAS: 2\u00a0345\u00a0617",
        "ATSAKYMAS: 2.345.617",
        "ATSAKYMAS: 2,345,617",
        "ATSAKYMAS: 2345617 eurų",
        "ATSAKYMAS: 2345617 €",
        "**ATSAKYMAS: 2345617**",
        "atsakymas: 2345617",
        "Milijonai: 2 000 000\nTūkstančiai: 345 000\nLikutis: 617\nATSAKYMAS: 2345617",
    ],
)
def test_accepted_ways_of_writing_the_answer(reply: str) -> None:
    assert parse_answer(reply) == 2345617


def test_the_last_marker_line_wins() -> None:
    reply = "ATSAKYMAS: 100\nPataisau.\nATSAKYMAS: 200"

    assert parse_answer(reply) == 200


def test_a_year_elsewhere_in_the_reply_is_not_the_answer() -> None:
    reply = "2021 metais suma buvo didelė.\nATSAKYMAS: 5300"

    assert score(reply, 5300).correct


def test_wrong_number_is_reported_with_what_was_read() -> None:
    result = score("ATSAKYMAS: 2021", 5300)

    assert result.status == "wrong_number"
    assert result.parsed == 2021


def test_reply_without_the_marker_is_a_format_failure() -> None:
    result = score("Suma yra 5300 eurų.", 5300)

    assert result.status == "no_marker"
    assert not result.correct


def test_words_after_the_marker_are_not_a_number() -> None:
    assert score("ATSAKYMAS: penki tūkstančiai", 5000).status == "not_a_number"
