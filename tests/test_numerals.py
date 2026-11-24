from __future__ import annotations

import pytest

from llm_ab.numerals import euros_in_words, number_to_words


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (1, "vienas"),
        (12, "dvylika"),
        (20, "dvidešimt"),
        (21, "dvidešimt vienas"),
        (100, "šimtas"),
        (215, "du šimtai penkiolika"),
        (1_000, "tūkstantis"),
        (2_001, "du tūkstančiai vienas"),
        (10_000, "dešimt tūkstančių"),
        (11_000, "vienuolika tūkstančių"),
        (21_000, "dvidešimt vienas tūkstantis"),
        (22_000, "dvidešimt du tūkstančiai"),
        (100_000, "šimtas tūkstančių"),
        (245_617, "du šimtai keturiasdešimt penki tūkstančiai šeši šimtai septyniolika"),
        (1_000_000, "milijonas"),
        (5_000_000, "penki milijonai"),
        (30_000_000, "trisdešimt milijonų"),
        (2_000_017, "du milijonai septyniolika"),
    ],
)
def test_nominative(number: int, words: str) -> None:
    assert number_to_words(number) == words


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (1, "vieną"),
        (3, "tris"),
        (100, "šimtą"),
        (300, "tris šimtus"),
        (1_000, "tūkstantį"),
        (4_000, "keturis tūkstančius"),
        (15_000, "penkiolika tūkstančių"),
        (21_000, "dvidešimt vieną tūkstantį"),
        (1_000_000, "milijoną"),
        (2_345_617, "du milijonus tris šimtus keturiasdešimt penkis tūkstančius "
                    "šešis šimtus septyniolika"),
    ],
)  # fmt: skip
def test_accusative(number: int, words: str) -> None:
    assert number_to_words(number, "accusative") == words


@pytest.mark.parametrize(
    ("number", "case", "words"),
    [
        (1, "nominative", "vienas euras"),
        (2, "nominative", "du eurai"),
        (10, "nominative", "dešimt eurų"),
        (21, "nominative", "dvidešimt vienas euras"),
        (1_000, "nominative", "tūkstantis eurų"),
        (21, "accusative", "dvidešimt vieną eurą"),
        (4, "accusative", "keturis eurus"),
        (1_500, "accusative", "tūkstantį penkis šimtus eurų"),
    ],
)
def test_euro_noun_agrees_with_the_number(number: int, case: str, words: str) -> None:
    assert euros_in_words(number, case) == words  # type: ignore[arg-type]


@pytest.mark.parametrize("number", [0, -5, 1_000_000_000])
def test_out_of_range_numbers_are_rejected(number: int) -> None:
    with pytest.raises(ValueError, match="between 1 and"):
        number_to_words(number)
