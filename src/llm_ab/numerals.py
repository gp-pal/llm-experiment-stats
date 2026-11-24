"""Lithuanian cardinal numerals, written out, in the nominative or the accusative.

The eval task is to read a sum written in words and return it in digits. Generating
the words from the number gives an exact ground truth for every item, with no
labelling step and no dataset licence to worry about.
"""

from __future__ import annotations

from typing import Literal

Case = Literal["nominative", "accusative"]

_UNITS: dict[Case, list[str]] = {
    "nominative": [
        "", "vienas", "du", "trys", "keturi", "penki", "šeši", "septyni", "aštuoni", "devyni",
    ],
    "accusative": [
        "", "vieną", "du", "tris", "keturis", "penkis", "šešis", "septynis", "aštuonis", "devynis",
    ],
}  # fmt: skip
# Teens and tens do not decline in these two cases.
_TEENS = [
    "dešimt", "vienuolika", "dvylika", "trylika", "keturiolika",
    "penkiolika", "šešiolika", "septyniolika", "aštuoniolika", "devyniolika",
]  # fmt: skip
_TENS = [
    "", "", "dvidešimt", "trisdešimt", "keturiasdešimt",
    "penkiasdešimt", "šešiasdešimt", "septyniasdešimt", "aštuoniasdešimt", "devyniasdešimt",
]  # fmt: skip

# (after 1, after 2-9, after 0 and 10-19): the third form is the genitive plural,
# which is the same in both cases.
_Forms = tuple[str, str, str]
_HUNDRED: dict[Case, _Forms] = {
    "nominative": ("šimtas", "šimtai", "šimtų"),
    "accusative": ("šimtą", "šimtus", "šimtų"),
}
_THOUSAND: dict[Case, _Forms] = {
    "nominative": ("tūkstantis", "tūkstančiai", "tūkstančių"),
    "accusative": ("tūkstantį", "tūkstančius", "tūkstančių"),
}
_MILLION: dict[Case, _Forms] = {
    "nominative": ("milijonas", "milijonai", "milijonų"),
    "accusative": ("milijoną", "milijonus", "milijonų"),
}
_EURO: dict[Case, _Forms] = {
    "nominative": ("euras", "eurai", "eurų"),
    "accusative": ("eurą", "eurus", "eurų"),
}

MAX_NUMBER = 999_999_999


def _noun_form(count: int, forms: _Forms) -> str:
    """Pick the noun form that agrees with the last two digits of `count`."""
    last_two = count % 100
    last = count % 10
    if last == 0 or 10 <= last_two <= 19:
        return forms[2]
    if last == 1:
        return forms[0]
    return forms[1]


def _below_thousand(n: int, case: Case) -> list[str]:
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds == 1:
        words.append(_HUNDRED[case][0])
    elif hundreds > 1:
        words += [_UNITS[case][hundreds], _HUNDRED[case][1]]
    if 10 <= rest <= 19:
        words.append(_TEENS[rest - 10])
    else:
        tens, units = divmod(rest, 10)
        if tens:
            words.append(_TENS[tens])
        if units:
            words.append(_UNITS[case][units])
    return words


def _group(count: int, forms: _Forms, case: Case) -> list[str]:
    if count == 0:
        return []
    if count == 1:
        return [forms[0]]
    return [*_below_thousand(count, case), _noun_form(count, forms)]


def number_to_words(n: int, case: Case = "nominative") -> str:
    """Write `n` out in Lithuanian words, e.g. 2001 -> 'du tūkstančiai vienas'."""
    if not 1 <= n <= MAX_NUMBER:
        raise ValueError(f"number must be between 1 and {MAX_NUMBER}")
    millions, rest = divmod(n, 1_000_000)
    thousands, units = divmod(rest, 1_000)
    words = [
        *_group(millions, _MILLION[case], case),
        *_group(thousands, _THOUSAND[case], case),
        *_below_thousand(units, case),
    ]
    return " ".join(words)


def euros_in_words(n: int, case: Case = "nominative") -> str:
    """Write a sum of euros out in words, with the noun agreeing with the number."""
    return f"{number_to_words(n, case)} {_noun_form(n, _EURO[case])}"
