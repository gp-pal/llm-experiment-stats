"""What is being compared: the prompts, the arms, and the planned comparisons.

The comparisons are listed here, before any model is called. Deciding what to
compare after looking at the results is how an experiment finds effects that are
not there.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_ab.scoring import ANSWER_MARKER

_FORMAT = (
    f"Paskutinėje atsakymo eilutėje parašyk sumą skaitmenimis tokiu formatu:\n"
    f"{ANSWER_MARKER} <skaičius be tarpų ir be valiutos>"
)

# Both prompts ask for the same output format, so the only thing that differs is
# the instruction on how to get to the number.
PROMPTS: dict[str, str] = {
    "baseline": f"Sakinyje pinigų suma parašyta žodžiais.\n{_FORMAT}",
    "decompose": (
        "Sakinyje pinigų suma parašyta žodžiais.\n"
        "Prieš atsakydamas atskirai užrašyk skaitmenimis milijonus, tūkstančius ir likutį, "
        "tada juos sudėk. Jei kurios nors dalies sakinyje nėra, ji lygi nuliui. "
        "Sakinyje esantys metai nėra sumos dalis.\n"
        f"{_FORMAT}"
    ),
}


@dataclass(frozen=True)
class Arm:
    name: str
    prompt: str


@dataclass(frozen=True)
class Comparison:
    a: str
    b: str
    question: str


ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in (
        Arm("baseline", "baseline"),
        # The same prompt a second time. Any difference from `baseline` is noise,
        # and it shows how large a difference noise alone produces.
        Arm("baseline-repeat", "baseline"),
        Arm("decompose", "decompose"),
    )
}

COMPARISONS: tuple[Comparison, ...] = (
    Comparison("baseline", "baseline-repeat", "A/A: how much does the same prompt vary?"),
    Comparison("baseline", "decompose", "A/B: does asking for a decomposition help?"),
)
