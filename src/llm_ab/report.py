"""Read the recorded replies and report what the experiment shows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from llm_ab.errors import ERROR_KINDS, ErrorKind, classify
from llm_ab.experiment import ARMS, COMPARISONS, Comparison
from llm_ab.items import Item, load_items
from llm_ab.records import read_records, run_path
from llm_ab.scoring import Score, score
from llm_ab.stats import (
    Interval,
    PairedCounts,
    holm,
    mcnemar_exact,
    paired_bootstrap_difference,
    paired_counts,
    wilson_interval,
)

if TYPE_CHECKING:
    from llm_ab.plots import ReportData

Scores = dict[str, Score]


@dataclass(frozen=True)
class ArmReplies:
    """One arm's recorded replies, scored."""

    scores: Scores
    model: str
    #: Items whose reply was cut off at the token limit before it finished.
    truncated: frozenset[str]


@dataclass(frozen=True)
class ArmResult:
    name: str
    model: str
    n: int
    correct: int
    interval: Interval
    statuses: Counter[str]
    truncated: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n


@dataclass(frozen=True)
class ComparisonResult:
    comparison: Comparison
    counts: PairedCounts
    interval: Interval
    p_value: float
    p_adjusted: float


def load_arm(runs_dir: Path, arm: str, items: Sequence[Item]) -> ArmReplies:
    """Score an arm's recorded replies against the items' answers."""
    answers = {item.id: item.answer for item in items}
    records = [r for r in read_records(run_path(runs_dir, arm)) if r["item_id"] in answers]
    models = {r["model"] for r in records}
    if len(models) > 1:
        raise ValueError(f"arm {arm!r} mixes models: {sorted(models)}")
    return ArmReplies(
        scores={r["item_id"]: score(r["reply"], answers[r["item_id"]]) for r in records},
        model=next(iter(models), ""),
        truncated=frozenset(r["item_id"] for r in records if r.get("finish_reason") == "length"),
    )


def summarise_arm(name: str, arm: ArmReplies) -> ArmResult:
    correct = sum(s.correct for s in arm.scores.values())
    return ArmResult(
        name=name,
        model=arm.model,
        n=len(arm.scores),
        correct=correct,
        interval=wilson_interval(correct, len(arm.scores)),
        statuses=Counter(s.status for s in arm.scores.values()),
        truncated=len(arm.truncated),
    )


def _paired(a: Scores, b: Scores, only: set[str] | None = None) -> tuple[list[bool], list[bool]]:
    """Outcomes on the items both arms answered, in a fixed order."""
    shared = sorted(i for i in a.keys() & b.keys() if only is None or i in only)
    return [a[i].correct for i in shared], [b[i].correct for i in shared]


def compare(all_scores: dict[str, Scores]) -> list[ComparisonResult]:
    """Run the planned comparisons that have data, with Holm-adjusted p-values."""
    unadjusted: list[ComparisonResult] = []
    for comparison in COMPARISONS:
        a, b = _paired(all_scores.get(comparison.a, {}), all_scores.get(comparison.b, {}))
        if not a:
            continue
        counts = paired_counts(a, b)
        p_value = mcnemar_exact(counts.only_a, counts.only_b)
        unadjusted.append(
            ComparisonResult(
                comparison, counts, paired_bootstrap_difference(a, b), p_value, p_value
            )
        )
    adjusted = holm([result.p_value for result in unadjusted])
    return [
        ComparisonResult(r.comparison, r.counts, r.interval, r.p_value, p_adjusted)
        for r, p_adjusted in zip(unadjusted, adjusted, strict=True)
    ]


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _p(value: float) -> str:
    """A p-value never prints as 0.0000: below the resolution it is shown as a bound."""
    return "< 0.0001" if value < 0.0001 else f"{value:.4f}"


def _pp(value: float) -> str:
    return f"{100 * value:+.1f} pp"


def _verdict(result: ComparisonResult) -> str:
    if result.p_adjusted < 0.05:
        better = result.comparison.b if result.counts.difference > 0 else result.comparison.a
        return f"`{better}` is better"
    return "no difference detected"


_SLICES: tuple[tuple[str, Callable[[Item], bool]], ...] = (
    ("thousands", lambda i: i.magnitude == "thousands"),
    ("millions", lambda i: i.magnitude == "millions"),
    ("nominative", lambda i: i.case == "nominative"),
    ("accusative", lambda i: i.case == "accusative"),
    ("a group is skipped", lambda i: i.skipped_group),
    ("no group is skipped", lambda i: not i.skipped_group),
)


def _arm_table(arms: Sequence[ArmResult]) -> list[str]:
    lines = [
        "## Accuracy per arm",
        "",
        "Truncated replies were cut off at the token limit; they are scored like any other.",
        "",
        "| Arm | Model | Items | Correct | Accuracy | 95% CI (Wilson) "
        "| Wrong number | Bad format | Truncated |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for arm in arms:
        bad_format = arm.statuses["no_marker"] + arm.statuses["not_a_number"]
        lines.append(
            f"| `{arm.name}` | {arm.model} | {arm.n} | {arm.correct} | {_pct(arm.accuracy)} | "
            f"{_pct(arm.interval.low)} to {_pct(arm.interval.high)} | "
            f"{arm.statuses['wrong_number']} | {bad_format} | {arm.truncated} |"
        )
    return lines


def _comparison_table(results: Sequence[ComparisonResult]) -> list[str]:
    lines = [
        "## Planned comparisons",
        "",
        "Difference is accuracy of B minus accuracy of A, on the items both arms answered.",
        "",
        "| A | B | Items | Only A right | Only B right | Difference | 95% CI (bootstrap) "
        "| p (McNemar) | p (Holm) | Verdict |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.comparison.a}` | `{r.comparison.b}` | {r.counts.n} | {r.counts.only_a} | "
            f"{r.counts.only_b} | {_pp(r.counts.difference)} | "
            f"{_pp(r.interval.low)} to {_pp(r.interval.high)} | {_p(r.p_value)} | "
            f"{_p(r.p_adjusted)} | {_verdict(r)} |"
        )
    return lines


def _slice_table(comparison: Comparison, items: Sequence[Item], a: Scores, b: Scores) -> list[str]:
    lines = [
        f"### Where `{comparison.a}` and `{comparison.b}` differ",
        "",
        "Descriptive only: these slices were not planned as tests and are small.",
        "",
        f"| Slice | Items | `{comparison.a}` | `{comparison.b}` | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, n, accuracy_a, accuracy_b in _slice_accuracy(items, a, b):
        lines.append(
            f"| {label} | {n} | {_pct(accuracy_a)} | {_pct(accuracy_b)} | "
            f"{_pp(accuracy_b - accuracy_a)} |"
        )
    return lines


def _truncation_table(
    results: Sequence[ComparisonResult], arms: dict[str, ArmReplies]
) -> list[str]:
    """How much of each difference survives without the truncated replies."""
    rows: list[str] = []
    for r in results:
        a, b = arms[r.comparison.a], arms[r.comparison.b]
        truncated = a.truncated | b.truncated
        kept = (a.scores.keys() & b.scores.keys()) - truncated
        if not truncated or not kept:
            continue
        right_a, right_b = _paired(a.scores, b.scores, set(kept))
        counts = paired_counts(right_a, right_b)
        interval = paired_bootstrap_difference(right_a, right_b)
        rows.append(
            f"| `{r.comparison.a}` | `{r.comparison.b}` | {counts.n} | {counts.only_a} | "
            f"{counts.only_b} | {_pp(counts.difference)} | "
            f"{_pp(interval.low)} to {_pp(interval.high)} |"
        )
    if not rows:
        return []
    return [
        "## Without truncated replies",
        "",
        "Not a planned test: a check of how much of each difference is replies cut off at",
        "the token limit. Items where either arm's reply was truncated are left out.",
        "",
        "| A | B | Items | Only A right | Only B right | Difference | 95% CI (bootstrap) |",
        "|---|---|---:|---:|---:|---:|---|",
        *rows,
    ]


def error_counts(items: Sequence[Item], arm: ArmReplies) -> Counter[ErrorKind]:
    """An arm's wrong answers by kind."""
    kinds = (
        classify(item.answer, arm.scores[item.id], item.id in arm.truncated)
        for item in items
        if item.id in arm.scores
    )
    return Counter(kind for kind in kinds if kind is not None)


def _error_table(items: Sequence[Item], arms: dict[str, ArmReplies]) -> list[str]:
    counts = {name: error_counts(items, arm) for name, arm in arms.items()}
    lines = [
        "## Kinds of wrong answer",
        "",
        "A description made after reading the replies, not a test. A reply cut off at the",
        "token limit is counted as that, whatever else is wrong with it.",
        "",
        "| Kind | " + " | ".join(f"`{name}`" for name in arms) + " |",
        "|---|" + "---:|" * len(arms),
    ]
    for kind, label in ERROR_KINDS.items():
        if any(c[kind] for c in counts.values()):
            lines.append(f"| {label} | " + " | ".join(str(c[kind]) for c in counts.values()) + " |")
    lines.append("| all wrong | " + " | ".join(str(c.total()) for c in counts.values()) + " |")
    return lines


def _slice_accuracy(
    items: Sequence[Item], a: Scores, b: Scores
) -> list[tuple[str, int, float, float]]:
    """Label, items, accuracy of A and accuracy of B for every slice that has items."""
    rows: list[tuple[str, int, float, float]] = []
    for label, belongs in _SLICES:
        right_a, right_b = _paired(a, b, {item.id for item in items if belongs(item)})
        if right_a:
            n = len(right_a)
            rows.append((label, n, sum(right_a) / n, sum(right_b) / n))
    return rows


def render(items: Sequence[Item], arms: dict[str, ArmReplies]) -> str:
    answered = {name: arm for name, arm in arms.items() if arm.scores}
    if not answered:
        return "No recorded replies found.\n"
    all_scores = {name: arm.scores for name, arm in answered.items()}
    results = compare(all_scores)
    sections = [
        _arm_table([summarise_arm(name, arm) for name, arm in answered.items()]),
        _comparison_table(results),
        *(
            _slice_table(
                r.comparison, items, all_scores[r.comparison.a], all_scores[r.comparison.b]
            )
            for r in results
        ),
        _truncation_table(results, answered),
        _error_table(items, answered),
    ]
    return "\n\n".join("\n".join(section) for section in sections if section) + "\n"


def report_data(items: Sequence[Item], arms: dict[str, ArmReplies]) -> ReportData:
    """The numbers behind the figures, computed the same way as the tables."""
    from llm_ab.plots import ArmAccuracy, Difference, ReportData, SliceAccuracy

    answered = {name: arm for name, arm in arms.items() if arm.scores}
    all_scores = {name: arm.scores for name, arm in answered.items()}
    results = compare(all_scores)
    summaries = [summarise_arm(name, arm) for name, arm in answered.items()]
    # The slices are worth drawing for a comparison of two prompts, not for the A/A pair.
    contrast = next(
        (
            r.comparison
            for r in results
            if ARMS[r.comparison.a].prompt != ARMS[r.comparison.b].prompt
        ),
        None,
    )
    slices = (
        _slice_accuracy(items, all_scores[contrast.a], all_scores[contrast.b]) if contrast else []
    )
    return ReportData(
        arms=[ArmAccuracy(s.name, s.accuracy, s.interval) for s in summaries],
        differences=[
            Difference(r.comparison.a, r.comparison.b, r.counts, r.interval) for r in results
        ],
        slices=[SliceAccuracy(label, a, b) for label, _, a, b in slices],
        slice_arms=(contrast.a, contrast.b) if contrast else None,
        errors={name: error_counts(items, arm) for name, arm in answered.items()},
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Report on the recorded replies.")
    parser.add_argument("--items", type=Path, default=Path("data/items.jsonl"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, help="write the report here instead of printing it")
    parser.add_argument("--plots", type=Path, help="write the figures into this directory")
    args = parser.parse_args()

    items = load_items(args.items)
    arms = {name: load_arm(args.runs, name, items) for name in ARMS}
    text = render(items, arms)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.plots:
        from llm_ab.plots import save_all

        save_all(report_data(items, arms), args.plots)


if __name__ == "__main__":
    main()
