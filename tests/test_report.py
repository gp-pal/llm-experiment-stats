from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_ab.errors import group_counts
from llm_ab.items import Item, generate_items
from llm_ab.plots import save_all
from llm_ab.records import run_path
from llm_ab.report import compare, error_counts, load_arm, render, report_data, summarise_arm


def _record(runs: Path, arm: str, items: list[Item], wrong: set[int], model: str = "m") -> None:
    """Record replies for `arm`: correct everywhere except at the `wrong` positions."""
    runs.mkdir(parents=True, exist_ok=True)
    with run_path(runs, arm).open("a", encoding="utf-8") as file:
        for position, item in enumerate(items):
            answer = item.answer + 1 if position in wrong else item.answer
            record = {"item_id": item.id, "model": model, "reply": f"ATSAKYMAS: {answer}"}
            file.write(json.dumps(record) + "\n")


def test_arm_summary_counts_correct_replies_and_failure_kinds(tmp_path: Path) -> None:
    items = generate_items(10)
    _record(tmp_path, "baseline", items[:9], wrong={0, 1})
    with run_path(tmp_path, "baseline").open("a", encoding="utf-8") as file:
        record = {"item_id": items[9].id, "model": "m", "reply": "Nežinau."}
        file.write(json.dumps(record) + "\n")

    arm = summarise_arm("baseline", load_arm(tmp_path, "baseline", items))

    assert (arm.n, arm.correct, arm.model) == (10, 7, "m")
    assert arm.statuses == {"correct": 7, "wrong_number": 2, "no_marker": 1}


def test_comparison_uses_only_items_both_arms_answered(tmp_path: Path) -> None:
    items = generate_items(40)
    _record(tmp_path, "baseline", items, wrong=set(range(12)))
    _record(tmp_path, "decompose", items[:30], wrong={0, 29})
    scores = {arm: load_arm(tmp_path, arm, items).scores for arm in ("baseline", "decompose")}

    (result,) = compare(scores)

    assert result.counts.n == 30
    assert (result.counts.only_a, result.counts.only_b) == (1, 11)
    assert result.counts.difference == pytest.approx(10 / 30)
    assert result.p_value < 0.05
    assert result.interval.low > 0


def test_holm_adjustment_is_applied_across_the_planned_comparisons(tmp_path: Path) -> None:
    items = generate_items(40)
    _record(tmp_path, "baseline", items, wrong=set(range(12)))
    _record(tmp_path, "baseline-repeat", items, wrong=set(range(1, 13)))
    _record(tmp_path, "decompose", items, wrong={0})
    scores = {
        arm: load_arm(tmp_path, arm, items).scores
        for arm in ("baseline", "baseline-repeat", "decompose")
    }

    a_a, a_b = compare(scores)

    assert a_a.p_value == 1.0
    assert a_b.p_adjusted == pytest.approx(2 * a_b.p_value)


def test_mixing_models_within_an_arm_is_refused(tmp_path: Path) -> None:
    items = generate_items(4)
    _record(tmp_path, "baseline", items[:2], wrong=set(), model="small")
    _record(tmp_path, "baseline", items[2:], wrong=set(), model="large")

    with pytest.raises(ValueError, match="mixes models"):
        load_arm(tmp_path, "baseline", items)


def test_report_states_the_verdict(tmp_path: Path) -> None:
    items = generate_items(40)
    _record(tmp_path, "baseline", items, wrong=set(range(12)))
    _record(tmp_path, "baseline-repeat", items, wrong=set(range(1, 13)))
    _record(tmp_path, "decompose", items, wrong={0})
    arms = {
        name: load_arm(tmp_path, name, items)
        for name in ("baseline", "baseline-repeat", "decompose")
    }

    text = render(items, arms)

    assert "| `baseline` | m | 40 | 28 | 70.0% |" in text
    assert "no difference detected" in text
    assert "`decompose` is better" in text
    assert "| millions | 20 |" in text
    assert "Without truncated replies" not in text


def test_report_without_replies_says_so(tmp_path: Path) -> None:
    arms = {"baseline": load_arm(tmp_path, "baseline", generate_items(2))}

    assert render(generate_items(2), arms).startswith("No recorded")


def test_a_tiny_p_value_is_shown_as_a_bound_not_as_zero() -> None:
    from llm_ab.report import _p

    assert _p(1e-9) == "< 0.0001"
    assert _p(0.0312) == "0.0312"


def test_truncated_replies_are_counted_and_the_difference_is_shown_without_them(
    tmp_path: Path,
) -> None:
    items = generate_items(20)
    _record(tmp_path, "baseline", items, wrong={0})
    _record(tmp_path, "decompose", items[:16], wrong={1, 2})
    with run_path(tmp_path, "decompose").open("a", encoding="utf-8") as file:
        for item in items[16:]:
            record = {"item_id": item.id, "model": "m", "reply": "", "finish_reason": "length"}
            file.write(json.dumps(record) + "\n")
    arms = {name: load_arm(tmp_path, name, items) for name in ("baseline", "decompose")}

    text = render(items, arms)

    assert summarise_arm("decompose", arms["decompose"]).truncated == 4
    # All 20 items: baseline 19 right, decompose 14 right.
    assert "| `baseline` | `decompose` | 20 | 6 | 1 | -25.0 pp |" in text
    # The 16 items neither arm truncated: baseline 15 right, decompose 14 right.
    assert "| `baseline` | `decompose` | 16 | 2 | 1 | -6.2 pp |" in text


def _three_arms(tmp_path: Path, items: list[Item]) -> None:
    _record(tmp_path, "baseline", items, wrong=set(range(12)))
    _record(tmp_path, "baseline-repeat", items, wrong=set(range(1, 13)))
    _record(tmp_path, "decompose", items, wrong={0})


def test_wrong_answers_are_reported_by_kind(tmp_path: Path) -> None:
    items = generate_items(6)
    with run_path(tmp_path, "baseline").open("a", encoding="utf-8") as file:
        replies = [
            (f"ATSAKYMAS: {items[0].answer}", "stop"),
            (f"ATSAKYMAS: {items[1].answer + 1}", "stop"),
            (f"ATSAKYMAS: {sum(group_counts(items[2].answer))}", "stop"),
            ("", "length"),
        ]
        for item, (reply, finish) in zip(items, replies, strict=False):
            record = {"item_id": item.id, "model": "m", "reply": reply, "finish_reason": finish}
            file.write(json.dumps(record) + "\n")
    arms = {"baseline": load_arm(tmp_path, "baseline", items)}

    assert error_counts(items, arms["baseline"]) == {
        "misread": 1,
        "added_groups": 1,
        "truncated": 1,
    }
    text = render(items, arms)
    assert "| added group counts, not place values | 1 |" in text
    assert "| all wrong | 3 |" in text


def test_every_figure_is_written_for_a_light_and_a_dark_page(tmp_path: Path) -> None:
    items = generate_items(40)
    _three_arms(tmp_path, items)
    arms = {
        name: load_arm(tmp_path, name, items)
        for name in ("baseline", "baseline-repeat", "decompose")
    }

    data = report_data(items, arms)
    written = save_all(data, tmp_path / "figures")

    assert data.slice_arms == ("baseline", "decompose")
    assert {path.name for path in written} == {
        f"{name}-{mode}.png"
        for name in (
            "accuracy",
            "differences",
            "paired-outcomes",
            "errors",
            "slices",
            "sample-size",
        )
        for mode in ("light", "dark")
    }
    assert all(path.stat().st_size > 0 for path in written)
