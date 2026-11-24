from __future__ import annotations

import numpy as np
import pytest

from llm_ab.stats import (
    PairedCounts,
    holm,
    mcnemar_exact,
    mcnemar_sample_size,
    paired_bootstrap_difference,
    paired_counts,
    wilson_interval,
)


def test_paired_counts_sorts_items_into_four_cells() -> None:
    a = [True, True, False, False, True]
    b = [True, False, True, False, True]

    assert paired_counts(a, b) == PairedCounts(both=2, only_a=1, only_b=1, neither=1)


def test_paired_counts_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        paired_counts([True], [True, False])


def test_difference_is_b_minus_a() -> None:
    counts = PairedCounts(both=50, only_a=5, only_b=15, neither=30)

    assert counts.difference == pytest.approx(0.10)


def test_wilson_interval_matches_published_value() -> None:
    # 8 of 10 at 95%: the standard worked example, (0.4902, 0.9433).
    interval = wilson_interval(8, 10)

    assert interval.low == pytest.approx(0.4902, abs=1e-4)
    assert interval.high == pytest.approx(0.9433, abs=1e-4)


def test_wilson_interval_has_width_at_perfect_accuracy() -> None:
    interval = wilson_interval(20, 20)

    assert interval.high == 1.0
    assert interval.low < 0.9


def test_mcnemar_exact_matches_hand_calculation() -> None:
    # 10 discordant items, 1 against 9: 2 * (C(10,0) + C(10,1)) / 2^10 = 22/1024.
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)


def test_mcnemar_exact_is_symmetric() -> None:
    assert mcnemar_exact(3, 11) == mcnemar_exact(11, 3)


def test_mcnemar_exact_is_one_without_evidence() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(4, 4) == 1.0


def test_bootstrap_interval_is_zero_for_identical_variants() -> None:
    outcomes = [True, False, True, True, False] * 20

    interval = paired_bootstrap_difference(outcomes, outcomes)

    assert (interval.low, interval.high) == (0.0, 0.0)


def test_bootstrap_interval_excludes_zero_for_a_clear_improvement() -> None:
    a = [False] * 60 + [True] * 140
    b = [True] * 200

    interval = paired_bootstrap_difference(a, b)

    assert interval.low > 0
    assert interval.contains(0.30)


def test_bootstrap_interval_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    a = [bool(x) for x in rng.random(150) < 0.6]
    b = [bool(x) for x in rng.random(150) < 0.7]

    assert paired_bootstrap_difference(a, b, seed=7) == paired_bootstrap_difference(a, b, seed=7)


def test_holm_matches_hand_calculation() -> None:
    # Sorted: 0.01*3, 0.03*2, 0.04*1 -> 0.03, 0.06, 0.04; the last is raised to 0.06.
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one() -> None:
    assert holm([0.6, 0.9]) == [1.0, 1.0]


def test_sample_size_matches_hand_calculation() -> None:
    # psi=0.3, delta=0.1: (1.960*sqrt(0.3) + 0.842*sqrt(0.29))^2 / 0.01 = 233.1
    assert mcnemar_sample_size(p_only_a=0.10, p_only_b=0.20) == 234


def test_sample_size_falls_when_variants_disagree_less() -> None:
    same_gap_many_disagreements = mcnemar_sample_size(p_only_a=0.20, p_only_b=0.30)
    same_gap_few_disagreements = mcnemar_sample_size(p_only_a=0.02, p_only_b=0.12)

    assert same_gap_few_disagreements < same_gap_many_disagreements


def test_planned_sample_size_delivers_the_planned_power() -> None:
    p_only_a, p_only_b = 0.10, 0.20
    n = mcnemar_sample_size(p_only_a, p_only_b, alpha=0.05, power=0.8)
    rng = np.random.default_rng(0)
    trials = 2000

    rejections = 0
    for _ in range(trials):
        only_a, only_b, _rest = rng.multinomial(n, [p_only_a, p_only_b, 1 - p_only_a - p_only_b])
        rejections += mcnemar_exact(int(only_a), int(only_b)) < 0.05

    # The formula is a normal approximation and the exact test is slightly
    # conservative, so the simulated power sits a little under the 0.8 target.
    assert 0.74 <= rejections / trials <= 0.86
