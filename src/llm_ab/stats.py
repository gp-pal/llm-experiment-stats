"""Statistics for a paired LLM experiment with a right-or-wrong outcome.

Both variants answer the same items, so the outcomes are paired. Items where both
variants agree carry no information about which one is better; only the discordant
items do. Every function here is built on that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class PairedCounts:
    """How two variants did on the same items."""

    both: int
    only_a: int
    only_b: int
    neither: int

    @property
    def n(self) -> int:
        return self.both + self.only_a + self.only_b + self.neither

    @property
    def difference(self) -> float:
        """Accuracy of B minus accuracy of A."""
        return (self.only_b - self.only_a) / self.n


def paired_counts(a: Sequence[bool], b: Sequence[bool]) -> PairedCounts:
    if len(a) != len(b):
        raise ValueError(f"paired outcomes differ in length: {len(a)} and {len(b)}")
    if not a:
        raise ValueError("no items to compare")
    pairs = list(zip(a, b, strict=True))
    return PairedCounts(
        both=sum(x and y for x, y in pairs),
        only_a=sum(x and not y for x, y in pairs),
        only_b=sum(y and not x for x, y in pairs),
        neither=sum(not x and not y for x, y in pairs),
    )


def _z(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Confidence interval for one variant's accuracy.

    Wilson rather than the textbook p +/- z*se: the textbook interval collapses to
    zero width at 0% or 100% accuracy, which is where small evals often land.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = _z(confidence)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return Interval(max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value.

    Under the null hypothesis a discordant item is equally likely to favour either
    variant, so the count favouring one of them is Binomial(discordant, 0.5).
    """
    if only_a < 0 or only_b < 0:
        raise ValueError("counts cannot be negative")
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (1 << discordant)
    return min(1.0, 2 * tail)


def paired_bootstrap_difference(
    a: Sequence[bool],
    b: Sequence[bool],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for accuracy(B) - accuracy(A).

    Items are resampled, not individual answers: a resampled item brings both of its
    answers with it, which keeps the pairing intact.
    """
    counts = paired_counts(a, b)
    per_item = np.asarray(b, dtype=np.int8) - np.asarray(a, dtype=np.int8)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, counts.n, size=(n_resamples, counts.n))
    differences = per_item[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(differences, [tail, 1 - tail])
    return Interval(float(low), float(high))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, in the order given.

    Needed as soon as one experiment makes several comparisons: with enough of them,
    one will look significant by chance.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def mcnemar_sample_size(
    p_only_a: float,
    p_only_b: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Items needed to detect a paired difference (Connor, 1987).

    `p_only_a` and `p_only_b` are the expected shares of items that only one variant
    gets right. The answer depends on how often the variants disagree, not only on
    the accuracy gap: two variants that fail on the same items need far fewer items
    than two that fail on different ones.
    """
    if p_only_a < 0 or p_only_b < 0 or p_only_a + p_only_b > 1:
        raise ValueError("discordant shares must be non-negative and sum to at most 1")
    delta = p_only_b - p_only_a
    if delta == 0:
        raise ValueError("no difference to detect")
    psi = p_only_a + p_only_b
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_beta = NormalDist().inv_cdf(power)
    n = (z_alpha * math.sqrt(psi) + z_beta * math.sqrt(psi - delta**2)) ** 2 / delta**2
    return math.ceil(n)
