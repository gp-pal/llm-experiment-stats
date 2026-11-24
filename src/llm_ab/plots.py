"""Figures for the report, each drawn for a light and a dark page.

An arm keeps its colour in every figure, so a reader who has learned that the
baseline is blue is never asked to relearn it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from llm_ab.errors import ERROR_KINDS, ErrorKind
from llm_ab.stats import Interval, PairedCounts, mcnemar_sample_size

Mode = Literal["light", "dark"]
MODES: tuple[Mode, ...] = ("light", "dark")


@dataclass(frozen=True)
class Theme:
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    #: Fills for the outcomes that say nothing about which arm is better.
    agree_right: str
    agree_wrong: str
    arms: tuple[str, str, str]
    #: One hue, three steps, for an ordered quantity.
    ramp: tuple[str, str, str]


THEMES: dict[Mode, Theme] = {
    "light": Theme(
        surface="#fcfcfb",
        ink="#0b0b0b",
        secondary="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        agree_right="#e1e0d9",
        agree_wrong="#898781",
        arms=("#2a78d6", "#eb6834", "#1baf7a"),
        ramp=("#86b6ef", "#2a78d6", "#104281"),
    ),
    "dark": Theme(
        surface="#1a1a19",
        ink="#f0efec",
        secondary="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        agree_right="#383835",
        agree_wrong="#898781",
        arms=("#3987e5", "#d95926", "#199e70"),
        ramp=("#184f95", "#3987e5", "#9ec5f4"),
    ),
}

# Colour follows the arm, not its position in a particular figure.
_ARM_SLOT = {"baseline": 0, "decompose": 1, "baseline-repeat": 2}


def arm_colour(theme: Theme, arm: str) -> str:
    return theme.arms[_ARM_SLOT.get(arm, len(_ARM_SLOT) - 1)]


@dataclass(frozen=True)
class ArmAccuracy:
    name: str
    accuracy: float
    interval: Interval


@dataclass(frozen=True)
class Difference:
    a: str
    b: str
    counts: PairedCounts
    interval: Interval


@dataclass(frozen=True)
class SliceAccuracy:
    label: str
    a: float
    b: float


@dataclass(frozen=True)
class ReportData:
    """Everything the figures show, already computed by the report."""

    arms: Sequence[ArmAccuracy]
    differences: Sequence[Difference]
    #: Slices of the first comparison whose two arms use different prompts.
    slices: Sequence[SliceAccuracy]
    slice_arms: tuple[str, str] | None
    errors: Mapping[str, Mapping[ErrorKind, int]]


def _figure(theme: Theme, width: float, height: float) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.axis)
    ax.tick_params(colors=theme.muted, length=0, labelsize=9)
    ax.xaxis.label.set_color(theme.secondary)
    ax.xaxis.label.set_fontsize(9)
    return fig, ax


def _title(fig: Figure, theme: Theme, title: str, subtitle: str) -> None:
    # Not suptitle: tight_layout would make room for it a second time.
    fig.text(0.02, 0.98, title, ha="left", va="top", color=theme.ink, fontsize=12.5)
    fig.text(
        0.02,
        0.98 - 0.26 / fig.get_figheight(),
        subtitle,
        ha="left",
        va="top",
        color=theme.secondary,
        fontsize=9.5,
    )


def _layout(fig: Figure, bottom_inches: float = 0.0) -> None:
    """Fit the plot between the title block and an optional legend strip below it."""
    height = fig.get_figheight()
    fig.tight_layout(rect=(0, bottom_inches / height, 1, 1 - 0.6 / height))


def _legend_below(fig: Figure, theme: Theme, handles: Sequence[Patch | Line2D], ncols: int) -> None:
    legend = fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.0),
        ncols=ncols,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.4,
    )
    _legend_ink(legend, theme)


def _legend_ink(legend: Legend, theme: Theme) -> None:
    for label in legend.get_texts():
        label.set_color(theme.secondary)


def _row_labels(ax: Axes, theme: Theme, labels: Sequence[str]) -> None:
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=theme.ink, fontsize=10)
    ax.set_ylim(len(labels) - 0.5, -0.5)


def _interval_mark(
    ax: Axes, theme: Theme, row: float, value: float, interval: Interval, colour: str
) -> None:
    ax.plot(
        [interval.low, interval.high],
        [row, row],
        color=colour,
        linewidth=2,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(
        value,
        row,
        "o",
        color=colour,
        markersize=9,
        markeredgecolor=theme.surface,
        markeredgewidth=2,
        zorder=3,
    )


def plot_accuracy(data: ReportData, theme: Theme) -> Figure:
    fig, ax = _figure(theme, 7.2, 1.0 + 0.5 * len(data.arms))
    for row, arm in enumerate(data.arms):
        interval = Interval(100 * arm.interval.low, 100 * arm.interval.high)
        _interval_mark(ax, theme, row, 100 * arm.accuracy, interval, arm_colour(theme, arm.name))
        ax.annotate(
            f"{100 * arm.accuracy:.1f}%",
            (interval.high, row),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=theme.ink,
            fontsize=10,
        )
    _row_labels(ax, theme, [arm.name for arm in data.arms])
    low = min(arm.interval.low for arm in data.arms)
    ax.set_xlim(10 * int(10 * low) - 5, 100)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(axis="x", color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    _title(
        fig,
        theme,
        "Accuracy per arm",
        "Share of items answered correctly, with 95% Wilson intervals",
    )
    _layout(fig)
    return fig


def plot_differences(data: ReportData, theme: Theme) -> Figure:
    fig, ax = _figure(theme, 7.2, 1.2 + 0.7 * len(data.differences))
    ax.axvline(0, color=theme.muted, linewidth=1, zorder=1)
    for row, d in enumerate(data.differences):
        interval = Interval(100 * d.interval.low, 100 * d.interval.high)
        value = 100 * d.counts.difference
        _interval_mark(ax, theme, row, value, interval, arm_colour(theme, d.b))
        ax.annotate(
            f"{value:+.1f} pp",
            (interval.low, row),
            xytext=(-8, 0),
            ha="right",
            textcoords="offset points",
            va="center",
            color=theme.ink,
            fontsize=10,
        )
    _row_labels(ax, theme, [f"{d.b}\nvs {d.a}" for d in data.differences])
    low = min(100 * d.interval.low for d in data.differences)
    high = max(100 * d.interval.high for d in data.differences)
    ax.set_xlim(min(low, 0) - 9, max(high, 0) + 3)
    ax.grid(axis="x", color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Difference in accuracy, percentage points")
    _title(
        fig,
        theme,
        "Is the difference real?",
        "Accuracy difference against the baseline, with 95% paired bootstrap intervals",
    )
    _layout(fig)
    return fig


def plot_paired_outcomes(data: ReportData, theme: Theme) -> Figure:
    """Each comparison as one bar of all its items, split by who got the item right."""
    fig, ax = _figure(theme, 7.2, 1.85 + 0.7 * len(data.differences))
    gap = 0.6  # in items: the sliver of surface that separates two segments
    for row, d in enumerate(data.differences):
        c = d.counts
        segments = [
            (c.both, theme.agree_right, theme.ink),
            (c.only_a, arm_colour(theme, d.a), "#ffffff"),
            (c.only_b, arm_colour(theme, d.b), "#ffffff"),
            (c.neither, theme.agree_wrong, "#ffffff"),
        ]
        left = 0.0
        for size, colour, text in segments:
            if size:
                ax.barh(row, max(size - gap, 0.2), left=left + gap / 2, height=0.5, color=colour)
                if size >= 0.045 * c.n:
                    ax.text(
                        left + size / 2,
                        row,
                        str(size),
                        ha="center",
                        va="center",
                        color=text,
                        fontsize=9.5,
                    )
            left += size
    _row_labels(
        ax,
        theme,
        [
            f"{d.a} vs {d.b}\ndisagree on {d.counts.only_a + d.counts.only_b}"
            for d in data.differences
        ],
    )
    ax.set_xlim(0, max(d.counts.n for d in data.differences))
    ax.set_xlabel("Items")
    ax.spines["bottom"].set_visible(False)
    arms = list(dict.fromkeys(name for d in data.differences for name in (d.a, d.b)))
    handles = [
        Patch(color=theme.agree_right, label="both right"),
        *(Patch(color=arm_colour(theme, name), label=f"only {name} right") for name in arms),
        Patch(color=theme.agree_wrong, label="both wrong"),
    ]
    _legend_below(fig, theme, handles, ncols=3)
    _title(
        fig,
        theme,
        "Only the disagreements carry information",
        "How each pair of arms did on the same items; the coloured segments decide the test",
    )
    _layout(fig, bottom_inches=0.58)
    return fig


def plot_errors(data: ReportData, theme: Theme) -> Figure:
    """Wrong answers by kind: a row per kind, a bar per arm."""
    kinds = [k for k in ERROR_KINDS if any(counts.get(k) for counts in data.errors.values())]
    names = list(data.errors)
    step = 0.26
    fig, ax = _figure(theme, 7.2, 1.2 + 0.36 * len(kinds) * len(names))
    for row, kind in enumerate(kinds):
        for position, name in enumerate(names):
            y = row + (position - (len(names) - 1) / 2) * step
            count = data.errors[name].get(kind, 0)
            ax.barh(y, count, height=step - 0.05, color=arm_colour(theme, name))
            ax.annotate(
                str(count),
                (count, y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                color=theme.ink,
                fontsize=9.5,
            )
    _row_labels(ax, theme, [ERROR_KINDS[kind] for kind in kinds])
    ax.set_xlim(0, 1.12 * max(max(c.values(), default=0) for c in data.errors.values()))
    ax.set_xlabel("Wrong answers")
    ax.grid(axis="x", color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    handles = [Patch(color=arm_colour(theme, name), label=name) for name in names]
    legend = ax.legend(
        handles=handles, loc="lower right", frameon=False, fontsize=9, handlelength=1.2
    )
    _legend_ink(legend, theme)
    _title(
        fig,
        theme,
        "What went wrong",
        "Wrong answers by kind; a description made after reading the replies, not a test",
    )
    _layout(fig)
    return fig


def plot_slices(data: ReportData, theme: Theme) -> Figure | None:
    """Where the two prompts differ, a row per slice: one dot per arm."""
    if not data.slices or data.slice_arms is None:
        return None
    a, b = data.slice_arms
    fig, ax = _figure(theme, 7.2, 1.5 + 0.38 * len(data.slices))
    for row, s in enumerate(data.slices):
        ax.plot([100 * s.a, 100 * s.b], [row, row], color=theme.axis, linewidth=2, zorder=1)
        for value, name in ((s.a, a), (s.b, b)):
            ax.plot(
                100 * value,
                row,
                "o",
                color=arm_colour(theme, name),
                markersize=9,
                markeredgecolor=theme.surface,
                markeredgewidth=2,
                zorder=3,
            )
        ax.annotate(
            f"{100 * (s.b - s.a):+.0f} pp",
            (100 * max(s.a, s.b), row),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=theme.ink,
            fontsize=9.5,
        )
    _row_labels(ax, theme, [s.label for s in data.slices])
    low = min(min(s.a, s.b) for s in data.slices)
    ax.set_xlim(10 * int(10 * low) - 5, 100)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(axis="x", color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=arm_colour(theme, name), label=name)
        for name in (a, b)
    ]
    _legend_below(fig, theme, handles, ncols=2)
    _title(
        fig,
        theme,
        "Where the two prompts differ",
        "Accuracy by slice of the eval set; descriptive, these were not planned as tests",
    )
    _layout(fig, bottom_inches=0.32)
    return fig


def plot_sample_size(theme: Theme, items: int | None = None) -> Figure:
    """Items needed against the gap to detect, for three rates of disagreement."""
    fig, ax = _figure(theme, 7.2, 3.6)
    gaps = [g / 2 for g in range(6, 31)]  # 3 to 15 percentage points
    for colour, discordant in zip(theme.ramp, (0.15, 0.25, 0.35), strict=True):
        needed = [
            mcnemar_sample_size((discordant - g / 100) / 2, (discordant + g / 100) / 2)
            for g in gaps
        ]
        ax.plot(gaps, needed, color=colour, linewidth=2, solid_capstyle="round")
        ax.annotate(
            f"arms disagree on {discordant:.0%}",
            (gaps[-1], needed[-1]),
            xytext=(7, 0),
            va="center",
            textcoords="offset points",
            color=theme.secondary,
            fontsize=9,
        )
    if items:
        ax.axhline(items, color=theme.muted, linewidth=1)
        ax.annotate(
            f"{items} items",
            (gaps[0], items),
            xytext=(6, 4),
            textcoords="offset points",
            color=theme.secondary,
            fontsize=9,
        )
    ax.set_yscale("log")
    ax.set_yticks([50, 100, 200, 500, 1000, 2000, 5000])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.minorticks_off()
    # Room on the right for the labels that sit at the line ends.
    ax.set_xlim(gaps[0], gaps[-1] + 3.6)
    ax.set_xticks(range(4, 16, 2))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f} pp")
    ax.set_xlabel("Accuracy gap to detect")
    ax.grid(color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    _title(
        fig,
        theme,
        "How many items does an experiment need?",
        "Items needed for 80% power at alpha 0.05, by how often the two arms disagree",
    )
    _layout(fig)
    return fig


def save_all(data: ReportData, out_dir: Path) -> list[Path]:
    """Write every figure for both modes as `<name>-<mode>.png`. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = max((d.counts.n for d in data.differences), default=None)
    figures: dict[str, Callable[[Theme], Any]] = {
        "accuracy": lambda theme: plot_accuracy(data, theme),
        "differences": lambda theme: plot_differences(data, theme) if data.differences else None,
        "paired-outcomes": lambda theme: (
            plot_paired_outcomes(data, theme) if data.differences else None
        ),
        "errors": lambda theme: plot_errors(data, theme) if any(data.errors.values()) else None,
        "slices": lambda theme: plot_slices(data, theme),
        "sample-size": lambda theme: plot_sample_size(theme, items),
    }
    written: list[Path] = []
    for mode in MODES:
        for name, draw in figures.items():
            fig = draw(THEMES[mode])
            if fig is None:
                continue
            path = out_dir / f"{name}-{mode}.png"
            fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
            plt.close(fig)
            written.append(path)
    return written
