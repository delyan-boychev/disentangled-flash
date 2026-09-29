"""Create the release figure for the kernel benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

GIB = 1024**3
SERIES = (
    ("base", "Hugging Face eager", "#555555", "o", "--", 2.0),
    ("torch", "DF PyTorch", "#3B6FB6", "s", ":", 2.0),
    ("triton", "DF Triton", "#D1495B", "o", "-", 3.0),
    ("flashdeberta", "FlashDeBERTa", "#2A9D8F", "D", "-.", 2.0),
)
PASS_TITLES = {
    "forward": "Forward",
    "forward_backward": "Forward + backward",
}


def load_results(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path} does not contain benchmark results")
    frame = pd.DataFrame.from_records(results)
    required = {
        "status",
        "implementation",
        "pass",
        "dropout",
        "sequence_length",
        "p50_ms",
        "effective_attention_tflops",
        "peak_allocated_bytes",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    return frame, report.get("configuration", {})


def _length_label(value: int) -> str:
    integer = int(value)
    return f"{integer // 1024}k" if integer >= 1024 else str(integer)


def _memory_label(value: float, _position: int) -> str:
    return f"{value:g}"


def _plot_series(axis: Any, panel: pd.DataFrame, metric: str) -> None:
    for implementation, label, color, marker, linestyle, linewidth in SERIES:
        rows = panel[panel["implementation"].eq(implementation)].sort_values("sequence_length")
        if rows.empty:
            continue
        axis.plot(
            rows["sequence_length"],
            rows[metric],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=5.5,
            markeredgewidth=0,
        )


def _style_axis(axis: Any, lengths: list[int], tick_labels: list[str]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(lengths, labels=tick_labels)
    axis.grid(axis="y", color="#D7DADF", linewidth=0.8)
    axis.grid(axis="x", color="#ECEEF1", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#4C5158")
    axis.spines["bottom"].set_color("#4C5158")
    axis.tick_params(colors="#30343A", length=4, width=0.8)


def _make_metric_figure(
    usable: pd.DataFrame,
    configuration: dict[str, Any],
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_stem: str,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.85), sharey=metric == "peak_memory_gib")
    figure.subplots_adjust(left=0.085, right=0.985, top=0.70, bottom=0.24, wspace=0.23)

    lengths = sorted(int(value) for value in usable["sequence_length"].unique())
    tick_labels = [_length_label(length) for length in lengths]

    for index, pass_mode in enumerate(("forward", "forward_backward")):
        axis = axes[index]
        panel = usable[usable["pass"].eq(pass_mode)]
        if panel.empty:
            continue
        _plot_series(axis, panel, metric)
        _style_axis(axis, lengths, tick_labels)
        axis.set_title(f"({chr(97 + index)})  {PASS_TITLES[pass_mode]}", loc="left", pad=11)
        axis.set_xlabel("Sequence length")
        if metric == "effective_attention_tflops":
            axis.set_ylim(bottom=0)
        else:
            axis.set_yscale("log", base=2)
            axis.set_ylim(0.25, 32)
            axis.set_yticks((0.25, 0.5, 1, 2, 4, 8, 16, 32))
            axis.yaxis.set_major_formatter(FuncFormatter(_memory_label))

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        ncol=4,
        frameon=False,
        handlelength=3.2,
        columnspacing=2.0,
    )

    gpu = str(usable["gpu"].iloc[0]) if "gpu" in usable else "CUDA GPU"
    figure.suptitle(f"{title} ({gpu})", y=0.965, fontsize=18, fontweight="bold")
    figure.text(
        0.5,
        0.045,
        (
            f"{configuration.get('dtype', 'bf16').upper()}   ·   head dimension "
            f"{configuration.get('head_dim', 64)}   ·   dropout 0   ·   "
            f"batch × length = {configuration.get('total_tokens', 16384):,} tokens   ·   "
            f"median of {configuration.get('iters', 25)} iterations"
        ),
        ha="center",
        fontsize=9.5,
        color="#555A61",
    )

    png = output_dir / f"{output_stem}.png"
    pdf = output_dir / f"{output_stem}.pdf"
    figure.savefig(png, dpi=dpi, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    return png, pdf


def make_release_figures(
    frame: pd.DataFrame,
    configuration: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, ...]:
    usable = frame[frame["status"].eq("ok") & frame["dropout"].eq(0.0)].copy()
    if usable.empty:
        raise ValueError("no successful dropout-zero benchmark rows to plot")
    usable["peak_memory_gib"] = usable["peak_allocated_bytes"] / GIB

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.edgecolor": "#4C5158",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    throughput = _make_metric_figure(
        usable,
        configuration,
        metric="effective_attention_tflops",
        title="Attention kernel speed",
        ylabel="Speed (TFLOP/s)",
        output_stem="kernel_throughput_h200",
        output_dir=output_dir,
        dpi=dpi,
    )
    memory = _make_metric_figure(
        usable,
        configuration,
        metric="peak_memory_gib",
        title="Attention peak memory",
        ylabel="Peak memory (GiB, log₂ scale)",
        output_stem="kernel_memory_h200",
        output_dir=output_dir,
        dpi=dpi,
    )
    return (*throughput, *memory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/kernel"))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    frame, configuration = load_results(args.input)
    for output in make_release_figures(frame, configuration, args.output_dir, args.dpi):
        print(output.resolve())


if __name__ == "__main__":
    main()
