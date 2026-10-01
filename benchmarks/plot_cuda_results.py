"""Create the release figure for the kernel benchmark."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

GIB = 1024**3
_PALETTE = sns.color_palette("colorblind")
SERIES = (
    ("base", "Hugging Face eager", _PALETTE[0], "o", (4, 2), 1.8),
    ("torch", "DF PyTorch", _PALETTE[2], "s", (1, 1.5), 1.8),
    ("flashdeberta", "FlashDeBERTa", _PALETTE[4], "D", (5, 2, 1, 2), 1.8),
    ("triton", "DF Triton", _PALETTE[3], "o", (), 2.8),
)
PASS_TITLES = {
    "forward": "Forward",
    "backward": "Backward",
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
        "sequence_length",
        "p50_ms",
        "batch_size",
        "head_dim",
        "incremental_peak_allocated_bytes",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    # Recompute from shape and time so older reports use the current FLOP count.
    frame["attention_tflops"] = frame.apply(_attention_tflops, axis=1)
    return frame, report.get("configuration", {})


def _attention_tflops(row: pd.Series) -> float:
    from benchmarks.benchmark_cuda import attention_flops, make_config

    if row["status"] != "ok":
        return math.nan
    flops = attention_flops(
        make_config(int(row["head_dim"])),
        int(row["batch_size"]),
        int(row["sequence_length"]),
        row["pass"],
    )
    return flops / (row["p50_ms"] * 1e9)


def _length_label(value: int) -> str:
    integer = int(value)
    return f"{integer // 1024}k" if integer >= 1024 else str(integer)


def _gpu_output_suffix(gpu: str) -> str:
    normalized = gpu.lower()
    if "h200" in normalized:
        return "h200"
    if "a6000" in normalized:
        return "a6000"
    if "rtx" in normalized and "6000" in normalized:
        return "rtx6000"
    suffix = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return suffix or "cuda"


def _memory_label(value: float, _position: int) -> str:
    return f"{value:g}"


def _memory_ticks(values: pd.Series) -> tuple[float, ...]:
    # Powers of two that bracket the data, so small incremental peaks stay visible.
    positive = values[values > 0]
    low = 2.0 ** math.floor(math.log2(positive.min())) if not positive.empty else 0.25
    high = 2.0 ** math.ceil(math.log2(values.max())) if not positive.empty else 32.0
    ticks = [low]
    while ticks[-1] < high or len(ticks) < 2:
        ticks.append(ticks[-1] * 2)
    return tuple(ticks)


def _plot_series(axis: Any, panel: pd.DataFrame, metric: str) -> None:
    for implementation, label, color, marker, dashes, linewidth in SERIES:
        rows = panel[panel["implementation"].eq(implementation)].sort_values("sequence_length")
        if rows.empty:
            continue
        (line,) = axis.plot(
            rows["sequence_length"],
            rows[metric],
            label=label,
            color=color,
            linewidth=linewidth,
            zorder=3 if implementation == "triton" else 2,
        )
        line.set_dashes(dashes or (None, None))
        axis.scatter(
            rows["sequence_length"],
            rows[metric],
            marker=marker,
            s=34 if implementation == "triton" else 26,
            facecolor=color,
            edgecolor=color,
            linewidth=1.4,
            zorder=4,
        )


def _style_axis(axis: Any, lengths: list[int], tick_labels: list[str]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(lengths, labels=tick_labels)
    axis.minorticks_off()
    axis.margins(x=0.04)
    sns.despine(ax=axis)


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
    passes = [mode for mode in PASS_TITLES if usable["pass"].eq(mode).any()]
    figure, axes = plt.subplots(
        1,
        len(passes),
        figsize=(4.6 * len(passes), 4.2),
        sharey=metric == "peak_memory_gib",
        squeeze=False,
    )
    axes = axes[0]
    timing = metric == "attention_tflops"

    lengths = sorted(int(value) for value in usable["sequence_length"].unique())
    tick_labels = [_length_label(length) for length in lengths]

    for index, pass_mode in enumerate(passes):
        axis = axes[index]
        panel = usable[usable["pass"].eq(pass_mode)]
        _plot_series(axis, panel, metric)
        _style_axis(axis, lengths, tick_labels)
        axis.set_title(PASS_TITLES[pass_mode], loc="left", fontweight="bold")
        axis.set_xlabel("Sequence length")
        if timing:
            axis.set_ylim(bottom=0)
        else:
            axis.set_yscale("log", base=2)
            ticks = _memory_ticks(usable[metric])
            axis.set_ylim(ticks[0], ticks[-1])
            axis.set_yticks(ticks)
            axis.yaxis.set_major_formatter(FuncFormatter(_memory_label))

    for axis in axes[1:]:
        axis.set_ylabel("")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(label) for _, label, *_ in reversed(SERIES) if label in labels]
    figure.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(order),
        frameon=False,
        handlelength=2.6,
        columnspacing=1.8,
    )

    gpu = str(usable["gpu"].iloc[0]) if "gpu" in usable else "CUDA GPU"
    figure.suptitle(f"{title} · {gpu}", x=0.5, y=0.985, fontsize=15, fontweight="bold")
    setup_footer = (
        f"{configuration.get('dtype', 'bf16').upper()}  ·  head dim "
        f"{configuration.get('head_dim', 64)}  ·  "
        f"{configuration.get('total_tokens', 16384):,} tokens per batch"
    )
    if timing:
        setup_footer += "  ·  median of do_bench"
    figure.text(0.5, 0.018, setup_footer, ha="center", fontsize=9, color="#5A5F66")
    figure.tight_layout(rect=(0, 0.05, 1, 0.9), w_pad=2.2)

    png = output_dir / f"{output_stem}.png"
    pdf = output_dir / f"{output_stem}.pdf"
    figure.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    figure.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    return png, pdf


def make_release_figures(
    frame: pd.DataFrame,
    configuration: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, ...]:
    usable = frame[frame["status"].eq("ok")].copy()
    if usable.empty:
        raise ValueError("no successful benchmark rows to plot")
    # Memory above the pre-measurement allocation (weights and inputs excluded).
    usable["peak_memory_gib"] = usable["incremental_peak_allocated_bytes"] / GIB

    sns.set_theme(
        context="notebook",
        style="whitegrid",
        font="DejaVu Sans",
        font_scale=0.95,
        rc={
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "legend.fontsize": 10.5,
            "grid.color": "#E4E6EA",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#6B7078",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_suffix = _gpu_output_suffix(str(usable["gpu"].iloc[0]))
    throughput = _make_metric_figure(
        usable,
        configuration,
        metric="attention_tflops",
        title="Disentangled-attention throughput",
        ylabel="TFLOP/s",
        output_stem=f"kernel_throughput_{gpu_suffix}",
        output_dir=output_dir,
        dpi=dpi,
    )
    memory = _make_metric_figure(
        usable,
        configuration,
        metric="peak_memory_gib",
        title="Attention incremental peak memory",
        ylabel="Incremental peak memory (GiB, log₂ scale)",
        output_stem=f"kernel_memory_{gpu_suffix}",
        output_dir=output_dir,
        dpi=dpi,
    )
    return (*throughput, *memory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/kernel"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    frame, configuration = load_results(args.input)
    for output in make_release_figures(frame, configuration, args.output_dir, args.dpi):
        print(output.resolve())


if __name__ == "__main__":
    main()
