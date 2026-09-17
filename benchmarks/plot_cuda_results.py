"""Create report-ready plots from a CUDA benchmark JSON report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

GIB = 1024**3
DTYPE_ORDER = ("fp16", "bf16", "fp32")
DTYPE_LABELS = {"fp16": "FP16", "bf16": "BF16", "fp32": "FP32"}

SERIES = (
    ("base", "padded", "Hugging Face · padded"),
    ("torch", "padded", "DF PyTorch · padded"),
    ("torch", "packed", "DF PyTorch · packed"),
    ("triton", "padded", "DF Triton · padded"),
    ("triton", "packed", "DF Triton · packed"),
    ("flashdeberta", "packed", "FlashDeBERTa · packed"),
)

IMPLEMENTATION_COLORS = {
    "base": "#4D4D4D",
    "torch": "#0072B2",
    "triton": "#D55E00",
    "flashdeberta": "#009E73",
}
LAYOUT_STYLES = {
    "padded": {"linestyle": "-", "marker": "o"},
    "packed": {"linestyle": "--", "marker": "s"},
}


def _load_rows(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)

    workers = report.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError(f"{path} does not contain any benchmark workers")

    rows: list[dict[str, Any]] = []
    for worker in workers:
        metadata = worker.get("metadata", {})
        for result in worker.get("results", []):
            row = dict(result)
            for key in (
                "implementation",
                "layout",
                "dtype",
                "execution",
                "model_architecture",
                "scope",
            ):
                row[key] = metadata.get(key)
            rows.append(row)

    frame = pd.DataFrame.from_records(rows)
    required = {
        "status",
        "implementation",
        "layout",
        "dtype",
        "batch_size",
        "sequence_length",
        "mean_ms",
        "peak_allocated_bytes",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required result fields: {', '.join(missing)}")
    return frame, workers[0].get("metadata", {})


def _available_series(frame: pd.DataFrame) -> tuple[tuple[str, str, str], ...]:
    available = {
        (str(row.implementation), str(row.layout))
        for row in frame[["implementation", "layout"]].drop_duplicates().itertuples()
    }
    return tuple(series for series in SERIES if series[:2] in available)


def _oom_note(panel: pd.DataFrame, series: tuple[tuple[str, str, str], ...]) -> str | None:
    names = {(implementation, layout): label for implementation, layout, label in series}
    failures = panel[panel["status"].eq("oom")]
    grouped: dict[str, list[int]] = {}
    for row in failures.itertuples():
        label = names.get((str(row.implementation), str(row.layout)))
        if label is not None:
            grouped.setdefault(label, []).append(int(row.sequence_length))
    if not grouped:
        return None

    fragments = []
    for _, _, label in series:
        if label in grouped:
            lengths = ", ".join(str(value) for value in sorted(set(grouped[label])))
            fragments.append(f"{label}: {lengths}")
    return "OOM — " + "; ".join(fragments)


def _metadata_caption(metadata: dict[str, Any], frame: pd.DataFrame) -> str:
    system = metadata.get("system", {})
    packages = system.get("packages", {})
    gpu = metadata.get("gpu") or "CUDA GPU"
    capability = metadata.get("compute_capability")
    capability_text = ""
    if isinstance(capability, list) and len(capability) == 2:
        capability_text = f" (SM {capability[0]}.{capability[1]})"
    samples = sorted(
        {int(value) for value in frame.loc[frame["status"].eq("ok"), "samples"].dropna().unique()}
    )
    samples_text = "/".join(str(value) for value in samples) if samples else "unknown"
    return (
        f"{gpu}{capability_text}  ·  eager  ·  {samples_text} measured runs  ·  "
        f"PyTorch {packages.get('torch', '?')}  ·  CUDA {metadata.get('cuda_version', '?')}"
    )


def _format_latency(value: float, _: int) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    if value >= 10:
        return f"{value:g}"
    return f"{value:.1f}"


def _format_memory(value: float, _: int) -> str:
    if value >= 10:
        return f"{value:g}"
    return f"{value:.1f}"


def _legend_handles(series: tuple[tuple[str, str, str], ...]) -> list[Line2D]:
    handles = []
    for implementation, layout, label in series:
        style = LAYOUT_STYLES[layout]
        handles.append(
            Line2D(
                [0],
                [0],
                color=IMPLEMENTATION_COLORS[implementation],
                label=label,
                linewidth=2.5,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=6,
            )
        )
    return handles


def _plot_batch(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    batch_size: int,
    metric: str,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    batch = frame[frame["batch_size"].eq(batch_size)].copy()
    dtypes = [dtype for dtype in DTYPE_ORDER if dtype in set(batch["dtype"])]
    if not dtypes:
        return []
    series = _available_series(batch)
    lengths = sorted(int(value) for value in batch["sequence_length"].unique())

    metric_name = "Mean latency"
    unit = "ms"
    file_prefix = "latency"
    formatter = FuncFormatter(_format_latency)
    if metric == "peak_allocated_bytes":
        metric_name = "Peak allocated GPU memory"
        unit = "GiB"
        file_prefix = "memory"
        batch[metric] = batch[metric] / GIB
        formatter = FuncFormatter(_format_memory)

    sns.set_theme(
        context="talk",
        style="whitegrid",
        font_scale=0.83,
        rc={
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
        },
    )
    figure, axes = plt.subplots(
        1,
        len(dtypes),
        figsize=(16.8, 6.1),
        sharey=False,
        squeeze=False,
    )

    for axis, dtype in zip(axes[0], dtypes, strict=True):
        panel = batch[batch["dtype"].eq(dtype)]
        valid = panel[panel["status"].eq("ok")]
        for implementation, layout, _ in series:
            values = valid[
                valid["implementation"].eq(implementation) & valid["layout"].eq(layout)
            ].sort_values("sequence_length")
            if values.empty:
                continue
            style = LAYOUT_STYLES[layout]
            sns.lineplot(
                data=values,
                x="sequence_length",
                y=metric,
                ax=axis,
                color=IMPLEMENTATION_COLORS[implementation],
                estimator=None,
                errorbar=None,
                linewidth=2.5,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=6,
                markeredgewidth=0.8,
                markeredgecolor="white",
                sort=True,
            )

        axis.set_title(DTYPE_LABELS.get(dtype, dtype.upper()), pad=11)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=10 if metric == "mean_ms" else 2)
        axis.set_xticks(lengths)
        axis.set_xticklabels([f"{length:,}" for length in lengths], rotation=35, ha="right")
        axis.yaxis.set_major_formatter(formatter)
        axis.set_xlabel("Sequence length")
        axis.set_ylabel(f"{metric_name} ({unit}, log scale)")
        axis.grid(which="major", axis="both", alpha=0.85)
        axis.grid(which="minor", axis="y", alpha=0.2, linewidth=0.5)
        axis.tick_params(axis="both", labelsize=9.5)
        axis.margins(x=0.04, y=0.12)

        note = _oom_note(panel, series)
        if note:
            axis.text(
                0.98,
                0.025,
                note,
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                color="#8B1A1A",
                fontsize=7.2,
                wrap=True,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "#FFF4F2",
                    "edgecolor": "#E8B4AE",
                    "alpha": 0.93,
                },
            )

    model = metadata.get("model_architecture", "DeBERTa-v3-base")
    figure.suptitle(
        f"{model} encoder — {metric_name} — batch {batch_size}",
        x=0.5,
        y=0.985,
        fontsize=18,
        fontweight="bold",
    )
    figure.text(0.5, 0.925, _metadata_caption(metadata, batch), ha="center", fontsize=9.5)
    figure.legend(
        handles=_legend_handles(series),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=3.3,
        columnspacing=1.7,
    )
    figure.text(
        0.5,
        0.002,
        "Curves contain successful measurements only; annotated OOM points are capacity "
        "limits, not benchmark failures.",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#555555",
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.24, wspace=0.27)

    outputs = []
    for file_format in formats:
        output = output_dir / f"{file_prefix}_batch_{batch_size}.{file_format}"
        figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(output)
    plt.close(figure)
    return outputs


def _parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    supported = {"png", "pdf", "svg"}
    invalid = sorted(set(formats).difference(supported))
    if not formats or invalid:
        detail = f": {', '.join(invalid)}" if invalid else ""
        raise argparse.ArgumentTypeError(f"formats must be png, pdf, or svg{detail}")
    return formats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="CUDA benchmark JSON report")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/cuda"),
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=("png", "pdf"),
        help="Comma-separated output formats (default: png,pdf)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dpi <= 0 or not math.isfinite(args.dpi):
        raise SystemExit("--dpi must be positive")

    frame, metadata = _load_rows(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for batch_size in sorted(int(value) for value in frame["batch_size"].unique()):
        outputs.extend(
            _plot_batch(
                frame,
                metadata,
                batch_size=batch_size,
                metric="mean_ms",
                output_dir=args.output_dir,
                formats=args.formats,
                dpi=args.dpi,
            )
        )
        outputs.extend(
            _plot_batch(
                frame,
                metadata,
                batch_size=batch_size,
                metric="peak_allocated_bytes",
                output_dir=args.output_dir,
                formats=args.formats,
                dpi=args.dpi,
            )
        )

    print(f"Generated {len(outputs)} files in {args.output_dir.resolve()}")
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
