"""Plot the paper-oriented kernel benchmark produced by ``benchmark_cuda``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

GIB = 1024**3
SERIES = (
    ("base", "Hugging Face eager", "#4D4D4D"),
    ("torch", "DF PyTorch", "#0072B2"),
    ("triton", "DF Triton", "#D55E00"),
    ("flashdeberta", "FlashDeBERTa", "#009E73"),
)


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
        "peak_allocated_bytes",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    return frame, report.get("configuration", {})


def _plot_one(
    frame: pd.DataFrame,
    configuration: dict[str, Any],
    *,
    pass_mode: str,
    dropout: float,
    metric: str,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    panel = frame[
        frame["pass"].eq(pass_mode) & frame["dropout"].eq(dropout) & frame["status"].eq("ok")
    ].copy()
    if panel.empty:
        return None
    if metric == "peak_allocated_bytes":
        panel[metric] = panel[metric] / GIB
        ylabel = "Peak allocated memory (GiB)"
        prefix = "memory"
    else:
        ylabel = "Median latency (ms)"
        prefix = "latency"

    sns.set_theme(context="talk", style="whitegrid", font_scale=0.9)
    figure, axis = plt.subplots(figsize=(9.2, 6.2))
    for implementation, label, color in SERIES:
        rows = panel[panel["implementation"].eq(implementation)].sort_values("sequence_length")
        if rows.empty:
            continue
        axis.plot(
            rows["sequence_length"],
            rows[metric],
            marker="o",
            linewidth=2.4,
            markersize=6,
            label=label,
            color=color,
        )

    lengths = sorted(int(value) for value in panel["sequence_length"].unique())
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(lengths)
    axis.set_xticklabels([f"{value:,}" for value in lengths], rotation=30, ha="right")
    axis.set_xlabel("Sequence length")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{pass_mode.replace('_', ' ').title()} · dropout {dropout:g}")
    axis.legend(frameon=False)
    axis.grid(which="major", alpha=0.75)
    figure.text(
        0.5,
        0.01,
        (
            f"{configuration.get('dtype', 'bf16').upper()} · head dim "
            f"{configuration.get('head_dim', 64)} · approximately "
            f"{configuration.get('total_tokens', 16384):,} tokens per point"
        ),
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    suffix = str(dropout).replace(".", "p")
    output = output_dir / f"{prefix}_{pass_mode}_dropout_{suffix}.png"
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/kernel"))
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    frame, configuration = load_results(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    combinations = frame[["pass", "dropout"]].drop_duplicates().itertuples(index=False)
    for pass_mode, dropout in combinations:
        for metric in ("p50_ms", "peak_allocated_bytes"):
            output = _plot_one(
                frame,
                configuration,
                pass_mode=str(pass_mode),
                dropout=float(dropout),
                metric=metric,
                output_dir=args.output_dir,
                dpi=args.dpi,
            )
            if output is not None:
                outputs.append(output)
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
