#!/usr/bin/env python
"""
Evaluate how strongly unordered IP pairs correlate with labels.

Default: training split, TCP (proto==6), exclude BENIGN, ignore ports
(treat A->B and B->A as the same pair).

Usage from repo root:
    python scripts/eval_ip_pair_purity.py \
        --data data/processed/features_enriched_train.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import polars as pl


def load_ip_pairs(
    path: Path,
    proto: int = 6,
    label_exclude: Iterable[str] = ("BENIGN",),
) -> pl.LazyFrame:
    """Return LazyFrame with ip_low, ip_high, label for the requested subset."""
    label_exclude = tuple(label_exclude)
    lf = pl.scan_parquet(path)

    filters = [pl.col("proto") == proto]
    if label_exclude:
        filters.append(~pl.col("label").is_in(label_exclude))

    lf = lf.filter(filters)
    # collapse direction: lexicographic min/max to represent unordered pair
    return lf.with_columns(
        pl.when(pl.col("src_ip") <= pl.col("dst_ip"))
        .then(pl.col("src_ip"))
        .otherwise(pl.col("dst_ip"))
        .alias("ip_low"),
        pl.when(pl.col("src_ip") <= pl.col("dst_ip"))
        .then(pl.col("dst_ip"))
        .otherwise(pl.col("src_ip"))
        .alias("ip_high"),
    ).select("ip_low", "ip_high", "label")


def pair_purity(lf: pl.LazyFrame) -> Tuple[int, pl.DataFrame, pl.DataFrame, float, int]:
    """Compute per-pair purity and return summary plus detailed counts."""
    n_rows = lf.select(pl.len()).collect(streaming=True)[0, 0]

    pair_label = (
        lf.group_by(["ip_low", "ip_high", "label"])
        .agg(pl.len().alias("cnt"))
        .collect(streaming=True)
    )
    pair_total = pair_label.group_by(["ip_low", "ip_high"]).agg(pl.sum("cnt").alias("pair_total"))
    pair_label = pair_label.join(pair_total, on=["ip_low", "ip_high"])
    pair_label = pair_label.with_columns((pl.col("cnt") / pl.col("pair_total")).alias("fraction"))

    # Pick dominant label per pair
    pair_top = (
        pair_label.sort(["ip_low", "ip_high", "fraction"], descending=[False, False, True])
        .group_by(["ip_low", "ip_high"])
        .agg(
            pl.first("label").alias("top_label"),
            pl.first("fraction").alias("top_fraction"),
            pl.first("pair_total").alias("pair_total"),
        )
    )

    weighted_purity = (pair_top["top_fraction"] * pair_top["pair_total"]).sum() / pair_top[
        "pair_total"
    ].sum()
    high_purity = int((pair_top["top_fraction"] >= 0.9).sum())

    return n_rows, pair_top, pair_label, weighted_purity, high_purity


def _format_pair(row: Tuple[str, str]) -> str:
    """Human-friendly pair label for plots."""
    return f"{row[0]}↔{row[1]}"


def make_plots(
    pair_top: pl.DataFrame,
    pair_label: pl.DataFrame,
    plots_dir: Path,
    top_k: int,
) -> None:
    """Generate diagnostic plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Volume histogram (log x)
    volumes = pair_top["pair_total"].to_list()
    plt.figure(figsize=(6, 4))
    plt.hist(volumes, bins=min(20, len(volumes)), color="#4c78a8", edgecolor="white")
    plt.xscale("log")
    plt.xlabel("Samples per IP pair (log scale)")
    plt.ylabel("Count of IP pairs")
    plt.title("IP pair volume distribution")
    plt.tight_layout()
    plt.savefig(plots_dir / "pair_volume_hist.png", dpi=200)
    plt.close()

    # 2) Cumulative coverage curve
    sorted_vol = sorted(volumes, reverse=True)
    cum = []
    total = sum(sorted_vol) or 1
    running = 0
    for v in sorted_vol:
        running += v
        cum.append(running / total)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(sorted_vol) + 1), cum, marker="o")
    plt.xlabel("Top N IP pairs (by volume)")
    plt.ylabel("Cumulative sample coverage")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.title("Coverage by top IP pairs")
    plt.tight_layout()
    plt.savefig(plots_dir / "coverage_curve.png", dpi=200)
    plt.close()

    # 3) Purity histogram
    plt.figure(figsize=(6, 4))
    plt.hist(pair_top["top_fraction"].to_list(), bins=10, range=(0, 1), color="#72b7b2")
    plt.xlabel("Purity (dominant label fraction)")
    plt.ylabel("Count of IP pairs")
    plt.title("Purity distribution")
    plt.tight_layout()
    plt.savefig(plots_dir / "purity_hist.png", dpi=200)
    plt.close()

    # 4) Purity vs volume scatter
    plt.figure(figsize=(6, 4))
    plt.scatter(pair_top["pair_total"], pair_top["top_fraction"], alpha=0.8, color="#e45756")
    plt.xscale("log")
    plt.xlabel("Samples per IP pair (log scale)")
    plt.ylabel("Purity")
    plt.ylim(0, 1.05)
    plt.title("Purity vs volume")
    plt.tight_layout()
    plt.savefig(plots_dir / "purity_vs_volume.png", dpi=200)
    plt.close()

    # 5) Stacked bar for top K pairs by volume
    top_pairs = pair_top.sort("pair_total", descending=True).head(top_k)
    pair_order = [f"{a}↔{b}" for a, b in zip(top_pairs["ip_low"], top_pairs["ip_high"])]
    subset = pair_label.filter(
        pl.concat_list([pl.col("ip_low"), pl.col("ip_high")]).list.join("↔").is_in(pair_order)
    )
    pivot = (
        subset.with_columns(pl.concat_list([pl.col("ip_low"), pl.col("ip_high")]).list.join("↔").alias("pair"))
        .group_by(["pair", "label"])
        .agg(pl.sum("cnt").alias("cnt"))
        .pivot(index="pair", columns="label", values="cnt")
        .fill_null(0)
    )
    if pivot.height > 0:
        pivot_pd = pivot.to_pandas().set_index("pair").reindex(pair_order)
        labels = [c for c in pivot_pd.columns]
        plt.figure(figsize=(max(6, 0.7 * len(pivot_pd)), 4))
        bottom = None
        for lbl in labels:
            plt.bar(pivot_pd.index, pivot_pd[lbl], bottom=bottom, label=lbl)
            bottom = pivot_pd[lbl] if bottom is None else bottom + pivot_pd[lbl]
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.ylabel("Samples")
        plt.title(f"Top {min(top_k, len(pair_order))} IP pairs by volume (label mix)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plots_dir / "top_pairs_stacked.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess label correlation for unordered IP pairs.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to features_enriched_{split}.parquet (usually the train split).",
    )
    parser.add_argument(
        "--proto", type=int, default=6, help="Protocol number to include (default: 6 for TCP)."
    )
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=["BENIGN"],
        help="Label(s) to exclude; can be repeated. Default: BENIGN.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top pairs by volume to show (default: 10).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="If set, save diagnostic plots (png) to this directory.",
    )
    args = parser.parse_args()

    lf = load_ip_pairs(args.data, proto=args.proto, label_exclude=args.exclude_label)
    n_rows, pair_top, pair_label, weighted_purity, high_purity = pair_purity(lf)

    num_pairs = pair_top.height
    print(f"Data: {args.data}")
    print(f"Filters: proto={args.proto}, exclude_label={args.exclude_label}")
    print(f"Non-BENIGN rows: {n_rows}")
    print(f"Unique unordered IP pairs: {num_pairs}")
    print(f"Weighted average purity (pair -> label): {weighted_purity:.3f}")
    print(f"Pairs with purity >= 0.9: {high_purity}/{num_pairs}")

    print("\nTop pairs by volume:")
    top_table = pair_top.sort("pair_total", descending=True).select(
        "ip_low", "ip_high", "pair_total", "top_label", "top_fraction"
    ).head(args.top)
    print(top_table)

    if args.plots_dir:
        make_plots(pair_top, pair_label, args.plots_dir, args.top)
        print(f"\nPlots saved to: {args.plots_dir}")


if __name__ == "__main__":
    main()
