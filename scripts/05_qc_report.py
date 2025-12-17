#!/usr/bin/env python3
"""Generate dataset summary describing windowed flow features and label makeup."""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

DEFAULT_SPLITS = ["train", "val", "test"]
PROTO_LABELS = {1: "ICMP", 2: "IGMP", 6: "TCP", 17: "UDP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in_dir",
        default="data/processed",
        help="Directory containing features_{split}.parquet (windowed structured features).",
    )
    parser.add_argument(
        "--enriched_dir",
        default="data/processed",
        help="Directory containing features_enriched_{split}.parquet (optional enriched features).",
    )
    parser.add_argument(
        "--out",
        default="data/reports/dataset_summary.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--config",
        default="configs/windows.yaml",
        help="Window/stride configuration YAML (for context metadata).",
    )
    parser.add_argument(
        "--plots_dir",
        default=None,
        help="Optional directory for placing generated plots (default: same folder as report).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Which split names to summarise (default: train val test).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def describe_split(df: pd.DataFrame) -> Dict:
    summary: Dict = {}
    summary["windows"] = int(len(df))
    summary["unique_tuple5"] = int(df["tuple5"].nunique())
    if "label_capture_id" in df.columns:
        summary["unique_label_capture"] = int(df["label_capture_id"].dropna().unique().size)
    if "capture_id" in df.columns:
        summary["unique_capture"] = int(df["capture_id"].dropna().unique().size)
    if "label_is_unknown" in df.columns:
        summary["unknown_ratio"] = float(df["label_is_unknown"].mean())
    return summary


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label"].value_counts().sort_index()
    stats = pd.DataFrame({"count": counts})
    stats["log10_count"] = np.log10(stats["count"].clip(lower=1))
    stats["share"] = stats["count"] / stats["count"].sum()
    return stats.reset_index(names="label")


def plot_log_distribution(distribution: pd.DataFrame, title: str, plots_dir: Path, filename: str) -> Path:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(
        data=distribution,
        x="label",
        y="count",
        hue="label",
        palette="viridis",
        legend=False,
    )
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count (log scale)")
    for patch, count in zip(ax.patches, distribution["count"]):
        ax.annotate(
            f"{count:,}",
            (patch.get_x() + patch.get_width() / 2, count),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
    )
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plot_path = plots_dir / filename
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def format_proto_display(proto: int) -> str:
    return f"{proto} ({PROTO_LABELS.get(int(proto), 'OTHER')})"


def plot_protocol_distribution(
    distribution: pd.DataFrame,
    title: str,
    plots_dir: Path,
    filename: str,
) -> Path:
    if distribution.empty:
        raise ValueError("Protocol distribution is empty; nothing to plot.")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.5, 4))
    ax = sns.barplot(
        data=distribution,
        x="proto_display",
        y="count",
        hue="proto_display",
        palette="crest",
        legend=False,
    )
    ax.set_title(title)
    ax.set_xlabel("Protocol")
    ax.set_ylabel("Window count")
    for patch, (_, row) in zip(ax.patches, distribution.iterrows()):
        ax.annotate(
            f"{int(row['count']):,} ({row['share']:.1%})",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plot_path = plots_dir / filename
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def detect_group_column(df: pd.DataFrame) -> str:
    if "label_capture_id" in df.columns and df["label_capture_id"].notna().any():
        return "label_capture_id"
    if "capture_id" in df.columns:
        return "capture_id"
    return "N/A"


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    enriched_dir = Path(args.enriched_dir) if args.enriched_dir else None
    out_path = Path(args.out)
    plots_dir = Path(args.plots_dir) if args.plots_dir else out_path.parent
    plots_dir.mkdir(parents=True, exist_ok=True)

    window_cfg = load_config(Path(args.config))
    window_sec = window_cfg.get("window_sec")
    stride_sec = window_cfg.get("stride_sec")
    payload_head_bytes = window_cfg.get("payload_head_bytes")
    payload_tail_bytes = window_cfg.get("payload_tail_bytes")

    combined_distributions: Dict[str, pd.DataFrame] = {}
    split_summaries: Dict[str, Dict] = {}
    group_column = None
    total_windows = 0
    total_unique_tuple5 = set()
    proto_window_counts: Dict[int, int] = defaultdict(int)
    proto_tuple_sets: Dict[int, set] = defaultdict(set)
    attack_window_counts: Dict[int, int] = defaultdict(int)
    attack_tuple_sets: Dict[int, set] = defaultdict(set)
    total_attack_windows = 0
    proto_summary_rows: List[List[str]] = []
    proto_summary_text_parts: List[str] = []

    for split in args.splits:
        split_path = in_dir / f"features_{split}.parquet"
        if not split_path.exists():
            continue
        df = load_split(split_path)
        split_summary = describe_split(df)
        split_summaries[split] = split_summary
        total_windows += split_summary["windows"]
        total_unique_tuple5.update(df["tuple5"].unique())

        # protocol breakdown (windows)
        proto_counts = df["proto"].value_counts()
        for proto, count in proto_counts.items():
            proto_window_counts[int(proto)] += int(count)

        # protocol breakdown (distinct tuple5)
        for proto, tuples in df.groupby("proto")["tuple5"].unique().items():
            proto_tuple_sets[int(proto)].update(tuples)

        proto_distribution = (
            proto_counts.rename_axis("proto")
            .reset_index(name="count")
            .assign(proto=lambda d: d["proto"].astype(int))
        )
        if not proto_distribution.empty:
            proto_distribution["share"] = proto_distribution["count"] / proto_distribution["count"].sum()
            proto_distribution["proto_display"] = proto_distribution["proto"].apply(format_proto_display)
            proto_plot_path = plot_protocol_distribution(
                proto_distribution,
                title=f"Protocol Distribution - {split}",
                plots_dir=plots_dir,
                filename=f"protocol_distribution_{split}.pdf",
            )
        else:
            proto_plot_path = None

        # attack-only stats (label != BENIGN, unknown)
        attack_mask = df["label"].ne("BENIGN") & df["label"].ne("unknown")
        total_attack_windows += int(attack_mask.sum())
        attack_proto_counts = df[attack_mask]["proto"].value_counts()
        for proto, count in attack_proto_counts.items():
            attack_window_counts[int(proto)] += int(count)
        for proto, tuples in df[attack_mask].groupby("proto")["tuple5"].unique().items():
            attack_tuple_sets[int(proto)].update(tuples)

        distribution = label_distribution(df)
        combined_distributions[split] = distribution

        if not group_column:
            group_column = detect_group_column(df)

        plot_path = plot_log_distribution(
            distribution,
            title=f"Label Counts (log scale) - {split}",
            plots_dir=plots_dir,
            filename=f"label_distribution_log_{split}.pdf",
        )
        split_summary["label_plot"] = plot_path.name
        if proto_plot_path:
            split_summary["proto_plot"] = proto_plot_path.name

    global_label_counts = defaultdict(int)
    for split, dist in combined_distributions.items():
        for _, row in dist.iterrows():
            global_label_counts[row["label"]] += int(row["count"])
    global_distribution = (
        pd.DataFrame(
            {
                "label": list(global_label_counts.keys()),
                "count": list(global_label_counts.values()),
            }
        )
        .sort_values("label")
        .reset_index(drop=True)
    )
    label_summary_text = ""
    if not global_distribution.empty:
        global_distribution["log10_count"] = np.log10(global_distribution["count"].clip(lower=1))
        global_distribution["share"] = global_distribution["count"] / global_distribution["count"].sum()
        global_plot_path = plot_log_distribution(
            global_distribution,
            title="Label Counts (log scale) - Overall",
            plots_dir=plots_dir,
            filename="label_distribution_log_overall.pdf",
        )
        label_summary_text = ", ".join(
            f"{row['label']}={int(row['count']):,} ({row['share']:.2%})"
            for _, row in global_distribution.iterrows()
        )
    else:
        global_plot_path = None

    for proto in sorted(proto_window_counts.keys()):
        window_count = proto_window_counts.get(proto, 0)
        tuple_count = len(proto_tuple_sets.get(proto, set()))
        attack_window_count = attack_window_counts.get(proto, 0)
        attack_tuple_count = len(attack_tuple_sets.get(proto, set()))
        window_share = window_count / total_windows if total_windows else 0.0
        attack_window_share = (
            attack_window_count / total_attack_windows if total_attack_windows else 0.0
        )
        proto_summary_rows.append(
            [
                format_proto_display(proto),
                f"{window_count:,}",
                f"{window_share:.2%}",
                f"{tuple_count:,}",
                f"{attack_window_count:,}",
                f"{attack_window_share:.2%}",
                f"{attack_tuple_count:,}",
            ]
        )
        text = (
            f"proto {proto} ({PROTO_LABELS.get(proto, 'n/a')}): {window_count:,} windows ({window_share:.2%}), "
            f"{tuple_count:,} flows"
        )
        if attack_window_count > 0:
            text += f", attacks {attack_window_count:,} windows ({attack_window_share:.2%})"
        proto_summary_text_parts.append(text)

    if proto_window_counts:
        overall_proto_distribution = (
            pd.DataFrame(
                {
                    "proto": list(proto_window_counts.keys()),
                    "count": list(proto_window_counts.values()),
                }
            )
            .sort_values("proto")
            .reset_index(drop=True)
        )
        overall_proto_distribution["share"] = (
            overall_proto_distribution["count"] / overall_proto_distribution["count"].sum()
        )
        overall_proto_distribution["proto_display"] = overall_proto_distribution["proto"].apply(format_proto_display)
        overall_proto_plot_path = plot_protocol_distribution(
            overall_proto_distribution,
            title="Protocol Distribution - Overall",
            plots_dir=plots_dir,
            filename="protocol_distribution_overall.pdf",
        )
    else:
        overall_proto_plot_path = None

    if total_attack_windows:
        attack_proto_distribution = (
            pd.DataFrame(
                {
                    "proto": list(attack_window_counts.keys()),
                    "count": list(attack_window_counts.values()),
                }
            )
            .sort_values("proto")
            .reset_index(drop=True)
        )
        attack_proto_distribution["share"] = (
            attack_proto_distribution["count"] / attack_proto_distribution["count"].sum()
        )
        attack_proto_distribution["proto_display"] = attack_proto_distribution["proto"].apply(format_proto_display)
        attack_proto_plot_path = plot_protocol_distribution(
            attack_proto_distribution,
            title="Protocol Distribution - Attack Windows",
            plots_dir=plots_dir,
            filename="protocol_distribution_attack.pdf",
        )
    else:
        attack_proto_plot_path = None

    enriched_available = enriched_dir is not None and any(
        (enriched_dir / f"features_enriched_{split}.parquet").exists() for split in args.splits
    )

    report_lines: List[str] = []
    report_lines.append("# Dataset Summary")
    report_lines.append("")
    if window_sec is not None and stride_sec is not None:
        report_lines.append(f"- Windowing: {window_sec}s window, {stride_sec}s stride")
    if payload_head_bytes is not None and payload_tail_bytes is not None:
        report_lines.append(
            f"- Payload previews: head/tail {payload_head_bytes}B / {payload_tail_bytes}B "
            + ("(enriched outputs produced)" if enriched_available else "(enriched outputs not produced)")
        )
    report_lines.append(f"- Total windowed samples: {total_windows:,}")
    report_lines.append(f"- Distinct tuple5 flows observed: {len(total_unique_tuple5):,}")
    if group_column:
        report_lines.append(
            f"- Split grouping column: `{group_column}` with stratified 6:2:2 sampling per label, seed 42"
        )
    if total_attack_windows:
        attack_protocols = sorted(proto for proto, count in attack_window_counts.items() if count > 0)
        proto_str = ", ".join(str(p) for p in attack_protocols) if attack_protocols else "n/a"
        report_lines.append(
            f"- Attack windows: {total_attack_windows:,} (protocols: {proto_str})"
        )
    if proto_summary_text_parts:
        report_lines.append("- Protocol mix: " + "; ".join(proto_summary_text_parts))
    if label_summary_text:
        report_lines.append("- Label mix: " + label_summary_text)
    report_lines.append("")

    report_lines.append("## Split Overview")
    report_lines.append("")
    columns = ["Split", "Windows", "Unique tuple5", "Unknown ratio", "Label plot", "Protocol plot"]
    table_rows = []
    for split, summary in split_summaries.items():
        row = [
            split,
            f"{summary.get('windows', 0):,}",
            f"{summary.get('unique_tuple5', 0):,}",
            f"{summary.get('unknown_ratio', 0.0):.2%}" if "unknown_ratio" in summary else "N/A",
            summary.get("label_plot", "N/A"),
            summary.get("proto_plot", "N/A"),
        ]
        table_rows.append(row)
    if table_rows:
        report_lines.append("| " + " | ".join(columns) + " |")
        report_lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in table_rows:
            report_lines.append("| " + " | ".join(row) + " |")
        report_lines.append("")

    if proto_summary_rows:
        report_lines.append("## Protocol Breakdown")
        report_lines.append("")
        proto_columns = [
            "Proto",
            "Windows",
            "Window share",
            "Distinct tuple5",
            "Attack windows",
            "Attack share",
            "Attack tuple5",
        ]
        report_lines.append("| " + " | ".join(proto_columns) + " |")
        report_lines.append("| " + " | ".join(["---"] * len(proto_columns)) + " |")
        for row in proto_summary_rows:
            report_lines.append("| " + " | ".join(map(str, row)) + " |")
        report_lines.append("")

    if any("proto_plot" in summary for summary in split_summaries.values()):
        report_lines.append("## Protocol Distribution by Split")
        report_lines.append("")
        for split, summary in split_summaries.items():
            plot_name = summary.get("proto_plot")
            if not plot_name or plot_name == "N/A":
                continue
            report_lines.append(f"### {split}")
            report_lines.append("")
            report_lines.append(f"![Protocol Distribution ({split})](./{plot_name})")
            report_lines.append("")

    if overall_proto_plot_path:
        report_lines.append("## Overall Protocol Distribution")
        report_lines.append("")
        report_lines.append(f"![Protocol Distribution (Overall)](./{overall_proto_plot_path.name})")
        report_lines.append("")

    if attack_proto_plot_path:
        report_lines.append("## Attack Protocol Distribution")
        report_lines.append("")
        report_lines.append(f"![Protocol Distribution (Attack Windows)](./{attack_proto_plot_path.name})")
        report_lines.append("")

    report_lines.append("## Label Distribution by Split")
    report_lines.append("")
    for split, dist in combined_distributions.items():
        report_lines.append(f"### {split}")
        report_lines.append("")
        report_lines.append("| Label | Count | log10(count) | Share |")
        report_lines.append("| --- | ---: | ---: | ---: |")
        for _, row in dist.iterrows():
            report_lines.append(
                f"| `{row['label']}` | {int(row['count']):,} | {row['log10_count']:.3f} | {row['share']:.2%} |"
            )
        report_lines.append(f"![Label Distribution ({split})](./{split_summaries[split]['label_plot']})")
        report_lines.append("")

    if not global_distribution.empty:
        report_lines.append("## Global Label Distribution")
        report_lines.append("")
        report_lines.append("| Label | Count | log10(count) | Share |")
        report_lines.append("| --- | ---: | ---: | ---: |")
        for _, row in global_distribution.iterrows():
            report_lines.append(
                f"| `{row['label']}` | {int(row['count']):,} | {row['log10_count']:.3f} | {row['share']:.2%} |"
            )
        if global_plot_path:
            report_lines.append(f"![Label Distribution (Overall)](./{global_plot_path.name})")
        report_lines.append("")

    report_lines.append("## Notes")
    report_lines.append("")
    report_lines.append("- Counts are based on windowed samples; a single long-lived flow contributes multiple windows.")
    report_lines.append("- `unknown` label ratio measures unaligned windows; target is < 5%.")
    report_lines.append("- Label plots use log scale to highlight class imbalance.")
    report_lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote dataset summary to {out_path}")


if __name__ == "__main__":
    main()
