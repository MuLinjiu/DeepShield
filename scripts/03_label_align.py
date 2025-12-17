#!/usr/bin/env python3
"""Align flow windows with ground-truth labels using IoU over time windows."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


EP_DELIM = ","
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows", required=True, help="Input Parquet with flow windows")
    parser.add_argument("--labels", required=True, help="Ground-truth labels CSV")
    parser.add_argument("--dataset", required=True, help="Dataset key in configs/dataset_map.yaml")
    parser.add_argument("--config", default="configs/dataset_map.yaml", help="Dataset map YAML path")
    parser.add_argument("--out", required=True, help="Output labeled Parquet path")
    parser.add_argument(
        "--rich",
        dest="rich_input",
        help="Optional enriched Parquet path produced by 02 (same rows as --flows)",
    )
    parser.add_argument(
        "--rich-out",
        dest="rich_out",
        help="Optional output Parquet path for enriched flows with labels",
    )
    parser.add_argument("--stats", help="Optional JSON file to write alignment summary")
    return parser.parse_args()


def load_dataset_config(path: Path, dataset: str) -> Dict:
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if dataset not in cfg:
        raise KeyError(f"Dataset '{dataset}' not found in {path}")
    return cfg[dataset]


def clean_label(value: str) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .strip()
        .replace("\u0096", "-")
        .replace("\u2013", "-")  # en dash
        .replace("\u2014", "-")  # em dash
        .replace("\u2212", "-")  # minus sign
    )


@dataclass
class LabelEntry:
    start: float
    end: float
    duration: float
    label: str
    flow_id: Optional[str]
    capture_id: Optional[str]
    priority: int  # 1 for attack, 0 for benign


def canonical_tuple(
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
    proto: int,
) -> Tuple[str, int, str, int, int]:
    ep1 = f"{src_ip}:{src_port}"
    ep2 = f"{dst_ip}:{dst_port}"
    if ep1 <= ep2:
        return src_ip, src_port, dst_ip, dst_port, proto
    return dst_ip, dst_port, src_ip, src_port, proto


def load_labels(
    labels_path: Path,
    dataset_cfg: Dict,
    flow_min_start: Optional[Dict[Tuple, float]] = None,
) -> Dict[Tuple, List[LabelEntry]]:
    dtype_map = {
        dataset_cfg["ip_columns"]["src"]: "string",
        dataset_cfg["ip_columns"]["dst"]: "string",
        dataset_cfg["port_columns"]["src"]: "float64",
        dataset_cfg["port_columns"]["dst"]: "float64",
        dataset_cfg["protocol_column"]: "float64",
    }
    df = pd.read_csv(labels_path, dtype=dtype_map, low_memory=False)

    label_col = dataset_cfg["label_column"]
    start_col = dataset_cfg["start_time_column"]
    end_col = dataset_cfg["end_time_column"]
    capture_col = dataset_cfg.get("capture_id_column")
    flow_id_col = dataset_cfg.get("flow_id_column", "flow_id") if "flow_id_column" in dataset_cfg else None
    benign_label = clean_label(dataset_cfg.get("benign_label", "BENIGN"))
    attack_labels = {clean_label(x) for x in dataset_cfg.get("attack_labels", [])}
    timestamp_tz = dataset_cfg.get("timezone")
    dayfirst = bool(dataset_cfg.get("timestamp_dayfirst", False))

    df[label_col] = df[label_col].map(clean_label)

    start = pd.to_datetime(df[start_col], dayfirst=dayfirst, errors="coerce")
    end = pd.to_datetime(df[end_col], dayfirst=dayfirst, errors="coerce")
    if timestamp_tz:
        start = start.dt.tz_localize(timestamp_tz, nonexistent="NaT", ambiguous="NaT")
        end = end.dt.tz_localize(timestamp_tz, nonexistent="NaT", ambiguous="NaT")

    ts_start = start.view("int64") / 1_000_000_000.0
    ts_end = end.view("int64") / 1_000_000_000.0
    durations = np.clip(ts_end - ts_start, a_min=0.0, a_max=None)

    df["_canon_key"] = [
        canonical_tuple(
            src_ip=str(src_ip),
            src_port=int(src_port) if not pd.isna(src_port) else 0,
            dst_ip=str(dst_ip),
            dst_port=int(dst_port) if not pd.isna(dst_port) else 0,
            proto=int(proto) if not pd.isna(proto) else 0,
        )
        for src_ip, src_port, dst_ip, dst_port, proto in zip(
            df[dataset_cfg["ip_columns"]["src"]],
            df[dataset_cfg["port_columns"]["src"]],
            df[dataset_cfg["ip_columns"]["dst"]],
            df[dataset_cfg["port_columns"]["dst"]],
            df[dataset_cfg["protocol_column"]],
        )
    ]

    # Prepare per-key time offsets using flow minima if available
    default_offset = float(dataset_cfg.get("default_time_offset_sec", 0.0))
    extra_offsets = [float(x) for x in dataset_cfg.get("time_offset_candidates_sec", [])]
    offset_map: Dict[Tuple, float] = {}
    if flow_min_start:
        ts_start_series = pd.Series(ts_start, index=df.index)
        label_mins = ts_start_series.groupby(df["_canon_key"]).min()
        for key, label_min in label_mins.items():
            flow_min = flow_min_start.get(tuple(key))
            if flow_min is not None and not math.isnan(label_min):
                offset_map[tuple(key)] = flow_min - float(label_min)

    label_lookup: Dict[Tuple, List[LabelEntry]] = {}
    for idx, row in df.iterrows():
        key = row["_canon_key"]
        label = row[label_col]
        start_ts = ts_start.iloc[idx]
        end_ts = ts_end.iloc[idx]
        duration = durations[idx]
        if math.isnan(start_ts) or math.isnan(end_ts):
            continue
        capture_id = row[capture_col] if capture_col else None
        flow_id = row[flow_id_col] if flow_id_col and flow_id_col in row else None
        priority = 0 if label == benign_label else 1 if label in attack_labels else 1
        base_offset = offset_map.get(tuple(key), default_offset)
        offsets = [base_offset] + extra_offsets
        seen_offsets = set()
        for offset_value in offsets:
            offset = float(offset_value)
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)
            entry = LabelEntry(
                start=float(start_ts + offset),
                end=float(end_ts + offset),
                duration=float(duration),
                label=str(label),
                flow_id=str(flow_id) if flow_id is not None and not pd.isna(flow_id) else None,
                capture_id=str(capture_id) if capture_id is not None and not pd.isna(capture_id) else None,
                priority=priority,
            )
            label_lookup.setdefault(key, []).append(entry)

    # sort label entries per key by start time (stable)
    for entries in label_lookup.values():
        entries.sort(key=lambda e: e.start)

    return label_lookup


def align_labels(
    flows_df: pd.DataFrame,
    label_lookup: Dict[Tuple, List[LabelEntry]],
    benign_label: str,
) -> Tuple[pd.DataFrame, Dict]:
    n = len(flows_df)
    labels = np.full(n, "unknown", dtype=object)
    match_iou = np.zeros(n, dtype=float)
    match_capture = np.full(n, "", dtype=object)
    match_flow_id = np.full(n, "", dtype=object)
    match_priority = np.zeros(n, dtype=np.int16)

    unknown = 0
    multiple_candidates = 0

    tuple_cols = ["src_ip", "src_port", "dst_ip", "dst_port", "proto"]

    for key, group in flows_df.groupby(tuple_cols):
        entries = label_lookup.get(tuple(key))
        if not entries:
            unknown += len(group)
            continue

        entry_starts = np.array([e.start for e in entries], dtype=float)
        entry_ends = np.array([e.end for e in entries], dtype=float)
        entry_durations = np.array([e.duration for e in entries], dtype=float)
        entry_labels = np.array([e.label for e in entries], dtype=object)
        entry_priority = np.array([e.priority for e in entries], dtype=np.int8)
        entry_capture = np.array([e.capture_id or "" for e in entries], dtype=object)
        entry_flow_id = np.array([e.flow_id or "" for e in entries], dtype=object)

        group_starts = group["win_start_ts"].to_numpy(dtype=float)
        group_ends = group["win_end_ts"].to_numpy(dtype=float)
        indices = group.index.to_numpy()

        for idx, win_start, win_end in zip(indices, group_starts, group_ends):
            overlaps_start = np.maximum(win_start, entry_starts)
            overlaps_end = np.minimum(win_end, entry_ends)
            intersections = np.maximum(0.0, overlaps_end - overlaps_start)

            valid = intersections > 0.0
            if not np.any(valid):
                if np.all(entry_priority == 0):
                    labels[idx] = benign_label
                    match_iou[idx] = 0.0
                    match_capture[idx] = entry_capture[0] if entry_capture.size else ""
                    match_flow_id[idx] = entry_flow_id[0] if entry_flow_id.size else ""
                    match_priority[idx] = 0
                else:
                    unknown += 1
                continue

            candidate_durations = entry_durations[valid]
            candidate_intersections = intersections[valid]
            candidate_labels = entry_labels[valid]
            candidate_priorities = entry_priority[valid]
            candidate_captures = entry_capture[valid]
            candidate_flow_ids = entry_flow_id[valid]

            candidate_unions = (win_end - win_start) + candidate_durations - candidate_intersections
            with np.errstate(divide="ignore", invalid="ignore"):
                candidate_ious = np.divide(
                    candidate_intersections,
                    candidate_unions,
                    out=np.zeros_like(candidate_intersections),
                    where=candidate_unions > 0,
                )

            best_idx = -1
            best_iou = 0.0
            best_priority = -1

            for pos, (iou, priority) in enumerate(zip(candidate_ious, candidate_priorities)):
                if iou > best_iou + EPS:
                    best_iou = float(iou)
                    best_idx = pos
                    best_priority = int(priority)
                elif abs(iou - best_iou) <= EPS and priority > best_priority:
                    best_iou = float(iou)
                    best_idx = pos
                    best_priority = int(priority)

            if best_idx == -1:
                unknown += 1
                continue

            if np.sum(valid) > 1:
                multiple_candidates += 1

            labels[idx] = candidate_labels[best_idx]
            match_iou[idx] = candidate_ious[best_idx]
            match_capture[idx] = candidate_captures[best_idx]
            match_flow_id[idx] = candidate_flow_ids[best_idx]
            match_priority[idx] = best_priority

    flows_df = flows_df.copy()
    flows_df["label"] = labels
    flows_df["label_iou"] = match_iou
    flows_df["label_capture_id"] = match_capture
    flows_df["label_flow_id"] = match_flow_id
    flows_df["label_priority"] = match_priority
    flows_df["label_is_unknown"] = flows_df["label"] == "unknown"

    stats = {
        "total_windows": int(len(flows_df)),
        "unknown_windows": int((flows_df["label_is_unknown"]).sum()),
        "unknown_ratio": float((flows_df["label_is_unknown"]).mean()),
        "label_counts": flows_df["label"].value_counts().to_dict(),
        "multi_candidate_assignments": int(multiple_candidates),
        "benign_label": benign_label,
    }
    return flows_df, stats


def main() -> None:
    args = parse_args()
    if bool(args.rich_input) != bool(args.rich_out):
        raise ValueError("Both --rich and --rich-out must be specified together.")

    dataset_cfg = load_dataset_config(Path(args.config), args.dataset)
    benign_label = clean_label(dataset_cfg.get("benign_label", "BENIGN"))

    flows_df = pd.read_parquet(args.flows)

    tuple_cols = ["src_ip", "src_port", "dst_ip", "dst_port", "proto"]
    flow_min_start_series = flows_df.groupby(tuple_cols)["win_start_ts"].min()
    flow_min_start = {tuple(idx): float(value) for idx, value in flow_min_start_series.items()}

    label_lookup = load_labels(Path(args.labels), dataset_cfg, flow_min_start=flow_min_start)

    labeled_df, stats = align_labels(flows_df, label_lookup, benign_label=benign_label)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(labeled_df):,} labeled windows to {out_path}")
    print(
        f"Unknown windows: {stats['unknown_windows']} "
        f"({stats['unknown_ratio']*100:.2f}% of total)"
    )

    if args.stats:
        stats_path = Path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2))

    if args.rich_input:
        rich_df = pd.read_parquet(args.rich_input)
        key_cols = ["tuple5", "win_start_ts", "win_end_ts"]
        missing_keys = [col for col in key_cols if col not in rich_df.columns]
        if missing_keys:
            raise ValueError(f"Enriched input missing key columns: {missing_keys}")
        label_cols = [
            "label",
            "label_iou",
            "label_capture_id",
            "label_flow_id",
            "label_priority",
            "label_is_unknown",
        ]
        rich_df = rich_df.copy()
        for col in label_cols:
            if col in rich_df.columns:
                rich_df.drop(columns=[col], inplace=True)
        merge_keys = key_cols
        enriched_labeled = rich_df.merge(
            labeled_df[merge_keys + label_cols],
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )
        if enriched_labeled[label_cols].isna().any().any():
            raise ValueError("Failed to propagate labels onto enriched data; unmatched rows detected.")
        rich_out_path = Path(args.rich_out)
        rich_out_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_labeled.to_parquet(rich_out_path, index=False)
        print(f"Wrote {len(enriched_labeled):,} labeled enriched windows to {rich_out_path}")


if __name__ == "__main__":
    main()
