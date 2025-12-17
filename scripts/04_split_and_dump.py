#!/usr/bin/env python3
"""Split labeled flows into train/val/test Parquet + JSONL with group-aware stratification."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


DEFAULT_SPLIT_NAMES = ["train", "val", "test"]
DEFAULT_SPLITS = [0.6, 0.2, 0.2]

ESSENTIAL_FEATURES = [
    "tuple5",
    "win_start_ts",
    "win_end_ts",
    "label",
    "pkt_cnt_fwd",
    "pkt_cnt_bwd",
    "byte_cnt_fwd",
    "byte_cnt_bwd",
    "fwd_bwd_ratio_pkt",
    "fwd_bwd_ratio_byte",
    "tcp_syn_ratio",
    "tcp_fin_ratio",
    "tcp_rst_ratio",
]

META_COLUMNS = {
    "tuple5",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "proto",
    "win_start_ts",
    "win_end_ts",
    "label",
    "dataset",
    "capture_id",
    "label_capture_id",
    "label_flow_id",
    "label_iou",
    "label_priority",
    "label_is_unknown",
}

KEY_COLUMNS = ["tuple5", "win_start_ts", "win_end_ts"]


def clean_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def clean_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, help="Input labeled Parquet path")
    parser.add_argument(
        "--enriched-in",
        dest="enriched_input",
        default=None,
        help="Optional enriched labeled Parquet path (from 03 with --rich-out)",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for processed splits")
    parser.add_argument(
        "--splits",
        nargs="+",
        type=float,
        default=DEFAULT_SPLITS,
        help="Split ratios (must sum to 1.0)",
    )
    parser.add_argument(
        "--split_names",
        nargs="+",
        default=DEFAULT_SPLIT_NAMES,
        help="Names for each split (must match number of ratios)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--group_column",
        default=None,
        help="Column used for leakage-safe grouping (default: capture_id or label_capture_id)",
    )
    parser.add_argument(
        "--stats",
        default=None,
        help="Optional JSON file to dump split statistics (class counts per split)",
    )
    return parser.parse_args()


def canonical_tuple(row: pd.Series) -> Tuple:
    src = f"{row.src_ip}:{row.src_port}"
    dst = f"{row.dst_ip}:{row.dst_port}"
    if src <= dst:
        return (row.src_ip, row.src_port, row.dst_ip, row.dst_port, row.proto)
    return (row.dst_ip, row.dst_port, row.src_ip, row.src_port, row.proto)


def detect_group_column(df: pd.DataFrame, override: str | None) -> str:
    if override and override in df.columns:
        return override
    if "label_capture_id" in df.columns and df["label_capture_id"].notna().any():
        return "label_capture_id"
    if "capture_id" in df.columns:
        return "capture_id"
    raise ValueError("No suitable group column found; specify --group_column explicitly.")


def stratified_split_within_group(
    group_df: pd.DataFrame,
    splits: Sequence[float],
    split_names: Sequence[str],
    base_seed: int,
    group_id: str,
) -> Dict[int, str]:
    if not math.isclose(sum(splits), 1.0, rel_tol=1e-6):
        raise ValueError("Splits must sum to 1.0")
    assignments: Dict[int, str] = {}

    for label, label_df in group_df.groupby("label"):
        indices = label_df.index.to_numpy()
        label_seed = int.from_bytes(
            hashlib.sha256(f"{base_seed}:{group_id}:{label}".encode("utf-8")).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(label_seed)
        rng.shuffle(indices)
        n = len(indices)
        raw_counts = [int(math.floor(n * ratio)) for ratio in splits]
        remainder = n - sum(raw_counts)
        for i in range(remainder):
            raw_counts[i % len(raw_counts)] += 1

        offset = 0
        for split_name, count in zip(split_names, raw_counts):
            if count <= 0:
                continue
            selected = indices[offset : offset + count]
            for idx in selected:
                assignments[int(idx)] = split_name
            offset += count

    # In case of any unassigned indices (due to zero counts), default to first split.
    default_split = split_names[0]
    for idx in group_df.index:
        assignments.setdefault(int(idx), default_split)

    return assignments


def prepare_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    pkt_hist_cols = sorted(
        [
            col
            for col in df.columns
            if col.startswith("pkt_size_hist_") and col.split("_")[-1].isdigit()
        ],
        key=lambda c: int(c.split("_")[-1]),
    )
    iat_hist_cols = sorted(
        [
            col
            for col in df.columns
            if col.startswith("iat_hist_") and col.split("_")[-1].isdigit()
        ],
        key=lambda c: int(c.split("_")[-1]),
    )

    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in pkt_hist_cols + iat_hist_cols
    ]
    feature_cols = [col for col in numeric_cols if col not in META_COLUMNS]
    feature_cols.extend(col for col in pkt_hist_cols + iat_hist_cols if col not in feature_cols)
    return feature_cols, pkt_hist_cols, iat_hist_cols


def build_features_dict(
    row: pd.Series,
    feature_cols: Sequence[str],
    pkt_hist_cols: Sequence[str],
    iat_hist_cols: Sequence[str],
) -> Dict[str, object]:
    features: Dict[str, object] = {}
    for col in feature_cols:
        if col in pkt_hist_cols or col in iat_hist_cols:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        if isinstance(value, (np.floating, float)):
            features[col] = float(value)
        elif isinstance(value, (np.integer, int)):
            features[col] = int(value)
        else:
            features[col] = value

    if pkt_hist_cols:
        features["pkt_size_hist"] = [float(row[col]) for col in pkt_hist_cols]
    if iat_hist_cols:
        features["iat_hist"] = [float(row[col]) for col in iat_hist_cols]

    return features


def write_jsonl(
    df: pd.DataFrame,
    json_path: Path,
    feature_cols: Sequence[str],
    pkt_hist_cols: Sequence[str],
    iat_hist_cols: Sequence[str],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        for flow_id, row in enumerate(df.itertuples(index=False), start=1):
            row_dict = row._asdict()
            tuple5 = [
                row_dict.get("src_ip"),
                row_dict.get("dst_ip"),
                int(row_dict.get("src_port", 0)),
                int(row_dict.get("dst_port", 0)),
                int(row_dict.get("proto", 0)),
            ]
            window = [
                float(row_dict.get("win_start_ts", 0.0)),
                float(row_dict.get("win_end_ts", 0.0)),
            ]
            features = build_features_dict(
                pd.Series(row_dict),
                feature_cols=feature_cols,
                pkt_hist_cols=pkt_hist_cols,
                iat_hist_cols=iat_hist_cols,
            )
            record = {
                "flow_id": flow_id,
                "tuple5": tuple5,
                "window": window,
                "features": features,
                "label": row_dict.get("label"),
            }
            fh.write(json.dumps(record) + "\n")


def write_enriched_jsonl(
    df: pd.DataFrame,
    json_path: Path,
    feature_cols: Sequence[str],
    pkt_hist_cols: Sequence[str],
    iat_hist_cols: Sequence[str],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        for flow_id, row in enumerate(df.itertuples(index=False), start=1):
            row_dict = row._asdict()
            tuple5 = [
                row_dict.get("src_ip"),
                row_dict.get("dst_ip"),
                clean_int(row_dict.get("src_port", 0)),
                clean_int(row_dict.get("dst_port", 0)),
                clean_int(row_dict.get("proto", 0)),
            ]
            window = [
                clean_float(row_dict.get("win_start_ts", 0.0)),
                clean_float(row_dict.get("win_end_ts", 0.0)),
            ]
            features = build_features_dict(
                pd.Series(row_dict),
                feature_cols=feature_cols,
                pkt_hist_cols=pkt_hist_cols,
                iat_hist_cols=iat_hist_cols,
            )
            events_raw = row_dict.get("rich_events") or "[]"
            try:
                events = json.loads(events_raw)
            except (json.JSONDecodeError, TypeError):
                events = []
            payload_info = {
                "total_len": clean_int(row_dict.get("payload_total_len", 0)),
                "entropy": clean_float(row_dict.get("payload_entropy", 0.0)),
                "ascii_ratio": clean_float(row_dict.get("payload_ascii_ratio", 0.0)),
                "head_ascii": row_dict.get("rich_payload_head_ascii", "") or "",
                "head_hex": row_dict.get("rich_payload_head_hex", "") or "",
                "tail_ascii": row_dict.get("rich_payload_tail_ascii", "") or "",
                "tail_hex": row_dict.get("rich_payload_tail_hex", "") or "",
                "truncated": bool(row_dict.get("rich_payload_truncated", False)),
            }
            protocols_raw = row_dict.get("rich_protocols") or ""
            protocols = [p for p in protocols_raw.split(",") if p] if protocols_raw else []
            enriched = {
                "context_summary": row_dict.get("rich_context_summary", "") or "",
                "events": events,
                "event_count": clean_int(row_dict.get("rich_event_count", len(events))),
                "payload": payload_info,
                "protocols": protocols,
                "proto_name": row_dict.get("proto_name", ""),
            }
            record = {
                "flow_id": flow_id,
                "tuple5": tuple5,
                "window": window,
                "features": features,
                "enriched": enriched,
                "label": row_dict.get("label"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if "label" not in df.columns:
        raise ValueError("Input Parquet must contain 'label' column.")

    enriched_df = None
    if args.enriched_input:
        enriched_path = Path(args.enriched_input)
        enriched_df = pd.read_parquet(enriched_path)
        for col in KEY_COLUMNS:
            if col not in enriched_df.columns:
                raise ValueError(f"Enriched Parquet missing key column '{col}'")
        if "split" in enriched_df.columns:
            enriched_df = enriched_df.drop(columns=["split"])

    group_col = detect_group_column(df, args.group_column)
    df[group_col] = df[group_col].fillna("")

    total_label_counts = Counter(df["label"])
    split_assignments: Dict[int, str] = {}
    for group_id, group_df in df.groupby(group_col):
        group_assignments = stratified_split_within_group(
            group_df=group_df,
            splits=args.splits,
            split_names=args.split_names,
            base_seed=args.seed,
            group_id=str(group_id),
        )
        split_assignments.update(group_assignments)

    df["split"] = df.index.map(split_assignments)
    if df["split"].isna().any():
        raise ValueError("Split assignment missing for some rows; check grouping logic.")

    assign_df = df[KEY_COLUMNS + ["split"]].copy()
    if assign_df.duplicated(subset=KEY_COLUMNS).any():
        raise ValueError("Duplicate window keys detected; cannot align splits reliably.")

    if enriched_df is not None:
        enriched_df = enriched_df.merge(
            assign_df,
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        if enriched_df["split"].isna().any():
            raise ValueError("Enriched data contains windows without split assignment.")

    feature_cols, pkt_hist_cols, iat_hist_cols = prepare_feature_columns(df)
    essential_missing = [col for col in ESSENTIAL_FEATURES if col not in df.columns]
    if essential_missing:
        print(f"Warning: missing essential feature columns {essential_missing}")

    stats = {"splits": {}}
    label_counts_per_split: Dict[str, Dict[str, int]] = {}

    for split_name in args.split_names:
        split_df = df[df["split"] == split_name].copy()
        if split_df.empty:
            print(f"Warning: split '{split_name}' is empty.")
            continue

        split_df_no_split = split_df.drop(columns=["split"])
        feature_cols_for_parquet = split_df_no_split.columns.tolist()
        parquet_path = out_dir / f"features_{split_name}.parquet"
        split_df_no_split.to_parquet(parquet_path, index=False)

        json_path = out_dir / f"llm_input_struct_{split_name}.jsonl"
        write_jsonl(
            split_df_no_split,
            json_path=json_path,
            feature_cols=feature_cols,
            pkt_hist_cols=pkt_hist_cols,
            iat_hist_cols=iat_hist_cols,
        )

        if enriched_df is not None:
            split_enriched = enriched_df[enriched_df["split"] == split_name].copy()
            split_enriched_no_split = split_enriched.drop(columns=["split"])
            if len(split_enriched_no_split) != len(split_df_no_split):
                raise ValueError(
                    f"Mismatch between structured ({len(split_df_no_split)}) and enriched "
                    f"({len(split_enriched_no_split)}) rows for split '{split_name}'."
                )
            enriched_cols = [
                col for col in split_enriched_no_split.columns if col.startswith("rich_")
            ]
            enriched_subset = split_enriched_no_split[KEY_COLUMNS + enriched_cols]
            combined_df = split_df_no_split.merge(
                enriched_subset,
                on=KEY_COLUMNS,
                how="left",
                validate="one_to_one",
            )
            if enriched_cols and combined_df[enriched_cols].isna().any().any():
                raise ValueError(f"Missing enriched data after merge for split '{split_name}'.")

            enriched_parquet_path = out_dir / f"features_enriched_{split_name}.parquet"
            combined_df.to_parquet(enriched_parquet_path, index=False)

            enriched_json_path = out_dir / f"llm_input_enriched_{split_name}.jsonl"
            write_enriched_jsonl(
                combined_df,
                json_path=enriched_json_path,
                feature_cols=feature_cols,
                pkt_hist_cols=pkt_hist_cols,
                iat_hist_cols=iat_hist_cols,
            )

        label_counts_split = Counter(split_df["label"])
        label_counts_per_split[split_name] = dict(label_counts_split)
        stats["splits"][split_name] = {
            "num_examples": int(len(split_df)),
            "label_counts": label_counts_per_split[split_name],
            "group_ids": sorted(split_df[group_col].unique().tolist()),
        }
        print(
            f"{split_name}: {len(split_df):,} windows "
            f"(groups={len(stats['splits'][split_name]['group_ids'])})"
        )

    if args.stats:
        stats_path = Path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
