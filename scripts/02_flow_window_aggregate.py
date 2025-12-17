#!/usr/bin/env python3
"""Aggregate packet-level tables into flow-window features (structured + enriched)."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


PROTO_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP"}
ASCII_RANGES = [(32, 126), (9, 13)]  # printable + whitespace (tab/newline/carriage return)


@dataclass
class RichOptions:
    enabled: bool
    payload_head_bytes: int
    payload_tail_bytes: int
    max_events: int
    context_windows_sec: Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True, help="Input packet Parquet produced by script 01")
    parser.add_argument("--payload", help="Optional payload parquet produced by script 01b")
    parser.add_argument("--config", required=True, help="Window/bin configuration YAML")
    parser.add_argument("--out", required=True, help="Output Parquet path for structured features")
    parser.add_argument("--rich-out", help="Output Parquet path for enriched features (requires compute_rich)")
    parser.add_argument("--dataset", required=True, help="Dataset identifier (e.g., CICIDS2017)")
    parser.add_argument(
        "--capture-id",
        default=None,
        help="Logical capture identifier; defaults to dataset or infers from input filename",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of packet rows to load (for debugging)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh) or {}
    required = {"window_sec", "stride_sec", "pkt_size_bins", "iat_bins_ms"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Missing keys in {path}: {sorted(missing)}")

    cfg.setdefault("compute_standard", True)
    cfg.setdefault("compute_rich", False)
    cfg.setdefault("payload_head_bytes", 1024)
    cfg.setdefault("payload_tail_bytes", 1024)
    cfg.setdefault("max_events_per_window", 10)
    context_windows = cfg.get("context_windows_sec", [10.0, 60.0])
    if not isinstance(context_windows, (list, tuple)) or len(context_windows) < 2:
        context_windows = [10.0, 60.0]
    cfg["context_windows_sec"] = (float(context_windows[0]), float(context_windows[1]))
    return cfg


def safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return float(num) if num else 0.0
    return float(num) / float(den)


def canonicalize_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical endpoint columns and direction indicator."""
    ep1 = df["src_ip"] + ":" + df["src_port"].astype(str)
    ep2 = df["dst_ip"] + ":" + df["dst_port"].astype(str)
    forward_mask = ep1 <= ep2

    df = df.copy()
    df["canon_src_ip"] = np.where(forward_mask, df["src_ip"], df["dst_ip"])
    df["canon_src_port"] = np.where(forward_mask, df["src_port"], df["dst_port"])
    df["canon_dst_ip"] = np.where(forward_mask, df["dst_ip"], df["src_ip"])
    df["canon_dst_port"] = np.where(forward_mask, df["dst_port"], df["src_port"])
    df["direction"] = np.where(forward_mask, 1, -1)
    return df


def build_histogram(values: np.ndarray, bins: Sequence[float]) -> Tuple[np.ndarray, int]:
    if values.size == 0:
        return np.zeros(len(bins) - 1, dtype=float), 0
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    if total == 0:
        norm = np.zeros_like(counts, dtype=float)
    else:
        norm = counts.astype(float) / float(total)
        if abs(norm.sum() - 1.0) > 1e-6:
            raise AssertionError("Histogram failed to normalize to 1.0")
    return norm, int(total)


def load_packets(path: Path, limit: Optional[int]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit)
    df = df.dropna(subset=["frame_number", "ts", "proto", "src_ip", "dst_ip"])
    df["frame_number"] = df["frame_number"].astype("Int64")
    df["src_port"] = df["src_port"].fillna(0).astype("Int32")
    df["dst_port"] = df["dst_port"].fillna(0).astype("Int32")
    df["pkt_len"] = df["pkt_len"].fillna(0).astype("Int32")
    for flag_col in ["tcp_syn", "tcp_ack", "tcp_fin", "tcp_rst", "tcp_psh"]:
        df[flag_col] = df[flag_col].fillna(0).astype("Int8")
    df["tcp_window"] = df["tcp_window"].fillna(0).astype("Int32")
    df["payload_hex"] = df["payload_hex"].fillna("").astype("string")
    df["proto"] = df["proto"].astype("Int32")
    return df


def load_payload_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    if "frame_number" not in df.columns:
        raise ValueError("Payload table must contain 'frame_number' column")
    df = df.fillna({"rich_payload_hex": "", "rich_payload_len": 0})
    df["frame_number"] = df["frame_number"].astype("Int64")
    if "rich_payload_len" in df.columns:
        df["rich_payload_len"] = df["rich_payload_len"].fillna(0).astype("Int32")
    for col in df.columns:
        if col == "frame_number":
            continue
        if df[col].dtype == object:
            df[col] = df[col].astype("string")
    return df


def merge_payload(df: pd.DataFrame, payload_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if payload_df is None:
        df = df.copy()
        df["payload_hex"] = df["payload_hex"].fillna("").astype("string")
        df["payload_len"] = (df["payload_hex"].str.len().fillna(0) // 2).astype("Int32")
        return df

    merged = df.merge(payload_df, on="frame_number", how="left")
    if "rich_payload_hex" in merged.columns:
        merged["payload_hex"] = merged["rich_payload_hex"].fillna(merged["payload_hex"]).fillna("")
    else:
        merged["payload_hex"] = merged["payload_hex"].fillna("")
    merged["payload_hex"] = merged["payload_hex"].astype("string")

    if "rich_payload_len" in merged.columns:
        fallback_len = (merged["payload_hex"].str.len().fillna(0) // 2).astype("Int32")
        rich_len = merged["rich_payload_len"].fillna(0).astype("Int32")
        merged["payload_len"] = rich_len.where(rich_len.gt(0), fallback_len).astype("Int32")
    else:
        merged["payload_len"] = (merged["payload_hex"].str.len().fillna(0) // 2).astype("Int32")
    return merged


def bytes_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / float(len(data))
    return float(-np.sum(probs * np.log2(probs)))


def bytes_to_ascii(data: bytes) -> str:
    if not data:
        return ""
    chars = []
    for b in data:
        printable = any(lo <= b <= hi for lo, hi in ASCII_RANGES)
        if printable:
            chars.append(chr(b) if 32 <= b <= 126 else " ")
        else:
            chars.append(".")
    return "".join(chars)


def format_bytes(num_bytes: float) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.2f} GiB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / (1024 ** 2):.2f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KiB"
    return f"{num_bytes:.0f} B"


def get_column_array(df: pd.DataFrame, column: str, length: int) -> np.ndarray:
    if column in df.columns:
        return df[column].fillna("").astype(str).to_numpy(dtype=object)
    return np.array([""] * length, dtype=object)


def select_event_indices(
    indices: np.ndarray,
    payload_lengths: np.ndarray,
    max_events: int,
) -> List[int]:
    if not len(indices):
        return []
    selected: List[int] = []
    selected.append(int(indices[0]))
    if len(indices) > 1:
        selected.append(int(indices[-1]))

    # prioritize large payloads
    payload_sorted = sorted(indices, key=lambda i: payload_lengths[int(i)], reverse=True)
    for idx in payload_sorted:
        idx_int = int(idx)
        if idx_int not in selected:
            selected.append(idx_int)
        if len(selected) >= max_events:
            break

    # fill remaining with chronological order
    if len(selected) < max_events:
        for idx in indices:
            idx_int = int(idx)
            if idx_int not in selected:
                selected.append(idx_int)
            if len(selected) >= max_events:
                break

    selected.sort()
    return selected[:max_events]


def build_event_dict(
    idx: int,
    offset_ms: float,
    direction: str,
    proto_name: str,
    pkt_len: float,
    payload_len: int,
    flags: List[str],
    tcp_window: float,
    previews: Dict[str, str],
    app_strings: List[str],
) -> Dict[str, object]:
    event = {
        "offset_ms": round(offset_ms, 3),
        "direction": direction,
        "proto": proto_name,
        "len": int(pkt_len),
        "payload_len": int(payload_len),
        "flags": flags,
    }
    if tcp_window > 0:
        event["tcp_window"] = int(tcp_window)
    if previews.get("preview"):
        event["preview"] = previews["preview"]
    if app_strings:
        event["application"] = " | ".join(app_strings)
    event["text"] = " ".join(
        part
        for part in [
            f"[+{offset_ms:.2f}ms]",
            direction,
            proto_name,
            f"len={int(pkt_len)}",
            f"payload={int(payload_len)}",
            f"flags={'|'.join(flags) if flags else '-'}",
            f"win={int(tcp_window)}" if tcp_window > 0 else "",
            (" | ".join(app_strings)) if app_strings else "",
            f'preview="{previews["preview"]}"' if previews.get("preview") else "",
        ]
        if part
    )
    return event


def process_group(
    group: pd.DataFrame,
    key: Tuple,
    window_sec: float,
    stride_sec: float,
    pkt_bins: Sequence[float],
    iat_bins_ms: Sequence[float],
    dataset: str,
    capture_id: str,
    rich_opts: RichOptions,
) -> List[Dict]:
    group = group.sort_values("ts").reset_index(drop=True)
    n_packets = len(group)
    if n_packets == 0:
        return []

    ts = group["ts"].to_numpy(dtype=float)
    pkt_len = group["pkt_len"].to_numpy(dtype=float)
    direction = group["direction"].to_numpy(dtype=np.int8) > 0
    syn = group["tcp_syn"].to_numpy(dtype=np.int8)
    ack = group["tcp_ack"].to_numpy(dtype=np.int8)
    fin = group["tcp_fin"].to_numpy(dtype=np.int8)
    rst = group["tcp_rst"].to_numpy(dtype=np.int8)
    psh = group["tcp_psh"].to_numpy(dtype=np.int8)
    tcp_window = group["tcp_window"].to_numpy(dtype=float)
    payload_len_arr = group["payload_len"].to_numpy(dtype=np.int64)
    payload_bytes_arr = group["payload_bytes"].tolist() if "payload_bytes" in group.columns else [b"" for _ in range(n_packets)]

    proto = int(group["proto"].iloc[0])
    proto_name = PROTO_NAMES.get(proto, str(proto))
    src_ip, src_port, dst_ip, dst_port, _ = key
    tuple5_str = f"{src_ip},{dst_ip},{src_port},{dst_port},{proto}"

    frame_protocols = get_column_array(group, "rich_frame_protocols", n_packets)
    http_method = get_column_array(group, "rich_http_method", n_packets)
    http_uri = get_column_array(group, "rich_http_uri", n_packets)
    http_host = get_column_array(group, "rich_http_host", n_packets)
    http_user_agent = get_column_array(group, "rich_http_user_agent", n_packets)
    http_status_code = get_column_array(group, "rich_http_status_code", n_packets)
    http_status_phrase = get_column_array(group, "rich_http_status_phrase", n_packets)
    dns_query = get_column_array(group, "rich_dns_query", n_packets)
    dns_qtype = get_column_array(group, "rich_dns_qtype", n_packets)
    dns_answers = get_column_array(group, "rich_dns_answers", n_packets)
    tls_server_name = get_column_array(group, "rich_tls_server_name", n_packets)
    tls_handshake_type = get_column_array(group, "rich_tls_handshake_type", n_packets)
    tls_content_type = get_column_array(group, "rich_tls_content_type", n_packets)
    tls_version = get_column_array(group, "rich_tls_version", n_packets)

    windows: List[Dict] = []
    start_time = float(ts.min())
    current = math.floor(start_time / stride_sec) * stride_sec
    end_time = float(ts.max())
    start_idx = 0
    end_idx = 0

    while current <= end_time:
        win_end = current + window_sec
        while start_idx < n_packets and ts[start_idx] < current:
            start_idx += 1
        while end_idx < n_packets and ts[end_idx] < win_end:
            end_idx += 1

        if end_idx <= start_idx:
            current += stride_sec
            continue

        idx_slice = slice(start_idx, end_idx)
        win_indices = np.arange(start_idx, end_idx, dtype=int)
        win_ts = ts[idx_slice]
        win_pkt_len = pkt_len[idx_slice]
        fwd_mask = direction[idx_slice]
        bwd_mask = ~fwd_mask
        win_syn = syn[idx_slice]
        win_ack = ack[idx_slice]
        win_fin = fin[idx_slice]
        win_rst = rst[idx_slice]
        win_psh = psh[idx_slice]
        win_window = tcp_window[idx_slice]
        total_packets = win_ts.size
        total_bytes = float(win_pkt_len.sum())

        pkt_cnt_fwd = int(fwd_mask.sum())
        pkt_cnt_bwd = int(bwd_mask.sum())
        byte_cnt_fwd = float(win_pkt_len[fwd_mask].sum())
        byte_cnt_bwd = float(win_pkt_len[bwd_mask].sum())

        flow_dur_ms = 0.0
        if win_ts.size > 1:
            flow_dur_ms = float((win_ts.max() - win_ts.min()) * 1000.0)

        pkt_size_hist, pkt_hist_sum = build_histogram(win_pkt_len, pkt_bins)

        if win_ts.size > 1:
            sorted_ts = np.sort(win_ts)
            iat = np.diff(sorted_ts) * 1000.0
        else:
            iat = np.empty(0, dtype=float)
        iat_hist, iat_hist_sum = build_histogram(iat, iat_bins_ms)

        tcp_packets = total_packets if proto == 6 else 0
        tcp_syn_ratio = safe_ratio(win_syn.sum(), tcp_packets)
        tcp_fin_ratio = safe_ratio(win_fin.sum(), tcp_packets)
        tcp_rst_ratio = safe_ratio(win_rst.sum(), tcp_packets)
        tcp_ack_ratio = safe_ratio(win_ack.sum(), tcp_packets)
        tcp_psh_ratio = safe_ratio(win_psh.sum(), tcp_packets)

        syn_fwd = int((win_syn & fwd_mask.astype(np.int8)).sum())
        synack_bwd = int(((win_syn > 0) & (win_ack > 0) & bwd_mask.astype(np.int8)).sum())
        ack_fwd = int((win_ack & fwd_mask.astype(np.int8)).sum())
        handshake_score = min(syn_fwd, synack_bwd, ack_fwd)

        payload_total_len = int(payload_len_arr[idx_slice].sum())

        pkt_len_mean = float(win_pkt_len.mean()) if win_pkt_len.size else 0.0
        pkt_len_std = float(win_pkt_len.std(ddof=0)) if win_pkt_len.size > 1 else 0.0
        pkt_len_min = float(win_pkt_len.min()) if win_pkt_len.size else 0.0
        pkt_len_max = float(win_pkt_len.max()) if win_pkt_len.size else 0.0

        iat_mean = float(iat.mean()) if iat.size else 0.0
        iat_std = float(iat.std(ddof=0)) if iat.size > 1 else 0.0
        iat_min = float(iat.min()) if iat.size else 0.0
        iat_max = float(iat.max()) if iat.size else 0.0

        tcp_window_mean = float(win_window.mean()) if win_window.size else 0.0
        tcp_window_max = float(win_window.max()) if win_window.size else 0.0

        record = {
            "tuple5": tuple5_str,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": int(src_port),
            "dst_port": int(dst_port),
            "proto": proto,
            "proto_name": proto_name,
            "win_start_ts": current,
            "win_end_ts": win_end,
            "packet_count": int(total_packets),
            "byte_count": total_bytes,
            "pkt_cnt_fwd": pkt_cnt_fwd,
            "pkt_cnt_bwd": pkt_cnt_bwd,
            "byte_cnt_fwd": byte_cnt_fwd,
            "byte_cnt_bwd": byte_cnt_bwd,
            "flow_dur_ms": flow_dur_ms,
            "fwd_bwd_ratio_pkt": safe_ratio(pkt_cnt_fwd, pkt_cnt_bwd),
            "fwd_bwd_ratio_byte": safe_ratio(byte_cnt_fwd, byte_cnt_bwd),
            "tcp_syn_ratio": tcp_syn_ratio,
            "tcp_fin_ratio": tcp_fin_ratio,
            "tcp_rst_ratio": tcp_rst_ratio,
            "tcp_ack_ratio": tcp_ack_ratio,
            "tcp_psh_ratio": tcp_psh_ratio,
            "tcp_handshake_score": handshake_score,
            "tcp_rst_count": int(win_rst.sum()),
            "tcp_psh_count": int(win_psh.sum()),
            "tcp_window_mean": tcp_window_mean,
            "tcp_window_max": tcp_window_max,
            "pkt_len_mean": pkt_len_mean,
            "pkt_len_std": pkt_len_std,
            "pkt_len_min": pkt_len_min,
            "pkt_len_max": pkt_len_max,
            "iat_mean_ms": iat_mean,
            "iat_std_ms": iat_std,
            "iat_min_ms": iat_min,
            "iat_max_ms": iat_max,
            "payload_total_len": payload_total_len,
            "dataset": dataset,
            "capture_id": capture_id,
        }

        for idx_hist, value in enumerate(pkt_size_hist):
            record[f"pkt_size_hist_{idx_hist}"] = float(value)
        record["pkt_size_hist_sum"] = pkt_hist_sum

        for idx_hist, value in enumerate(iat_hist):
            record[f"iat_hist_{idx_hist}"] = float(value)
        record["iat_hist_sum"] = iat_hist_sum

        # Rich augmentations
        if rich_opts.enabled:
            payload_bytes_segment = payload_bytes_arr[start_idx:end_idx]
            payload_bytes = b"".join(payload_bytes_segment)
            ascii_count = sum(
                1
                for b in payload_bytes
                if any(lo <= b <= hi for lo, hi in ASCII_RANGES)
            )
            payload_entropy = bytes_entropy(payload_bytes)
            ascii_ratio = safe_ratio(ascii_count, len(payload_bytes))
            head_bytes = payload_bytes[: rich_opts.payload_head_bytes]
            tail_bytes = (
                payload_bytes
                if len(payload_bytes) <= rich_opts.payload_tail_bytes
                else payload_bytes[-rich_opts.payload_tail_bytes :]
            )
            payload_truncated = len(payload_bytes) > (
                rich_opts.payload_head_bytes + rich_opts.payload_tail_bytes
            )
            assert len(head_bytes) <= rich_opts.payload_head_bytes
            assert len(tail_bytes) <= rich_opts.payload_tail_bytes

            selected_indices = select_event_indices(win_indices, payload_len_arr, rich_opts.max_events)
            event_dicts: List[Dict[str, object]] = []
            event_offsets: List[float] = []
            for idx in selected_indices:
                offset_ms = float((ts[idx] - current) * 1000.0)
                event_offsets.append(offset_ms)
                direction_str = "FWD" if direction[idx] else "BWD"
                flags = []
                if syn[idx]:
                    flags.append("SYN")
                if ack[idx]:
                    flags.append("ACK")
                if fin[idx]:
                    flags.append("FIN")
                if rst[idx]:
                    flags.append("RST")
                if psh[idx]:
                    flags.append("PSH")

                app_parts: List[str] = []
                if http_method[idx]:
                    uri_part = http_uri[idx] or "/"
                    part = f"HTTP {http_method[idx]} {uri_part}"
                    if http_host[idx]:
                        part += f" host={http_host[idx]}"
                    app_parts.append(part)
                elif http_status_code[idx]:
                    part = f"HTTP {http_status_code[idx]}"
                    if http_status_phrase[idx]:
                        part += f" {http_status_phrase[idx]}"
                    app_parts.append(part)
                if dns_query[idx]:
                    qtype = dns_qtype[idx] or "Q"
                    part = f"DNS {qtype} {dns_query[idx]}"
                    if dns_answers[idx]:
                        part += f" ans={dns_answers[idx]}"
                    app_parts.append(part)
                if tls_server_name[idx]:
                    part = f"SNI={tls_server_name[idx]}"
                    if tls_handshake_type[idx]:
                        part += f" type={tls_handshake_type[idx]}"
                    if tls_version[idx]:
                        part += f" ver={tls_version[idx]}"
                    app_parts.append(f"TLS {part}")
                elif tls_content_type[idx]:
                    app_parts.append(f"TLS content={tls_content_type[idx]}")
                elif frame_protocols[idx]:
                    app_parts.append(f"Protocols={frame_protocols[idx]}")

                preview = ""
                if payload_bytes_arr[idx]:
                    preview = bytes_to_ascii(payload_bytes_arr[idx][:32])

                event_dicts.append(
                    build_event_dict(
                        idx=idx,
                        offset_ms=offset_ms,
                        direction=direction_str,
                        proto_name=proto_name,
                        pkt_len=pkt_len[idx],
                        payload_len=int(payload_len_arr[idx]),
                        flags=flags,
                        tcp_window=float(tcp_window[idx]),
                        previews={"preview": preview} if preview else {},
                        app_strings=app_parts,
                    )
                )

            assert event_offsets == sorted(event_offsets)

            proto_hints = set()
            for idx in win_indices:
                for token in frame_protocols[idx].split(">"):
                    token = token.strip()
                    if token:
                        proto_hints.add(token)

            record.update(
                {
                    "payload_entropy": payload_entropy,
                    "payload_ascii_ratio": ascii_ratio,
                    "rich_payload_head_hex": head_bytes.hex(),
                    "rich_payload_head_ascii": bytes_to_ascii(head_bytes),
                    "rich_payload_tail_hex": tail_bytes.hex(),
                    "rich_payload_tail_ascii": bytes_to_ascii(tail_bytes),
                    "rich_payload_truncated": bool(payload_truncated),
                    "rich_event_count": len(event_dicts),
                    "rich_events": json.dumps(event_dicts, ensure_ascii=False),
                    "rich_protocols": ",".join(sorted(proto_hints)) if proto_hints else "",
                }
            )
        else:
            record.update(
                {
                    "payload_entropy": 0.0,
                    "payload_ascii_ratio": 0.0,
                    "rich_payload_head_hex": "",
                    "rich_payload_head_ascii": "",
                    "rich_payload_tail_hex": "",
                    "rich_payload_tail_ascii": "",
                    "rich_payload_truncated": False,
                    "rich_event_count": 0,
                    "rich_events": "[]",
                    "rich_protocols": "",
                }
            )

        windows.append(record)
        current += stride_sec

    return windows


def add_context_features(
    df: pd.DataFrame,
    stride_sec: float,
    rich_opts: RichOptions,
) -> None:
    window_counts = [
        max(1, int(math.ceil(window / max(stride_sec, 1e-6))))
        for window in rich_opts.context_windows_sec
    ]
    labels = ["10s", "60s"]

    df.sort_values(["tuple5", "win_start_ts"], inplace=True)
    grouped = df.groupby("tuple5", sort=False)
    for window_count, label in zip(window_counts, labels):
        df[f"packet_count_{label}"] = (
            grouped["packet_count"]
            .rolling(window=window_count, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df[f"byte_count_{label}"] = (
            grouped["byte_count"]
            .rolling(window=window_count, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df[f"payload_total_len_{label}"] = (
            grouped["payload_total_len"]
            .rolling(window=window_count, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    def build_summary(row: pd.Series) -> str:
        base = (
            f"{row['proto_name']} window {row['win_start_ts']:.3f}-{row['win_end_ts']:.3f}s: "
            f"{int(row['packet_count'])} pkts ({format_bytes(row['byte_count'])}) "
            f"fwd/bwd bytes={format_bytes(row['byte_cnt_fwd'])}/{format_bytes(row['byte_cnt_bwd'])} "
            f"summarized entropy={row['payload_entropy']:.2f} bits"
        )
        ten_sec = (
            f"last {int(rich_opts.context_windows_sec[0])}s: "
            f"{int(row.get('packet_count_10s', 0))} pkts / {format_bytes(row.get('byte_count_10s', 0))}"
        )
        sixty_sec = (
            f"last {int(rich_opts.context_windows_sec[1])}s: "
            f"{int(row.get('packet_count_60s', 0))} pkts / {format_bytes(row.get('byte_count_60s', 0))}"
        )
        extras = []
        if row.get("rich_event_count", 0):
            extras.append(f"{int(row['rich_event_count'])} key events captured")
        if row.get("rich_protocols"):
            extras.append(f"protocols={row['rich_protocols']}")
        extras.append(f"payload_ascii_ratio={row['payload_ascii_ratio']:.2f}")
        return " | ".join([base, ten_sec, sixty_sec] + extras)

    df["rich_context_summary"] = df.apply(build_summary, axis=1)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    compute_standard = bool(cfg.get("compute_standard", True))
    compute_rich = bool(cfg.get("compute_rich", False))
    if not compute_standard and not compute_rich:
        raise ValueError("At least one of compute_standard or compute_rich must be enabled.")
    if compute_rich and not args.rich_out:
        raise ValueError("compute_rich requires --rich-out to be specified.")

    packets_path = Path(args.packets)
    payload_path = Path(args.payload) if args.payload else None

    df_packets = load_packets(packets_path, args.limit)
    payload_df = load_payload_table(payload_path) if payload_path else None
    df_packets = merge_payload(df_packets, payload_df)
    df_packets = canonicalize_endpoints(df_packets)

    capture_id = args.capture_id or Path(packets_path.stem).name.replace("_packets", "")

    rich_opts = RichOptions(
        enabled=compute_rich,
        payload_head_bytes=int(cfg.get("payload_head_bytes", 1024)),
        payload_tail_bytes=int(cfg.get("payload_tail_bytes", 1024)),
        max_events=int(cfg.get("max_events_per_window", 10)),
        context_windows_sec=(float(cfg["context_windows_sec"][0]), float(cfg["context_windows_sec"][1])),
    )

    window_sec = float(cfg["window_sec"])
    stride_sec = float(cfg["stride_sec"])
    pkt_bins = list(map(float, cfg["pkt_size_bins"]))
    iat_bins_ms = list(map(float, cfg["iat_bins_ms"]))

    group_cols = ["canon_src_ip", "canon_src_port", "canon_dst_ip", "canon_dst_port", "proto"]
    records: List[Dict] = []
    for key, group in df_packets.groupby(group_cols, sort=False):
        records.extend(
            process_group(
                group=group,
                key=key,
                window_sec=window_sec,
                stride_sec=stride_sec,
                pkt_bins=pkt_bins,
                iat_bins_ms=iat_bins_ms,
                dataset=args.dataset,
                capture_id=capture_id,
                rich_opts=rich_opts,
            )
        )

    if not records:
        raise RuntimeError("No windows were generated; check input data or configuration.")

    full_df = pd.DataFrame.from_records(records)
    add_context_features(full_df, stride_sec=stride_sec, rich_opts=rich_opts)

    # Ensure deterministic ordering
    full_df.sort_values(["win_start_ts", "tuple5"], inplace=True)

    # Paths
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if compute_standard:
        standard_cols = [col for col in full_df.columns if not col.startswith("rich_")]
        standard_df = full_df[standard_cols].copy()
        standard_df.to_parquet(out_path, index=False)
        print(f"Wrote {len(standard_df):,} windows to {out_path}")
    else:
        print("Structured output disabled by configuration.")

    if compute_rich:
        rich_path = Path(args.rich_out)
        rich_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(rich_path, index=False)
        print(f"Wrote enriched windows to {rich_path}")


if __name__ == "__main__":
    main()
