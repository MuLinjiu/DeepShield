#!/usr/bin/env python3
"""Extract per-packet data from PCAP using tshark and store as Parquet."""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import pandas as pd


TSHARK_FIELDS = [
    "frame.number",
    "frame.time_epoch",
    "ip.proto",
    "ip.src",
    "ip.dst",
    "tcp.srcport",
    "udp.srcport",
    "tcp.dstport",
    "udp.dstport",
    "frame.len",
    "tcp.flags.syn",
    "tcp.flags.ack",
    "tcp.flags.fin",
    "tcp.flags.reset",
    "tcp.flags.push",
    "tcp.window_size_value",
    "data.data",
]

OUTPUT_FIELDS = [
    "frame_number",
    "ts",
    "proto",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "pkt_len",
    "tcp_syn",
    "tcp_ack",
    "tcp_fin",
    "tcp_rst",
    "tcp_psh",
    "tcp_window",
    "payload_hex",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset key (e.g., CICIDS2017)")
    parser.add_argument(
        "--pcap",
        help="Optional PCAP path (defaults to data/raw/{dataset}/capture.pcap)",
    )
    parser.add_argument(
        "--out",
        help="Optional output Parquet path (defaults to data/staging/{dataset}_packets.parquet)",
    )
    parser.add_argument(
        "--field-size-limit",
        type=int,
        default=0,
        help="Maximum CSV field size accepted (bytes). Use 0 for Python's maxsize (default).",
    )
    return parser.parse_args()


def ensure_tshark() -> None:
    if shutil.which("tshark") is None:
        raise SystemExit("tshark not found. Please install Wireshark/tshark before running this script.")


def build_paths(dataset: str, pcap_arg: str | None, out_arg: str | None) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    staging_dir = root / "data" / "staging"
    pcap_path = Path(pcap_arg) if pcap_arg else root / "data" / "raw" / dataset / "capture.pcap"
    default_out = staging_dir / f"{dataset}_packets.parquet"
    out_path = Path(out_arg) if out_arg else default_out
    return pcap_path, out_path


def run_tshark(pcap_path: Path) -> subprocess.Popen:
    cmd = [
        "tshark",
        "-r",
        str(pcap_path),
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=,",
        "-E",
        "quote=d",
        "-E",
        "occurrence=f",
    ]
    for field in TSHARK_FIELDS:
        cmd.extend(["-e", field])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")


def normalize_flag(value: str | None) -> str:
    if value is None:
        return "0"
    stripped = value.strip().lower()
    if stripped in {"", "0", "false"}:
        return "0"
    return "1"


def process_rows(reader: Iterable[Dict[str, str]]) -> Iterator[Dict[str, object]]:
    for row in reader:
        frame_number = (row.get("frame.number") or "").strip()
        ts = (row.get("frame.time_epoch") or "").strip()
        src_ip = (row.get("ip.src") or "").strip()
        dst_ip = (row.get("ip.dst") or "").strip()
        proto = (row.get("ip.proto") or "").strip()

        if not frame_number or not ts or not src_ip or not dst_ip or not proto:
            continue

        try:
            frame_number_int = int(float(frame_number))
        except ValueError:
            continue

        try:
            ts_val = float(ts)
        except ValueError:
            continue

        try:
            proto_val = int(proto)
        except ValueError:
            continue

        tcp_src = (row.get("tcp.srcport") or "").strip()
        udp_src = (row.get("udp.srcport") or "").strip()
        tcp_dst = (row.get("tcp.dstport") or "").strip()
        udp_dst = (row.get("udp.dstport") or "").strip()

        def safe_int(value: str) -> int:
            value = value.strip()
            if not value:
                return 0
            try:
                return int(float(value))
            except ValueError:
                return 0

        src_port = safe_int(tcp_src or udp_src or "0")
        dst_port = safe_int(tcp_dst or udp_dst or "0")

        payload_hex = (row.get("data.data") or "").replace(":", "").strip().lower()
        pkt_len_value = safe_int(row.get("frame.len") or "0")
        tcp_window = safe_int(row.get("tcp.window_size_value") or "0")

        yield {
            "frame_number": frame_number_int,
            "ts": ts_val,
            "proto": proto_val,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "pkt_len": pkt_len_value,
            "tcp_syn": int(normalize_flag(row.get("tcp.flags.syn"))),
            "tcp_ack": int(normalize_flag(row.get("tcp.flags.ack"))),
            "tcp_fin": int(normalize_flag(row.get("tcp.flags.fin"))),
            "tcp_rst": int(normalize_flag(row.get("tcp.flags.reset"))),
            "tcp_psh": int(normalize_flag(row.get("tcp.flags.push"))),
            "tcp_window": tcp_window,
            "payload_hex": payload_hex,
        }


def main() -> None:
    args = parse_args()
    ensure_tshark()
    pcap_path, out_path = build_paths(args.dataset, args.pcap, args.out)

    if not pcap_path.exists():
        raise SystemExit(f"Missing input PCAP: {pcap_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    limit = args.field_size_limit
    if limit is None or limit == 0:
        csv.field_size_limit(sys.maxsize)
    else:
        csv.field_size_limit(limit)

    rows: List[Dict[str, object]] = []
    with run_tshark(pcap_path) as proc:
        if proc.stdout is None:
            raise SystemExit("Failed to read tshark stdout.")
        reader = csv.DictReader(proc.stdout)
        rows.extend(process_rows(reader))
        return_code = proc.wait()
        if return_code != 0:
            raise SystemExit(f"tshark exited with code {return_code}")

    if not rows:
        pd.DataFrame(columns=OUTPUT_FIELDS).to_parquet(out_path, index=False)
    else:
        df = pd.DataFrame(rows, columns=OUTPUT_FIELDS)
        df = df.astype(
            {
                "frame_number": "int64",
                "ts": "float64",
                "proto": "int32",
                "src_ip": "string",
                "dst_ip": "string",
                "src_port": "int32",
                "dst_port": "int32",
                "pkt_len": "int32",
                "tcp_syn": "int8",
                "tcp_ack": "int8",
                "tcp_fin": "int8",
                "tcp_rst": "int8",
                "tcp_psh": "int8",
                "tcp_window": "int32",
                "payload_hex": "string",
            }
        )
        df.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
