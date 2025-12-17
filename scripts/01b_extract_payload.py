#!/usr/bin/env python3
"""Extract packet payload/context features using tshark and write Parquet."""

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
    "data.data",
    "tcp.payload",
    "udp.payload",
    "frame.protocols",
    "http.request.method",
    "http.request.uri",
    "http.host",
    "http.user_agent",
    "http.response.code",
    "http.response.phrase",
    "dns.qry.name",
    "dns.qry.type",
    "dns.a",
    "dns.aaaa",
    "tls.handshake.extensions_server_name",
    "tls.handshake.type",
    "tls.record.content_type",
    "ssl.record.version",
]

OUTPUT_COLUMNS = [
    "frame_number",
    "rich_payload_hex",
    "rich_payload_len",
    "rich_frame_protocols",
    "rich_http_method",
    "rich_http_uri",
    "rich_http_host",
    "rich_http_user_agent",
    "rich_http_status_code",
    "rich_http_status_phrase",
    "rich_dns_query",
    "rich_dns_qtype",
    "rich_dns_answers",
    "rich_tls_server_name",
    "rich_tls_handshake_type",
    "rich_tls_content_type",
    "rich_tls_version",
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
        help="Optional output Parquet path (defaults to data/staging/{dataset}_packets_payload.parquet)",
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
    out_path = Path(out_arg) if out_arg else staging_dir / f"{dataset}_packets_payload.parquet"
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


def clean(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def stream_rows(reader: Iterable[Dict[str, str]]) -> Iterator[Dict[str, str]]:
    for row in reader:
        frame_number_raw = clean(row.get("frame.number"))
        if not frame_number_raw:
            continue
        try:
            frame_number = int(float(frame_number_raw))
        except ValueError:
            continue

        payload_hex = clean(row.get("data.data"))
        if not payload_hex:
            payload_hex = clean(row.get("tcp.payload"))
        if not payload_hex:
            payload_hex = clean(row.get("udp.payload"))
        payload_hex = payload_hex.replace(":", "").lower()
        payload_len = len(payload_hex) // 2

        dns_answers: List[str] = []
        dns_a = clean(row.get("dns.a")).replace(",", ";")
        dns_aaaa = clean(row.get("dns.aaaa")).replace(",", ";")
        if dns_a:
            dns_answers.append(dns_a)
        if dns_aaaa:
            dns_answers.append(dns_aaaa)

        yield {
            "frame_number": frame_number,
            "rich_payload_hex": payload_hex,
            "rich_payload_len": payload_len,
            "rich_frame_protocols": clean(row.get("frame.protocols")).replace(":", ">"),
            "rich_http_method": clean(row.get("http.request.method")),
            "rich_http_uri": clean(row.get("http.request.uri")),
            "rich_http_host": clean(row.get("http.host")),
            "rich_http_user_agent": clean(row.get("http.user_agent")),
            "rich_http_status_code": clean(row.get("http.response.code")),
            "rich_http_status_phrase": clean(row.get("http.response.phrase")),
            "rich_dns_query": clean(row.get("dns.qry.name")),
            "rich_dns_qtype": clean(row.get("dns.qry.type")),
            "rich_dns_answers": ";".join(filter(None, dns_answers)),
            "rich_tls_server_name": clean(row.get("tls.handshake.extensions_server_name")),
            "rich_tls_handshake_type": clean(row.get("tls.handshake.type")),
            "rich_tls_content_type": clean(row.get("tls.record.content_type")),
            "rich_tls_version": clean(row.get("ssl.record.version")),
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
        rows.extend(stream_rows(reader))
        return_code = proc.wait()
        if return_code != 0:
            raise SystemExit(f"tshark exited with code {return_code}")

    if not rows:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_parquet(out_path, index=False)
    else:
        df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        df.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
