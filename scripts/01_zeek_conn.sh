#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET" >&2
  exit 1
fi

DATASET="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_PCAP="${ROOT_DIR}/data/raw/${DATASET}/capture.pcap"
STAGING_DIR="${ROOT_DIR}/data/staging"
OUTPUT_LOG="${STAGING_DIR}/${DATASET}_conn.log"

if [[ ! -f "${RAW_PCAP}" ]]; then
  echo "Missing input PCAP: ${RAW_PCAP}" >&2
  exit 1
fi

mkdir -p "${STAGING_DIR}"

TMPDIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMPDIR}"
}
trap cleanup EXIT

pushd "${TMPDIR}" >/dev/null

zeek -r "${RAW_PCAP}" LogAscii::use_json=T "Site::local_nets += { 0.0.0.0/0 }"

if [[ ! -f conn.log ]]; then
  echo "zeek did not produce conn.log as expected" >&2
  exit 1
fi

mv conn.log "${OUTPUT_LOG}"

popd >/dev/null

echo "Wrote ${OUTPUT_LOG}"
