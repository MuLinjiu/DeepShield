#!/usr/bin/env bash
# 在测试集上评估 baseline 与指定 checkpoint（默认 checkpoint-1100）。
# 默认测试集可通过 TEST 环境变量覆盖（当前默认 natural5k）。

set -euo pipefail

TEST="${TEST:-data/llm_input_enriched_test_natural5k.jsonl}"
BASE="${BASE:-Qwen/Qwen3-4B-Instruct-2507}"
CKPT_ROOT="${CKPT_ROOT:-lora-qwen3-12h}"
CKPT_NAME="${CKPT_NAME:-checkpoint-1100}"
CKPT_DIR="${CKPT_ROOT%/}/${CKPT_NAME}"
BASELINE_DIR="${BASELINE_DIR:-${CKPT_ROOT%/}/baseline}"
BATCH="${BATCH:-32}"
MAX_NEW="${MAX_NEW:-512}"
TEMP="${TEMP:-0}"
IP_MASK="${IP_MASK:-0.0}"
METRIC="${METRIC:-weighted_f1}"
EVAL_DEVICES="${EVAL_DEVICES:-}"
FORCE="${FORCE:-0}"
USE_ALL_GPUS="${USE_ALL_GPUS:-0}"

if [ ! -f "$TEST" ]; then
  echo "[error] 测试集不存在: $TEST" >&2
  exit 1
fi

if [ ! -d "$CKPT_DIR" ]; then
  echo "[error] 未找到 checkpoint 目录: $CKPT_DIR" >&2
  exit 1
fi

detect_devices() {
  # 输出以空格分隔的设备列表
  if [ -n "$EVAL_DEVICES" ]; then
    echo "$EVAL_DEVICES"
    return
  fi
  if [ "$USE_ALL_GPUS" != "1" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "$CUDA_VISIBLE_DEVICES"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus | awk -F: '{print NR-1}' | paste -sd' ' -
    return
  fi
  echo ""
}

read -ra DEVICES <<< "$(detect_devices | tr ',' ' ')"

if [ "${#DEVICES[@]}" -ge 1 ] && [ -n "${DEVICES[0]}" ]; then
  echo "[info] 可用设备: ${DEVICES[*]}"
else
  echo "[warn] 未检测到 GPU，将在 CPU 上运行，可能较慢。"
  DEVICES=("")
fi

echo "[info] 测试集: $TEST"
echo "[info] 基座模型: $BASE"
echo "[info] 检查点: $CKPT_DIR"
echo "[info] Baseline 输出目录: $BASELINE_DIR"
echo "[info] 评估指标: $METRIC"
echo

run_eval() {
  local name="$1" adapter="$2" out_path="$3" log_path="$4" device="$5"

  if [ -f "$out_path" ] && [ "$FORCE" != "1" ]; then
    echo "[info] $name 输出已存在，跳过：$out_path (FORCE=1 可重跑)"
    return 0
  fi

  echo "[info] 开始评估 $name (device=${device:-CPU})..."
  mkdir -p "$(dirname "$out_path")"

  cmd=(python train_lora_netflow_refined.py
       --mode eval
       --eval_path "$TEST"
       --base_model "$BASE"
       --eval_batch_size "$BATCH"
       --eval_temperature "$TEMP"
       --eval_max_new_tokens "$MAX_NEW"
       --ip_mask_rate "$IP_MASK"
       --eval_output_path "$out_path")

  if [ -n "$adapter" ]; then
    cmd+=(--adapter "$adapter")
  fi

  (
    set -euo pipefail
    if [ -n "$device" ]; then
      export CUDA_VISIBLE_DEVICES="$device"
    else
      unset CUDA_VISIBLE_DEVICES
    fi

    PYTHONUNBUFFERED=1 stdbuf -oL -eL "${cmd[@]}" | tee "$log_path"
  )
}

BASELINE_OUT="$BASELINE_DIR/test_predictions.jsonl"
BASELINE_LOG="$BASELINE_DIR/test_log.txt"
CKPT_OUT="$CKPT_DIR/test_predictions.jsonl"
CKPT_LOG="$CKPT_DIR/test_log.txt"

# 选择设备：若有至少2张 GPU，则并行运行 baseline 与 ckpt
DEV_BASE="${DEVICES[0]:-}"
DEV_CKPT="${DEVICES[1]:-${DEVICES[0]:-}}"

echo "[info] baseline 使用设备: ${DEV_BASE:-CPU}"
echo "[info] ckpt(${CKPT_NAME}) 使用设备: ${DEV_CKPT:-CPU}"

PIDS=()
NAMES=()

need_base=1
need_ckpt=1
if [ -f "$BASELINE_OUT" ] && [ "$FORCE" != "1" ]; then
  need_base=0
fi
if [ -f "$CKPT_OUT" ] && [ "$FORCE" != "1" ]; then
  need_ckpt=0
fi

task_count=$((need_base + need_ckpt))

if [ $task_count -eq 0 ]; then
  echo "[info] baseline 与 $CKPT_NAME 的测试输出均已存在；设置 FORCE=1 可重跑。"
elif [ $task_count -eq 2 ] && [ -n "$DEV_CKPT" ] && [ "${#DEVICES[@]}" -ge 2 ]; then
  echo "[info] 检测到 >=2 张 GPU，将并行评估 baseline 与 $CKPT_NAME"
  run_eval "baseline (no adapter)" "" "$BASELINE_OUT" "$BASELINE_LOG" "$DEV_BASE" &
  PIDS+=("$!"); NAMES+=("baseline")
  run_eval "$CKPT_NAME" "$CKPT_DIR" "$CKPT_OUT" "$CKPT_LOG" "$DEV_CKPT" &
  PIDS+=("$!"); NAMES+=("$CKPT_NAME")

  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[error] 任务失败: ${NAMES[$i]}" >&2
      exit 1
    fi
  done
else
  echo "[info] 将按顺序评估：baseline_needed=$need_base, ckpt_needed=$need_ckpt"
  if [ $need_base -eq 1 ]; then
    run_eval "baseline (no adapter)" "" "$BASELINE_OUT" "$BASELINE_LOG" "$DEV_BASE"
  else
    echo "[info] baseline 输出已存在，跳过 (FORCE=1 可重跑)：$BASELINE_OUT"
  fi
  if [ $need_ckpt -eq 1 ]; then
    run_eval "$CKPT_NAME" "$CKPT_DIR" "$CKPT_OUT" "$CKPT_LOG" "$DEV_CKPT"
  else
    echo "[info] $CKPT_NAME 输出已存在，跳过 (FORCE=1 可重跑)：$CKPT_OUT"
  fi
fi

echo
echo "[info] 评估完成，汇总 $METRIC ..."
METRIC_NAME="$METRIC" BASELINE_LOG="$BASELINE_LOG" CKPT_LOG="$CKPT_LOG" CKPT_LABEL="$CKPT_NAME" python - <<'PY'
import json
import os
import re

metric = os.environ.get("METRIC_NAME", "weighted_f1")
logs = {
    "baseline": os.environ.get("BASELINE_LOG"),
    os.environ.get("CKPT_LABEL", "checkpoint"): os.environ.get("CKPT_LOG"),
}

def extract_metric(path, key):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    patterns = [
        rf'"{re.escape(key)}"\s*:\s*([-+]?\d*\.?\d+)',
        rf'{re.escape(key.replace("_", " "))}\s*[:=]\s*([-+]?\d*\.?\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None

results = {}
for name, path in logs.items():
    score = extract_metric(path, metric)
    results[name] = (score, path)

print(f"[summary] metric={metric}")
for name, (score, path) in results.items():
    if score is None:
        print(f"  - {name}: 未在日志中找到 {metric} (log={path})")
    else:
        print(f"  - {name}: {metric}={score:.4f} (log={path})")

if all(v[0] is not None for v in results.values()):
    names = list(results.keys())
    diff = results[names[1]][0] - results[names[0]][0]
    print(f"\n[delta] {names[1]} - {names[0]} = {diff:.4f}")
PY

echo "[info] 完成。"
