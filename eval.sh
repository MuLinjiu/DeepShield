#!/usr/bin/env bash
# 并行评估所有 LoRA checkpoint，在验证集上找出 macro_f1 最高的那个。
set -euo pipefail

# 可通过环境变量覆盖这些默认值
VAL="${VAL:-data/val_12h.jsonl}"
BASE="${BASE:-Qwen/Qwen3-4B-Instruct-2507}"
CKPT_ROOT="${CKPT_ROOT:-lora-qwen3-12h}"
BASELINE_DIR="${BASELINE_DIR:-$CKPT_ROOT/baseline}"
BATCH="${BATCH:-32}"
MAX_NEW="${MAX_NEW:-512}"
TEMP="${TEMP:-0}"
IP_MASK="${IP_MASK:-0.0}"

if ! command -v parallel >/dev/null 2>&1; then
  echo "[error] GNU parallel 未安装，请先安装后再运行。" >&2
  exit 1
fi

# 检测可用 GPU 数；若用户已设置 CUDA_VISIBLE_DEVICES，则严格使用该列表
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  # 支持逗号或空格分隔
  read -ra GPU_IDS <<< "$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ')"
  # 过滤空字符串
  TMP=()
  for g in "${GPU_IDS[@]}"; do
    [ -n "$g" ] && TMP+=("$g")
  done
  GPU_IDS=("${TMP[@]}")
  GPU_COUNT=${#GPU_IDS[@]}
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    GPU_COUNT=${GPU_COUNT:-0}
  else
    GPU_COUNT=0
  fi
  # 默认使用编号 0..GPU_COUNT-1
  if [ "$GPU_COUNT" -gt 0 ]; then
    GPU_IDS=($(seq 0 $((GPU_COUNT-1))))
  else
    GPU_IDS=()
  fi
fi

if [ "$GPU_COUNT" -lt 1 ]; then
  echo "[warn] 未检测到 GPU，将在 CPU 上运行，可能很慢。"
  GPU_COUNT=1
  GPU_IDS=()
fi

JOBS="${JOBS:-$GPU_COUNT}"

mapfile -t CKPTS < <(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "[error] 在 $CKPT_ROOT 没有找到 checkpoint-*/ 目录。" >&2
  exit 1
fi

# 跳过已经有 val_predictions.jsonl 的 checkpoint
SKIP_COUNT=0
TO_EVAL=()
for ckpt in "${CKPTS[@]}"; do
  if [ -f "$ckpt/val_predictions.jsonl" ]; then
    echo "[info] $ckpt 已存在 val_predictions.jsonl，跳过评估。"
    SKIP_COUNT=$((SKIP_COUNT + 1))
  else
    TO_EVAL+=("$ckpt")
  fi
done
CKPTS=("${TO_EVAL[@]}")

export VAL BASE BATCH MAX_NEW TEMP IP_MASK GPU_COUNT
# 传递 GPU 列表到子进程（空代表使用整数索引）
GPU_IDS_CSV="${GPU_IDS[*]}"
export GPU_IDS_CSV

echo "[info] 验证集: $VAL"
echo "[info] 基座模型: $BASE"
echo "[info] Checkpoints 待评估数: ${#CKPTS[@]} (已跳过 $SKIP_COUNT)"
echo "[info] 并行作业数: $JOBS (GPU_COUNT=$GPU_COUNT)"
echo "[info] Baseline 输出目录: $BASELINE_DIR"
echo

# 先运行 baseline（无 LoRA adapter）
BASELINE_OUT="$BASELINE_DIR/val_predictions.jsonl"
if [ -f "$BASELINE_OUT" ]; then
  echo "[info] baseline 已存在输出，跳过：$BASELINE_OUT"
else
  echo "[info] 开始运行 baseline（无 LoRA adapter）..."
  mkdir -p "$BASELINE_DIR"
  (
    if [ "${#GPU_IDS[@]}" -gt 0 ]; then
      base_dev="${GPU_IDS[0]}"
      export CUDA_VISIBLE_DEVICES="$base_dev"
      echo "[baseline GPU $base_dev] evaluating base model"
    else
      unset CUDA_VISIBLE_DEVICES
      echo "[baseline CPU] evaluating base model"
    fi
    PYTHONUNBUFFERED=1 stdbuf -oL -eL python train_lora_netflow_refined.py \
      --mode eval \
      --eval_path "$VAL" \
      --base_model "$BASE" \
      --eval_batch_size "$BATCH" \
      --eval_temperature "$TEMP" \
      --eval_max_new_tokens "$MAX_NEW" \
      --ip_mask_rate "$IP_MASK" \
      --eval_output_path "$BASELINE_OUT" \
      | tee "$BASELINE_DIR/val_log.txt"
  )
fi

if [ "${#CKPTS[@]}" -gt 0 ]; then
  parallel --jobs "$JOBS" --lb --halt now,fail=1 --bar '
    ckpt={};
    # 根据任务序号选择设备；若 GPU_IDS_CSV 为空则使用裸索引
    gpu_idx=$(( ({#}-1) % GPU_COUNT ))
    if [ -n "${GPU_IDS_CSV:-}" ]; then
      read -ra IDS <<< "$GPU_IDS_CSV"
      dev=${IDS[$gpu_idx]}
    else
      dev=$gpu_idx
    fi
    export CUDA_VISIBLE_DEVICES=$dev
    echo "[GPU $dev] evaluating $ckpt"
    PYTHONUNBUFFERED=1 stdbuf -oL -eL python train_lora_netflow_refined.py \
      --mode eval \
      --eval_path "$VAL" \
      --base_model "$BASE" \
      --adapter "$ckpt" \
      --eval_batch_size "$BATCH" \
      --eval_temperature "$TEMP" \
      --eval_max_new_tokens "$MAX_NEW" \
      --ip_mask_rate "$IP_MASK" \
      --eval_output_path "$ckpt/val_predictions.jsonl" \
      | tee "$ckpt/val_log.txt"
  ' ::: "${CKPTS[@]}"
else
  echo "[info] 没有需要评估的 checkpoints（均已存在 val_predictions.jsonl）。"
fi

echo
echo "[info] 所有评估完成，汇总最优 checkpoint..."
