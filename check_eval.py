import glob
import os
import re

root = os.environ.get("CKPT_ROOT", "lora-qwen3-12h")
baseline_dir = os.environ.get("BASELINE_DIR", os.path.join(root, "baseline"))
pattern = os.path.join(root, "checkpoint-*", "val_log.txt")
baseline_log = os.path.join(baseline_dir, "val_log.txt")

METRIC_NAME = os.environ.get("METRIC_NAME", "macro_f1")


def extract_metric(log_path: str, metric_key: str):
    """从日志中提取指定指标，先查 JSON，后备查文本行。"""
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    patterns = [
        rf'"{re.escape(metric_key)}"\s*:\s*([-+]?\d*\.?\d+)',
        rf'{re.escape(metric_key.replace("_", " "))}\s*[:=]\s*([-+]?\d*\.?\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


results = []  # (score, name, log_path)

# checkpoints
for log_path in sorted(glob.glob(pattern)):
    score = extract_metric(log_path, METRIC_NAME)
    if score is not None:
        ckpt_name = os.path.basename(os.path.dirname(log_path))
        results.append((score, ckpt_name, log_path))

# baseline
if os.path.exists(baseline_log):
    score = extract_metric(baseline_log, METRIC_NAME)
    if score is not None:
        results.append((score, "baseline", baseline_log))
    else:
        print(f"[warn] baseline 日志未找到 {METRIC_NAME}: {baseline_log}")
else:
    print(f"[info] 未找到 baseline 日志，跳过: {baseline_log}")

if results:
    results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name, best_path = results[0]
    best_dir = os.path.dirname(best_path)
    print(f"[best] {METRIC_NAME}={best_score:.4f} at {best_dir} ({best_name})")

    print(f"\n[ranking] {METRIC_NAME} sorted (desc):")
    for rank, (score, name, path) in enumerate(results, start=1):
        print(f"  {rank:2d}. {name:<20} {METRIC_NAME}={score:.4f}")
else:
    print(f"[warn] 没在日志中找到 {METRIC_NAME}，请检查日志输出。")
