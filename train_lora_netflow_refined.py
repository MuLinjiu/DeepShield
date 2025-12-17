# filename: train_lora_netflow_refined.py
# Usage:
#   Train:
#     python train_lora_netflow_refined.py --train_path data/train.jsonl --val_path data/val.jsonl --out_dir lora-netflow
#   Test (eval on a jsonl):
#     python train_lora_netflow_refined.py --eval_path data/test.jsonl --adapter lora-netflow --mode eval
#   Predict (single flow json file or inline string):
#     python train_lora_netflow_refined.py --predict_path sample_flow.json --adapter lora-netflow --mode predict
#     # 或者
#     python train_lora_netflow_refined.py --predict_inline '{"flow_id":1,...}' --adapter lora-netflow --mode predict

import os, json, math, argparse, random, copy
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from tqdm.auto import tqdm
import inspect

from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# 更激进的矩阵计算设置，提升推理/训练吞吐
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

# -------------------------
# Utils
# -------------------------
BASE_SYSTEM_PROMPT = (
    "You are a network traffic security analyst.\n"
    "Analyze the network flow record and classify it as BENIGN or a specific attack type.\n"
    "Output a JSON with two fields:\n"
    '- \"label\": The security classification. Choose ONLY from this list:\n{label_options_bullet}\n'
    '- \"explanation\": A brief analysis citing specific features from the flow that support your classification. '
    "IMPORTANT: The explanation MUST describe the evidence, such as unusual packet counts, suspicious protocols, "
    "abnormal payload patterns, TCP flag combinations, port numbers, or timing patterns.\n\n"
    "Example output format:\n"
    "{{\"label\": \"Web Attack - XSS\", \"explanation\": \"High packet count (1250 pkts) with HTTP traffic "
    "to port 80, payload contains script tags and suspicious ASCII patterns (ratio 0.85)\"}}\n\n"
    "IMPORTANT: You MUST provide a non-empty explanation describing the key features that led to your classification.\n\n"
    "Analyze the following flow:\n"
)

# Will be populated at runtime from the train set (or fallback list)
LABEL_OPTIONS = [
    "BENIGN", "DDoS", "PortScan", "Bot", "Infiltration",
    "Web Attack - XSS", "Web Attack - Brute Force", "Web Attack - Sql Injection"
]

def set_label_options(options):
    """Set the global label options list used in system prompt."""
    global LABEL_OPTIONS
    if options:
        LABEL_OPTIONS = sorted(set(options))

def label_options_bullet_text():
    """Return label options as bullet list string."""
    if not LABEL_OPTIONS:
        return "  - (no labels found)"
    return "  - " + "\n  - ".join(LABEL_OPTIONS)

def get_system_prompt():
    """Build system prompt with current label options."""
    return BASE_SYSTEM_PROMPT.format(label_options_bullet=label_options_bullet_text())

def mask_ip_fields(flow_record: Dict[str, Any], ip_mask_rate: float = 0.0) -> Dict[str, Any]:
    """按概率掩码源/目的IP，默认不掩码"""
    if ip_mask_rate <= 0:
        return flow_record

    masked = copy.deepcopy(flow_record)

    def maybe_mask(ip):
        if isinstance(ip, str) and random.random() < ip_mask_rate:
            return "REDACTED"
        return ip

    tuple5 = masked.get("tuple5", None)
    if isinstance(tuple5, (list, tuple)) and len(tuple5) >= 2:
        tuple5 = list(tuple5)
        tuple5[0] = maybe_mask(tuple5[0])
        tuple5[1] = maybe_mask(tuple5[1])
        masked["tuple5"] = tuple5

    for key in ["src_ip", "dst_ip"]:
        if key in masked:
            masked[key] = maybe_mask(masked[key])

    return masked


def build_prompt(flow_record: Dict[str, Any], ip_mask_rate: float = 0.0) -> str:
    """构建输入提示词，重点展示关键特征，可选掩码IP"""
    flow_record = preprocess(flow_record, ip_mask_rate=ip_mask_rate)
    # 提取关键信息
    features = flow_record.get("features", {})
    enriched = flow_record.get("enriched", {})
    tuple5 = flow_record.get("tuple5", [])

    # 构建简洁的特征描述
    summary = []

    # 基本连接信息
    if len(tuple5) >= 5:
        src_ip, dst_ip, dst_port, src_port, protocol = tuple5
        summary.append(f"Connection: {src_ip}:{src_port} -> {dst_ip}:{dst_port} (proto={protocol})")

    # 关键统计特征
    if features:
        summary.append(f"Packets: {features.get('packet_count', 0)} total "
                      f"(fwd={features.get('pkt_cnt_fwd', 0)}, bwd={features.get('pkt_cnt_bwd', 0)})")
        summary.append(f"Bytes: {features.get('byte_count', 0)} total "
                      f"(fwd={features.get('byte_cnt_fwd', 0)}, bwd={features.get('byte_cnt_bwd', 0)})")
        summary.append(f"Flow duration: {features.get('flow_dur_ms', 0):.2f}ms")

        # TCP标志
        tcp_info = []
        if features.get('tcp_syn_ratio', 0) > 0:
            tcp_info.append(f"SYN={features['tcp_syn_ratio']:.2f}")
        if features.get('tcp_ack_ratio', 0) > 0:
            tcp_info.append(f"ACK={features['tcp_ack_ratio']:.2f}")
        if features.get('tcp_psh_ratio', 0) > 0:
            tcp_info.append(f"PSH={features['tcp_psh_ratio']:.2f}")
        if features.get('tcp_fin_ratio', 0) > 0:
            tcp_info.append(f"FIN={features['tcp_fin_ratio']:.2f}")
        if features.get('tcp_rst_ratio', 0) > 0:
            tcp_info.append(f"RST={features['tcp_rst_ratio']:.2f}")
        if tcp_info:
            summary.append(f"TCP flags: {', '.join(tcp_info)}")

        # Payload信息
        if features.get('payload_total_len', 0) > 0:
            summary.append(f"Payload: {features['payload_total_len']} bytes, "
                          f"entropy={features.get('payload_entropy', 0):.2f}, "
                          f"ASCII ratio={features.get('payload_ascii_ratio', 0):.2f}")

    # Enriched信息
    if enriched:
        protocols = enriched.get('protocols', [])
        if protocols:
            summary.append(f"Protocols: {', '.join(protocols[:5])}")  # 只显示前5个

        context = enriched.get('context_summary', '')
        if context:
            summary.append(f"Context: {context[:150]}")  # 限制长度

    prompt_text = f"{get_system_prompt()}\n" + "\n".join(summary)
    return prompt_text

def preprocess(flow_record, ip_mask_rate):
    # 先做IP掩码
    flow_record = mask_ip_fields(flow_record, ip_mask_rate=ip_mask_rate)

    # 将绝对时间窗口改成相对时长，并把上下文字符串里的 window 也改为相对描述
    def replace_window_string(text: str) -> str:
        # 例: "TCP window 1499355897.000-1499355899.000s" -> "TCP window 0.000-2.000s"
        import re
        # 匹配 "window 1499355897.000-1499355899.000s" 样式
        pat = re.compile(r"window\s+([0-9]+\.?[0-9]*)\s*-\s*([0-9]+\.?[0-9]*)s", re.IGNORECASE)

        def repl(m):
            s0, s1 = float(m.group(1)), float(m.group(2))
            return f"window 0.000-{(s1 - s0):.3f}s"

        return pat.sub(repl, text)

    def _convert_window(obj):
        if isinstance(obj, dict):
            # 只处理键名为"window"且值是长度为2的数字序列，避免误伤 tcp_window 等字段
            if "window" in obj:
                win = obj.get("window")
                if (
                    isinstance(win, (list, tuple))
                    and len(win) == 2
                    and all(isinstance(x, (int, float)) for x in win)
                ):
                    obj["windows_length"] = win[1] - win[0]
                    obj.pop("window", None)
            # 递归处理子字段，防止嵌套对象里也有 window
            for k, v in list(obj.items()):
                obj[k] = _convert_window(v)
        elif isinstance(obj, list):
            obj = [_convert_window(x) for x in obj]
        elif isinstance(obj, str):
            obj = replace_window_string(obj)
        return obj

    flow_record = _convert_window(flow_record)
    return flow_record

def build_prompt_raw_json(flow_record: Dict[str, Any], ip_mask_rate: float = 0.0) -> str:
    """直接把原始flow_record JSON拼到系统prompt后，用于对比实验，可选掩码IP"""
    masked = preprocess(flow_record, ip_mask_rate=ip_mask_rate)
    # 使用ensure_ascii=False保留原字段格式，便于模型看到真实特征
    raw = json.dumps(masked, ensure_ascii=False)
    return f"{get_system_prompt()}\n{raw}"


def build_completion_train(label: str) -> str:
    """训练时的completion格式：explanation字段开了个头但没写完，鼓励模型继续生成"""
    # 输出到 "explanation": " 就停止（引号未闭合，暗示要继续）
    # 配合 --remove_eos_from_training，让模型学习这是未完成的输出
    parts = ['{"label": "', label, '", "explanation": "']
    return "".join(parts)


def build_completion_infer() -> str:
    """推理时的completion起始格式，让模型自动补全"""
    return '{"label": "'


def add_explanation_fallback(flow_record: Dict[str, Any], js: Dict[str, Any]) -> Dict[str, Any]:
    """给没有explanation的预测结果补充一个基于特征的解释，避免空字段"""
    if not isinstance(js, dict) or js.get("explanation"):
        return js

    features = flow_record.get("features", {})
    enriched = flow_record.get("enriched", {})
    tuple5 = flow_record.get("tuple5", [])
    label = js.get("label", "")

    parts = []

    if "XSS" in label or "SQL" in label or "Web Attack" in label:
        if len(tuple5) >= 5 and tuple5[2] in [80, 443, 8080]:
            parts.append(f"HTTP traffic on port {tuple5[2]}")
        pkt_count = features.get('packet_count', 0)
        if pkt_count > 100:
            parts.append(f"high packet count ({pkt_count})")
        payload_len = features.get('payload_total_len', 0)
        if payload_len > 1000:
            parts.append(f"large payload ({payload_len} bytes)")
        ascii_ratio = features.get('payload_ascii_ratio', 0)
        if ascii_ratio > 0.7:
            parts.append(f"text-heavy content (ASCII {ascii_ratio:.2f})")

    elif "Brute Force" in label or "Bot" in label:
        pkt_count = features.get('packet_count', 0)
        parts.append(f"repeated attempts ({pkt_count} packets)")
        if len(tuple5) >= 5:
            parts.append(f"targeting port {tuple5[2]}")
        tcp_syn = features.get('tcp_syn_ratio', 0)
        if tcp_syn > 0.3:
            parts.append(f"connection-heavy (SYN {tcp_syn:.2f})")

    elif "DDoS" in label or "DoS" in label:
        pkt_count = features.get('packet_count', 0)
        byte_count = features.get('byte_count', 0)
        parts.append(f"volumetric attack ({pkt_count} pkts, {byte_count//1000}KB)")
        flow_dur = features.get('flow_dur_ms', 0)
        if flow_dur > 0:
            rate = pkt_count / (flow_dur / 1000) if flow_dur > 0 else 0
            if rate > 100:
                parts.append(f"high rate ({rate:.0f} pkt/s)")

    elif "PortScan" in label:
        pkt_count = features.get('packet_count', 0)
        parts.append(f"probing behavior ({pkt_count} packets)")
        tcp_syn = features.get('tcp_syn_ratio', 0)
        if tcp_syn > 0.5:
            parts.append(f"SYN scanning (ratio {tcp_syn:.2f})")

    elif "Infiltration" in label:
        pkt_count = features.get('packet_count', 0)
        byte_count = features.get('byte_count', 0)
        parts.append(f"suspicious activity ({pkt_count} pkts, {byte_count} bytes)")
        entropy = features.get('payload_entropy', 0)
        if entropy > 5:
            parts.append(f"encrypted/encoded data (entropy {entropy:.1f})")

    elif "BENIGN" in label:
        pkt_count = features.get('packet_count', 0)
        byte_count = features.get('byte_count', 0)
        parts.append(f"normal pattern ({pkt_count} pkts, {byte_count} bytes)")
        protocols = enriched.get('protocols', [])
        if protocols:
            parts.append(f"using {protocols[0]}")

    if not parts:
        pkt_count = features.get('packet_count', 0)
        byte_count = features.get('byte_count', 0)
        parts.append(f"Flow with {pkt_count} packets and {byte_count} bytes")

    js["explanation"] = ", ".join(parts[:3])
    return js

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows

def collect_labels_from_jsonl(path: str) -> List[str]:
    """扫描jsonl文件，收集label集合"""
    labels = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                label = obj.get("label")
                if isinstance(label, str) and label:
                    labels.add(label)
    except FileNotFoundError:
        pass
    return sorted(labels)

def make_hf_dataset(
    path: str,
    shuffle: bool = True,
    seed: int = 42,
    ip_mask_rate: float = 0.0,
    prompt_fn=build_prompt_raw_json,
) -> Dataset:
    """
    使用 Python 逐行解析并直接生成所需字段，避免 PyArrow 在推断原始字段类型时因混合类型报错。
    """

    def _gen():
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"Warning: JSON parse failed at line {idx}: {e}")
                    continue

                label = obj.get("label")
                if label is None:
                    print(f"Warning: Missing label in record {obj.get('flow_id', 'unknown')}")
                    continue

                flow_record = dict(obj)
                flow_record.pop("label", None)
                flow_record_masked = preprocess(flow_record, ip_mask_rate=ip_mask_rate)

                yield {
                    "prompt": prompt_fn(flow_record_masked),
                    "completion": build_completion_train(label),
                    "label": label,
                    "flow_record": json.dumps(flow_record_masked, ensure_ascii=False),
                }

    ds = Dataset.from_generator(_gen)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    return ds

# -------------------------
# Tokenize & mask (only supervise "label" value tokens)
# -------------------------
def tokenize_with_label_mask(ex, tok, max_len=4096, remove_eos=False):
    """
    使用chat模板格式进行训练，只对assistant的response进行监督。

    Args:
        ex: 训练样本
        tok: tokenizer
        max_len: 最大长度
        remove_eos: 是否去掉训练序列末尾的EOS token (鼓励模型生成explanation)
    """
    prompt_text = ex["prompt"].strip()
    completion_raw = ex["completion"].strip()

    # 构建chat格式的消息
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": completion_raw}
    ]

    # 使用tokenizer的apply_chat_template
    if hasattr(tok, 'apply_chat_template'):
        # 生成完整的chat格式文本
        full_text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # 生成只有user部分的文本(带assistant起始标记)
        user_text = tok.apply_chat_template(
            [messages[0]],
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize完整文本
        full_enc = tok(
            full_text,
            truncation=True,
            max_length=max_len,
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        # 如果启用remove_eos，在chat模板模式下从结尾向前搜索并删除最后一个EOS (<|im_end|>)
        # 以及EOS之后仅包含的空白token，避免模型学到“输出模板后立刻用<|im_end|>结束”。
        if remove_eos and tok.eos_token_id is not None and len(input_ids) > 0:
            eos_ids = tok.eos_token_id
            if isinstance(eos_ids, (list, tuple)):
                eos_id_set = set(eos_ids)
            else:
                eos_id_set = {eos_ids}

            # 先跳过结尾的纯空白token
            last_non_ws = len(input_ids) - 1
            while last_non_ws >= 0:
                piece = tok.decode([input_ids[last_non_ws]])
                if piece.strip() == "":
                    last_non_ws -= 1
                else:
                    break

            # 如果最后一个非空白token是EOS，则截断到它之前
            if last_non_ws >= 0 and input_ids[last_non_ws] in eos_id_set:
                new_len = last_non_ws  # 不包括EOS本身
                if new_len > 0:
                    input_ids = input_ids[:new_len]
                    attention_mask = attention_mask[:new_len]

        # Tokenize user部分以确定assistant回复的起始位置
        user_enc = tok(
            user_text,
            add_special_tokens=True,
        )
        user_len = len(user_enc["input_ids"])

        # 创建labels: user部分mask为-100, assistant部分参与训练
        labels = [-100] * len(input_ids)
        for idx in range(user_len, len(input_ids)):
            labels[idx] = input_ids[idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    else:
        # Fallback: 如果没有chat template，使用原来的方式
        completion_text = "\n\n" + completion_raw
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        comp_enc = tok(completion_text, add_special_tokens=False, return_offsets_mapping=True)
        comp_ids = comp_enc["input_ids"]
        comp_offsets = comp_enc["offset_mapping"]

        special_extra = getattr(tok, "_special_extra", None)
        if special_extra is None:
            dummy = tok.build_inputs_with_special_tokens([0])
            special_extra = len(dummy) - 1
            tok._special_extra = special_extra

        max_prompt_tokens = max_len - len(comp_ids) - special_extra
        if max_prompt_tokens < 0:
            max_prompt_tokens = 0
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[:max_prompt_tokens]

        combined = prompt_ids + comp_ids
        input_ids = tok.build_inputs_with_special_tokens(combined)

        # 如果启用remove_eos，去掉末尾的EOS token，让模型不学习"在completion后立即结束"
        if remove_eos and tok.eos_token_id is not None and len(input_ids) > 0:
            if input_ids[-1] == tok.eos_token_id:
                input_ids = input_ids[:-1]

        attention_mask = [1] * len(input_ids)
        special_mask = tok.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        labels = [-100] * len(input_ids)

        label_start = label_end = -1
        label_key = '"label": "'
        idx = completion_text.find(label_key)
        if idx != -1:
            label_start = idx + len(label_key)
            label_end = completion_text.find('"', label_start)
        if label_end == -1:
            label_start = label_end = -1

        real_ptr = 0
        for seq_idx, is_special in enumerate(special_mask):
            if is_special:
                continue
            if real_ptr < len(prompt_ids):
                pass
            else:
                comp_idx = real_ptr - len(prompt_ids)
                if 0 <= comp_idx < len(comp_offsets):
                    a, b = comp_offsets[comp_idx]
                    if label_start != -1 and not (b <= label_start or a >= label_end):
                        labels[seq_idx] = input_ids[seq_idx]
            real_ptr += 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# -------------------------
# Data collator (pad)
# -------------------------
@dataclass
class SimpleDataCollator:
    tokenizer: AutoTokenizer
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 提取需要的字段
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        # 使用tokenizer进行padding
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True, return_tensors="pt"
        )
        
        # 手动pad labels
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for l in labels:
            pad_len = max_len - len(l)
            padded = l + [-100] * pad_len
            padded_labels.append(padded)
        
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

# -------------------------
# Metrics (accuracy / macro F1 on parsed label)
# -------------------------
from collections import Counter

def simple_accuracy(preds: List[str], refs: List[str]) -> float:
    eq = sum(p == r for p, r in zip(preds, refs))
    return eq / max(1, len(refs))

def macro_f1(preds: List[str], refs: List[str]) -> float:
    classes = sorted(set(refs) | set(preds))
    f1s = []
    for c in classes:
        tp = sum((p == c and r == c) for p, r in zip(preds, refs))
        fp = sum((p == c and r != c) for p, r in zip(preds, refs))
        fn = sum((p != c and r == c) for p, r in zip(preds, refs))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / max(1, len(f1s))

def compute_metrics(preds: List[str], refs: List[str]) -> dict:
    """计算详细的分类指标"""
    classes = sorted(set(refs) | set(preds))

    # 整体指标
    acc = simple_accuracy(preds, refs)
    macro_f1_score = macro_f1(preds, refs)

    # 每个类别的指标
    per_class_metrics = {}
    for c in classes:
        tp = sum((p == c and r == c) for p, r in zip(preds, refs))
        fp = sum((p == c and r != c) for p, r in zip(preds, refs))
        fn = sum((p != c and r == c) for p, r in zip(preds, refs))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

        support = sum(1 for r in refs if r == c)

        per_class_metrics[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support
        }

    # 计算macro和weighted平均
    total_support = len(refs)
    macro_precision = sum(m["precision"] for m in per_class_metrics.values()) / len(classes)
    macro_recall = sum(m["recall"] for m in per_class_metrics.values()) / len(classes)

    weighted_precision = sum(m["precision"] * m["support"] for m in per_class_metrics.values()) / total_support
    weighted_recall = sum(m["recall"] * m["support"] for m in per_class_metrics.values()) / total_support
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class_metrics.values()) / total_support

    return {
        "accuracy": acc,
        "macro_f1": macro_f1_score,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "total_samples": total_support,
        "per_class": per_class_metrics
    }

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """从生成的文本中提取JSON，支持多种格式"""
    import re

    # 尝试提取最后一个完整的 {...}
    if "{" in text and "}" in text:
        # 找到所有可能的JSON起始位置
        json_starts = []
        for i, char in enumerate(text):
            if char == '{':
                json_starts.append(i)

        # 从后往前尝试解析JSON
        for start in reversed(json_starts):
            for end in range(len(text), start, -1):
                if text[end-1] == '}':
                    try:
                        candidate = text[start:end]
                        return json.loads(candidate)
                    except:
                        continue

        # 如果找不到完整JSON，尝试提取部分信息
        # 查找 "label": "..." 模式 (允许多个词和连字符)
        label_match = re.search(r'"label"\s*:\s*"([^"]+)"', text)
        if label_match:
            label_value = label_match.group(1)
            # 提取explanation (如果有的话)
            exp_match = re.search(r'"explanation"\s*:\s*"([^"]*(?:"[^"]*)*)', text)
            explanation = exp_match.group(1) if exp_match else ""
            return {"label": label_value, "explanation": explanation}

    # 如果没有JSON，尝试从自然语言中提取标签
    # 常见的标签类型
    known_labels = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration",
                    "Web Attack", "Brute Force", "DoS", "Heartbleed",
                    "FTP-Patator", "SSH-Patator", "DoS slowloris",
                    "DoS Slowhttptest", "DoS Hulk", "DoS GoldenEye"]

    # 尝试匹配标签词
    text_upper = text.upper()
    for label in known_labels:
        if label.upper() in text_upper:
            # 提取包含该标签的句子作为explanation
            sentences = re.split(r'[.!?]', text)
            explanation = ""
            for sent in sentences:
                if label.upper() in sent.upper():
                    explanation = sent.strip()
                    break
            return {"label": label, "explanation": explanation or text.strip()[:200]}

    # 最后的fallback：查找 "label would be" 或 "classified as" 等模式
    label_patterns = [
        r'label (?:would be|is|:)\s*["\']?(\w+)["\']?',
        r'classified as\s*["\']?(\w+)["\']?',
        r'classification (?:is|:)\s*["\']?(\w+)["\']?',
    ]
    for pattern in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {"label": match.group(1).upper(), "explanation": text.strip()[:200]}

    return None

def generate_one(
    flow_record: Dict[str, Any],
    tok,
    model,
    max_new_tokens=2048,
    temperature=0.0,
    max_context: Optional[int] = None,
    disable_fallback: bool = False,
    ip_mask_rate: float = 0.0,
) -> Dict[str, Any]:
    """生成单个流量记录的分类结果 - 使用chat模板

    Args:
        disable_fallback: 如果True，不使用hard-coded规则生成explanation fallback
    """
    prompt = build_prompt_raw_json(flow_record, ip_mask_rate=ip_mask_rate)

    # 使用chat模板格式(匹配训练时的格式)
    if hasattr(tok, 'apply_chat_template'):
        messages = [
            {"role": "user", "content": prompt}
        ]
        full_prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 添加assistant起始标记
        )
    else:
        # Fallback: 简单拼接
        full_prompt = prompt + "\n\n"

    # Prompt 太长会挤占生成空间，这里确保至少保留一段上下文给解码器输出
    ctx_window = max_context or getattr(tok, "model_max_length", 4096)
    if ctx_window is None or ctx_window <= 0 or ctx_window > 8192:
        ctx_window = 4096
    # 预留出 max_new_tokens 以及少量缓冲，避免输入占满整个窗口
    max_prompt_tokens = ctx_window - max_new_tokens - 32
    if max_prompt_tokens < 32:
        max_prompt_tokens = ctx_window // 2

    # print('-'*101)
    # print(full_prompt)
    # print('-'*101)
    inputs = tok(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens
    ).to(model.device)

    # 生成时使用更长的max_new_tokens以便生成完整的explanation
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": (temperature > 0),
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "repetition_penalty": 1.2,  # 防止重复生成相同词语
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # 解码生成的文本
    # 只解码新生成的tokens (从input_ids长度开始)
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = out[0][input_len:]  # 只取新生成的部分
    response = tok.decode(generated_tokens, skip_special_tokens=True).strip()
    # print('='*100)
    # print(response)
    # print('='*100)

    # 提取JSON结果
    js = extract_json_from_text(response)

    if not disable_fallback:
        js = add_explanation_fallback(flow_record, js)

    return js if isinstance(js, dict) else {"label": "", "explanation": ""}

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--predict_inline", type=str, default=None)
    parser.add_argument("--eval_output_path", type=str, default=None,
                        help="Save per-sample eval results to this JSONL file; defaults to <adapter|out_dir>/eval_predictions.jsonl")
    parser.add_argument("--label_source_path", type=str, default=None,
                        help="JSONL file to enumerate label options; defaults to train_path if provided, else eval_path")

    parser.add_argument("--base_model", type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)  # 用于 eval / predict

    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Total training steps; >0 overrides epochs (HF default -1).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from (e.g., lora-qwen3-12h/checkpoint-900).")
    parser.add_argument("--per_device_bs", type=int, default=1)  # 降低batch size
    parser.add_argument("--grad_accum", type=int, default=16)    # 增加梯度累积
    parser.add_argument("--max_len", type=int, default=4096)     # 增加最大长度
    parser.add_argument("--ip_mask_rate", type=float, default=0.2,
                        help="Probability to mask src/dst IP in prompts (0.2=20%)")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="DataLoader worker数量，提高数据加载吞吐")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="评估间隔step数")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="保存checkpoint间隔step数")
    parser.add_argument("--logging_steps", type=int, default=20,
                        help="日志打印间隔step数")

    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Eval阶段并行生成的batch size，显著提升吞吐")
    parser.add_argument("--eval_max_new_tokens", type=int, default=2048,
                        help="Eval生成的新token上限，默认与之前脚本一致，可自行调大")
    parser.add_argument("--eval_temperature", type=float, default=0.3,
                        help="Eval阶段temperature，0为贪心解码")

    parser.add_argument("--mode", type=str, default="train", choices=["train","eval","predict"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--remove_eos_from_training", action="store_true",
                        help="Remove EOS token from training sequences to encourage explanation generation", default=True)
    parser.add_argument("--disable_explanation_fallback", action="store_true",
                        help="Disable hard-coded explanation generation fallback (test if model generates explanations)")
    args = parser.parse_args()

    set_seed(args.seed)

    # 设置label列表供系统prompt使用
    label_source = args.label_source_path or args.train_path or args.eval_path
    if label_source:
        labels_found = collect_labels_from_jsonl(label_source)
        if labels_found:
            set_label_options(labels_found)
            print(f"[*] Loaded {len(labels_found)} labels from {label_source}")
        else:
            print(f"[!] No labels found in {label_source}, using default label list.")

    # Tokenizer & Model
    print("[*] Loading tokenizer...")
    # 从环境变量读取HF token，如果没有则使用None（公开模型不需要token）
    hf_token = os.getenv("HF_TOKEN", None)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, token=hf_token)
    tok.padding_side = "left" if args.mode in ["eval", "predict"] else "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[*] Loading base model...")
    # 多卡训练不允许 device_map="auto"，需按 local_rank 手动映射
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    # Prefer FlashAttention2 for faster attention if available
    _attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_4bit=args.load_in_4bit,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            attn_implementation=_attn_impl,
            token=hf_token
        )
        print(f"[*] Using attention implementation: {_attn_impl}")
    except Exception as e:
        print(f"[!] FlashAttention2 unavailable ({e}), falling back to default attention.")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_4bit=args.load_in_4bit,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            token=hf_token
        )

    if args.mode == "train":
        assert args.train_path and args.val_path, "Need --train_path and --val_path for training."
        if not args.out_dir:
            args.out_dir = "lora-netflow"  # 默认输出目录
            print(f"[*] No --out_dir specified, using default: {args.out_dir}")

        print("[*] Loading datasets...")
        # Build datasets
        train_ds_raw = make_hf_dataset(args.train_path, ip_mask_rate=args.ip_mask_rate)
        val_ds_raw   = make_hf_dataset(args.val_path,   ip_mask_rate=args.ip_mask_rate)
        
        print(f"[*] Train samples: {len(train_ds_raw)}, Val samples: {len(val_ds_raw)}")
        print(f"[*] Remove EOS from training: {args.remove_eos_from_training}")

        def _map_fn(ex):
            return tokenize_with_label_mask(ex, tok, max_len=args.max_len,
                                           remove_eos=args.remove_eos_from_training)
        
        print("[*] Tokenizing datasets...")
        train_ds = train_ds_raw.map(_map_fn, remove_columns=train_ds_raw.column_names, desc="Tokenizing train")
        val_ds   = val_ds_raw.map(_map_fn,   remove_columns=val_ds_raw.column_names,   desc="Tokenizing val")

        # LoRA配置
        print("[*] Setting up LoRA...")
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.r, 
            lora_alpha=args.alpha, 
            lora_dropout=args.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_cfg)
        
        # 打印可训练参数
        model.print_trainable_parameters()

        # 训练参数
        # Build TrainingArguments kwargs then filter to supported args to maintain
        # compatibility with older/newer transformers versions.

        # 计算warmup步数: 用固定值避免warmup_ratio计算错误
        # 20K samples / (bs=4 * grad_accum=8) = 625 steps/epoch
        # warmup 100步 ≈ 0.16 epoch,足够稳定训练
        warmup_steps = 100

        tr_kwargs = dict(
            output_dir=args.out_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.per_device_bs,
            per_device_eval_batch_size=args.per_device_bs,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,

            # LR scheduler: warmup + cosine decay
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,  # 使用固定步数而不是ratio

            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            bf16=args.bf16,
            fp16=not args.bf16,
            report_to="none",
            save_total_limit=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=args.dataloader_num_workers,
        )

        sig = inspect.signature(TrainingArguments)
        supported = set(sig.parameters.keys())
        filtered = {k: v for k, v in tr_kwargs.items() if k in supported}

        # If evaluation_strategy not supported by this transformers version,
        # disable load_best_model_at_end and drop metric_for_best_model to avoid validation errors.
        if "evaluation_strategy" not in supported:
            filtered.pop("evaluation_strategy", None)
            filtered["load_best_model_at_end"] = False
            filtered.pop("metric_for_best_model", None)

        try:
            args_tr = TrainingArguments(**filtered)
        except ValueError as e:
            # Handle validation errors like mismatched save/eval strategies by disabling
            # load_best_model_at_end and retrying.
            msg = str(e)
            if "load_best_model_at_end" in filtered and filtered.get("load_best_model_at_end"):
                filtered["load_best_model_at_end"] = False
                filtered.pop("metric_for_best_model", None)
                args_tr = TrainingArguments(**filtered)
            else:
                raise

        collator = SimpleDataCollator(tokenizer=tok)
        trainer = Trainer(
            model=model,
            args=args_tr,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator
        )

        # Decide resume checkpoint
        resume_arg = args.resume_from_checkpoint
        if resume_arg is None:
            last_ckpt = get_last_checkpoint(args.out_dir) if args.out_dir and os.path.isdir(args.out_dir) else None
            if last_ckpt:
                resume_arg = last_ckpt
                print(f"[*] Auto-resuming from last checkpoint: {last_ckpt}")

        print("[*] Start training...")
        trainer.train(resume_from_checkpoint=resume_arg)
        print("[*] Saving adapter to:", args.out_dir)
        model.save_pretrained(args.out_dir)
        tok.save_pretrained(args.out_dir)
        print("[*] Training completed.")

    elif args.mode == "eval":
        assert args.eval_path, "Need --eval_path"

        # Distributed eval setup
        is_distributed = world_size > 1 and dist.is_available()
        rank = 0
        if is_distributed and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        elif is_distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = int(os.environ.get("RANK", "0"))
        is_main = (rank == 0)

        # 如果提供了adapter，加载LoRA；否则直接用base model
        if args.adapter or args.out_dir:
            adapter_dir = args.adapter or args.out_dir
            if is_main:
                print(f"[*] Loading LoRA adapter from {adapter_dir}...")
            # 直接使用PeftModel.from_pretrained加载训练好的adapter
            model = PeftModel.from_pretrained(
                model,
                adapter_dir,
                is_trainable=False
            )

            # 调试: 打印adapter信息
            if is_main:
                print(f"[*] Model type: {type(model)}")
                print(f"[*] Active adapters: {model.active_adapters if hasattr(model, 'active_adapters') else 'N/A'}")
        else:
            if is_main:
                print("[*] Evaluating base model (no LoRA adapter)")

        # 确保模型在eval模式
        model.eval()

        if is_main:
            print("[*] Evaluating...")
        eval_rows_all = read_jsonl(args.eval_path)

        # rank-based sharding for multi-GPU eval
        preds, refs = [], []
        eval_records = []  # store per-sample outputs for optional JSONL dump

        # 预先构建所有prompt，避免循环里重复tokenize；保留原始prompt便于落盘
        prompts, prompts_raw, flow_records, global_indices = [], [], [], []
        for g_idx, row in enumerate(eval_rows_all):
            if is_distributed and (g_idx % world_size != rank):
                continue

            refs.append(row["label"])
            flow_record = dict(row)
            flow_record.pop("label", None)
            flow_records.append(flow_record)
            global_indices.append(g_idx)

            prompt_raw = build_prompt_raw_json(flow_record, ip_mask_rate=args.ip_mask_rate)
            prompt_for_model = prompt_raw
            if hasattr(tok, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt_raw}]
                prompt_for_model = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_for_model = prompt_raw + "\n\n"

            prompts_raw.append(prompt_raw)
            prompts.append(prompt_for_model)

        max_new_tokens = args.eval_max_new_tokens if args.eval_max_new_tokens > 0 else 2048

        max_prompt_tokens = args.max_len - max_new_tokens - 32
        if max_prompt_tokens < 64:
            max_prompt_tokens = max(32, args.max_len // 2)
        local_total = len(prompts)

        def batched(iterable, n):
            for i in range(0, len(iterable), n):
                yield i, iterable[i:i+n]

        progress_desc = "Eval" if not is_distributed else f"Eval rank{rank}"
        progress_bar = tqdm(total=local_total, desc=progress_desc, ncols=100, disable=False)

        for start_idx, batch_prompts in batched(prompts, args.eval_batch_size):
            batch_flows = flow_records[start_idx:start_idx+len(batch_prompts)]
            batch_refs = refs[start_idx:start_idx+len(batch_prompts)]
            batch_prompts_raw = prompts_raw[start_idx:start_idx+len(batch_prompts)]
            batch_indices = global_indices[start_idx:start_idx+len(batch_prompts)]

            inputs = tok(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens
            ).to(model.device)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=(args.eval_temperature > 0),
                temperature=args.eval_temperature if args.eval_temperature > 0 else None,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                repetition_penalty=1.2,
            )
            if gen_kwargs["temperature"] is None:
                gen_kwargs.pop("temperature")

            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)

            gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
            responses = tok.batch_decode(gen_tokens, skip_special_tokens=True)

            for resp, flow_record, ref, prompt_raw, g_idx in zip(
                responses, batch_flows, batch_refs, batch_prompts_raw, batch_indices
            ):
                js = extract_json_from_text(resp)

                if not args.disable_explanation_fallback:
                    js = add_explanation_fallback(flow_record, js if isinstance(js, dict) else {})

                pred = js.get("label", "") if isinstance(js, dict) else ""
                preds.append(pred)
                eval_records.append({
                    "index": g_idx,
                    "prompt": prompt_raw,
                    "reference_label": ref,
                    "predicted_label": pred,
                    "model_explanation": (js.get("explanation", "") if isinstance(js, dict) else "")
                })

            if is_main and start_idx == 0:
                for j, (resp, ref, flow_record) in enumerate(zip(responses[:2], batch_refs[:2], batch_flows[:2])):
                    js = extract_json_from_text(resp) or {}
                    print(f"Sample {j+1}:")
                    print(f"  Reference: {ref}")
                    print(f"  Predicted: {js.get('label', '')}")
                    print(f"  Explanation: {(js.get('explanation') or '')[:160]}")
                    print()

            progress_bar.update(len(batch_prompts))

        progress_bar.close()

        # Gather preds/refs from all ranks
        if is_distributed:
            preds_list = [None for _ in range(world_size)]
            refs_list = [None for _ in range(world_size)]
            dist.all_gather_object(preds_list, preds)
            dist.all_gather_object(refs_list, refs)
            records_list = [None for _ in range(world_size)]
            dist.all_gather_object(records_list, eval_records)
            if is_main:
                preds = [p for sub in preds_list for p in sub]
                refs = [r for sub in refs_list for r in sub]
                eval_records = [r for sub in records_list for r in sub]
            dist.barrier()

        # Only main rank computes / prints metrics
        if not is_main and is_distributed:
            dist.destroy_process_group()
            return

        # 计算详细指标
        metrics = compute_metrics(preds, refs)

        # 简单debug: 检查预测与真实标签的一致性
        try:
            from collections import Counter
            same_count = sum(p == r for p, r in zip(preds, refs))
            total = len(refs)
            print("\n" + "="*60)
            print("[DEBUG] Label matching stats")
            print("="*60)
            print(f"Same predictions: {same_count}/{total}")
            print(f"Unique preds: {len(set(preds))}, Unique refs: {len(set(refs))}")
            print("[DEBUG] Pred label distribution:", Counter(preds))
            print("[DEBUG] Ref  label distribution:", Counter(refs))
        except Exception as e:
            print(f"[DEBUG] Failed to compute label stats: {e}")

        # 打印总体指标
        print("\n" + "="*60)
        print("Overall Metrics:")
        print("="*60)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Macro Precision:   {metrics['macro_precision']:.4f}")
        print(f"Macro Recall:      {metrics['macro_recall']:.4f}")
        print(f"Macro F1:          {metrics['macro_f1']:.4f}")
        print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
        print(f"Weighted Recall:    {metrics['weighted_recall']:.4f}")
        print(f"Weighted F1:        {metrics['weighted_f1']:.4f}")
        print(f"Total Samples:      {metrics['total_samples']}")

        # 打印每个类别的指标
        print("\n" + "="*60)
        print("Per-Class Metrics:")
        print("="*60)
        print(f"{'Class':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-"*80)
        for class_name in sorted(metrics['per_class'].keys()):
            m = metrics['per_class'][class_name]
            print(f"{class_name:<35} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10d}")

        # 保存JSON格式结果
        print("\n" + "="*60)
        print("JSON Format:")
        print("="*60)
        print(json.dumps(metrics, indent=2))

        # 保存逐条评估结果
        save_path = args.eval_output_path
        if not save_path:
            base_dir = args.out_dir or args.adapter or "."
            os.makedirs(base_dir, exist_ok=True)
            save_path = os.path.join(base_dir, "eval_predictions.jsonl")

        eval_records_sorted = sorted(eval_records, key=lambda x: x.get("index", 0))
        with open(save_path, "w", encoding="utf-8") as f:
            for rec in eval_records_sorted:
                line_obj = {
                    "prompt": rec["prompt"],
                    "reference_label": rec["reference_label"],
                    "predicted_label": rec["predicted_label"],
                    "model_explanation": rec["model_explanation"],
                }
                f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
        print(f"[*] Saved per-sample eval results to {save_path}")

        if is_distributed:
            dist.destroy_process_group()

    else:  # predict
        assert (args.predict_path or args.predict_inline) and (args.adapter or args.out_dir), \
            "Need --adapter and either --predict_path or --predict_inline"
        adapter_dir = args.adapter or args.out_dir

        print(f"[*] Loading LoRA adapter from {adapter_dir}...")
        # 直接使用PeftModel.from_pretrained加载训练好的adapter
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
            is_trainable=False
        )

        if args.predict_path:
            with open(args.predict_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            obj = json.loads(args.predict_inline)

        print("[*] Generating prediction...")
        out = generate_one(
            obj, tok, model, temperature=0.0, max_context=args.max_len,
            disable_fallback=args.disable_explanation_fallback,
            ip_mask_rate=args.ip_mask_rate
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        
if __name__ == "__main__":
    main()
