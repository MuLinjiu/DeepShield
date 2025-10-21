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

import os, json, math, argparse, random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, set_seed
)

from peft import LoraConfig, get_peft_model, TaskType

# -------------------------
# Utils
# -------------------------
SYSTEM_PROMPT = (
    "You are a network flow security classifier.\n"
    'Given an enriched network flow record, output a JSON with fields "label" and "explanation".\n'
    'The "label" should be the security classification (e.g., BENIGN, DDoS, PortScan, etc.).\n'
    'The "explanation" should provide 1-3 sentences explaining the classification reasoning.\n'
    "Respond with only one JSON object.\n"
)

def build_prompt(flow_record: Dict[str, Any]) -> str:
    """构建输入提示词，使用完整的enriched flow record"""
    # 移除label字段，只保留输入特征
    flow_input = {k: v for k, v in flow_record.items() if k != "label"}
    return f"{SYSTEM_PROMPT}Flow Record:\n{json.dumps(flow_input, ensure_ascii=False, indent=2)}"

def build_completion_train(label: str) -> str:
    """训练时的completion格式：explanation为空字符串"""
    return json.dumps({"label": label, "explanation": ""}, ensure_ascii=False)

def build_completion_infer() -> str:
    """推理时的completion起始格式，让模型自动补全"""
    return '{"label": "'

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

def make_hf_dataset(path: str) -> Dataset:
    """根据新的schema格式处理数据"""
    data = read_jsonl(path)
    rows = []
    for d in data:
        # 现在数据格式直接包含所有字段，label是其中一个字段
        if "label" not in d:
            print(f"Warning: Missing label in record {d.get('flow_id', 'unknown')}")
            continue
            
        label = d["label"]
        rows.append({
            "prompt": build_prompt(d),
            "completion": build_completion_train(label),
            "label": label,
            "flow_record": d
        })
    return Dataset.from_list(rows)

# -------------------------
# Tokenize & mask (only supervise "label" value tokens)
# -------------------------
def tokenize_with_label_mask(ex, tok, max_len=4096):
    """
    只对label值进行监督学习，explanation部分不参与损失计算
    """
    full = ex["prompt"].strip() + "\n\n" + ex["completion"].strip()
    enc = tok(full, truncation=True, max_length=max_len)
    labels = enc["input_ids"][:]

    # 定位 completion 在 full 中的位置
    comp = ex["completion"].strip()
    start = full.rfind(comp)
    
    if start == -1:
        # 如果找不到completion，全部mask掉
        labels = [-100] * len(labels)
        enc["labels"] = labels
        return enc

    # 在 completion 内定位 label 的值（含引号）
    # {"label": "<VALUE>", "explanation": ""}
    try:
        label_key = '"label": "'
        i0 = comp.index(label_key) + len(label_key)
        i1 = comp.index('"', i0)  # 结束引号
        lab_abs_start = start + i0
        lab_abs_end = start + i1
    except ValueError:
        # 如果格式不匹配，全部mask掉
        labels = [-100] * len(labels)
        enc["labels"] = labels
        return enc

    # 对非 label 段全部 mask 掉（-100）
    # 重新编码获取 offset_mapping
    try:
        enc_off = tok(
            full, add_special_tokens=False, return_offsets_mapping=True,
            truncation=True, max_length=max_len
        )
        offsets = enc_off["offset_mapping"]
    except:
        # 如果offset_mapping不支持，使用简单策略
        labels = [-100] * len(labels)
        enc["labels"] = labels
        return enc

    # 对齐到最短长度
    n = min(len(labels), len(offsets))
    for t_idx in range(n):
        a, b = offsets[t_idx]
        overlap = not (b <= lab_abs_start or a >= lab_abs_end)
        if not overlap:
            labels[t_idx] = -100
    
    # 如果 labels 比 offsets 长，超出部分也不需要监督
    for t_idx in range(n, len(labels)):
        labels[t_idx] = -100

    enc["labels"] = labels
    return enc

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

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """从生成的文本中提取JSON，支持多种格式"""
    # 尝试提取最后一个完整的 {...}
    if "{" not in text or "}" not in text:
        return None
    
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
    # 查找 "label": "..." 模式
    import re
    label_match = re.search(r'"label"\s*:\s*"([^"]*)"', text)
    if label_match:
        return {"label": label_match.group(1), "explanation": ""}
    
    return None

def generate_one(flow_record: Dict[str, Any], tok, model, max_new_tokens=256, temperature=0.0) -> Dict[str, Any]:
    """生成单个流量记录的分类结果"""
    prompt = build_prompt(flow_record)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    
    # 生成时使用更长的max_new_tokens以便生成完整的explanation
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    
    # 解码生成的文本
    generated_text = tok.decode(out[0], skip_special_tokens=True)
    
    # 提取prompt之后的部分
    prompt_len = len(prompt)
    if len(generated_text) > prompt_len:
        response = generated_text[prompt_len:].strip()
    else:
        response = generated_text
    
    # 提取JSON结果
    js = extract_json_from_text(response)
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

    parser.add_argument("--base_model", type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--out_dir", type=str, default="lora-netflow")
    parser.add_argument("--adapter", type=str, default=None)  # 用于 eval / predict

    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_bs", type=int, default=1)  # 降低batch size
    parser.add_argument("--grad_accum", type=int, default=16)    # 增加梯度累积
    parser.add_argument("--max_len", type=int, default=4096)     # 增加最大长度

    parser.add_argument("--mode", type=str, default="train", choices=["train","eval","predict"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    set_seed(args.seed)

    # Tokenizer & Model
    print("[*] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[*] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_4bit=args.load_in_4bit,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    if args.mode == "train":
        assert args.train_path and args.val_path, "Need --train_path and --val_path for training."

        print("[*] Loading datasets...")
        # Build datasets
        train_ds_raw = make_hf_dataset(args.train_path)
        val_ds_raw   = make_hf_dataset(args.val_path)
        
        print(f"[*] Train samples: {len(train_ds_raw)}, Val samples: {len(val_ds_raw)}")

        def _map_fn(ex): 
            return tokenize_with_label_mask(ex, tok, max_len=args.max_len)
        
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
        args_tr = TrainingArguments(
            output_dir=args.out_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.per_device_bs,
            per_device_eval_batch_size=args.per_device_bs,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=100,
            logging_steps=20,
            bf16=args.bf16,
            fp16=not args.bf16,
            report_to="none",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        collator = SimpleDataCollator(tokenizer=tok)
        trainer = Trainer(
            model=model,
            args=args_tr,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator
        )

        print("[*] Start training...")
        trainer.train()
        print("[*] Saving adapter to:", args.out_dir)
        model.save_pretrained(args.out_dir)
        tok.save_pretrained(args.out_dir)
        print("[*] Training completed.")

    elif args.mode == "eval":
        assert args.eval_path and (args.adapter or args.out_dir), "Need --eval_path and --adapter"
        adapter_dir = args.adapter or args.out_dir
        
        print(f"[*] Loading LoRA adapter from {adapter_dir}...")
        # 重新加载模型并应用LoRA
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_4bit=args.load_in_4bit,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_cfg)
        model.load_adapter(adapter_dir, adapter_name="default")
        model.set_adapter("default")

        print("[*] Evaluating...")
        eval_rows = read_jsonl(args.eval_path)
        preds, refs = [], []
        
        for i, row in enumerate(eval_rows):
            if i % 10 == 0:
                print(f"[*] Processing {i+1}/{len(eval_rows)}")
            
            ref = row["label"]
            js = generate_one(row, tok, model, temperature=0.0)
            pred = js.get("label", "")
            preds.append(pred)
            refs.append(ref)
            
            # 打印一些样例结果
            if i < 5:
                print(f"Sample {i+1}:")
                print(f"  Reference: {ref}")
                print(f"  Predicted: {pred}")
                print(f"  Explanation: {js.get('explanation', '')}")
                print()

        acc = simple_accuracy(preds, refs)
        f1  = macro_f1(preds, refs)
        result = {"accuracy": acc, "macro_f1": f1, "total_samples": len(eval_rows)}
        print(json.dumps(result, indent=2))

    else:  # predict
        assert (args.predict_path or args.predict_inline) and (args.adapter or args.out_dir), \
            "Need --adapter and either --predict_path or --predict_inline"
        adapter_dir = args.adapter or args.out_dir

        print(f"[*] Loading LoRA adapter from {adapter_dir}...")
        # 载入 + 启用 LoRA
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_4bit=args.load_in_4bit,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_cfg)
        model.load_adapter(adapter_dir, adapter_name="default")
        model.set_adapter("default")

        if args.predict_path:
            with open(args.predict_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            obj = json.loads(args.predict_inline)

        print("[*] Generating prediction...")
        out = generate_one(obj, tok, model, temperature=0.0)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        
if __name__ == "__main__":
    main()
