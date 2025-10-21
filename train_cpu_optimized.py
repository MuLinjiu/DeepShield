#!/usr/bin/env python3
# filename: train_cpu_optimized.py
# CPU优化版本 - 使用更小的模型和更少的资源

import os, json, argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
from datasets import Dataset
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
    flow_input = {k: v for k, v in flow_record.items() if k != "label"}
    return f"{SYSTEM_PROMPT}Flow Record:\n{json.dumps(flow_input, ensure_ascii=False, indent=2)}"

def build_completion_train(label: str) -> str:
    """训练时的completion格式：explanation为空字符串"""
    return json.dumps({"label": label, "explanation": ""}, ensure_ascii=False)

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

def make_hf_dataset(path: str, max_samples: int = None) -> Dataset:
    """根据新的schema格式处理数据，可以限制样本数量"""
    data = read_jsonl(path)
    if max_samples:
        data = data[:max_samples]
        print(f"[*] Limited to {len(data)} samples for CPU training")
    
    rows = []
    for d in data:
        if "label" not in d:
            continue
        label = d["label"]
        rows.append({
            "prompt": build_prompt(d),
            "completion": build_completion_train(label),
            "label": label
        })
    return Dataset.from_list(rows)

def tokenize_with_label_mask(ex, tok, max_len=2048):
    """只对label值进行监督学习"""
    full = ex["prompt"].strip() + "\n\n" + ex["completion"].strip()
    enc = tok(full, truncation=True, max_length=max_len)
    labels = enc["input_ids"][:]

    comp = ex["completion"].strip()
    start = full.rfind(comp)
    
    if start == -1:
        labels = [-100] * len(labels)
        enc["labels"] = labels
        return enc

    try:
        label_key = '"label": "'
        i0 = comp.index(label_key) + len(label_key)
        i1 = comp.index('"', i0)
        lab_abs_start = start + i0
        lab_abs_end = start + i1
    except ValueError:
        labels = [-100] * len(labels)
        enc["labels"] = labels
        return enc

    try:
        enc_off = tok(
            full, add_special_tokens=False, return_offsets_mapping=True,
            truncation=True, max_length=max_len
        )
        offsets = enc_off["offset_mapping"]
    except:
        labels = [-100] * len(labels)
        enc["labels"] = labels
        return enc

    n = min(len(labels), len(offsets))
    for t_idx in range(n):
        a, b = offsets[t_idx]
        overlap = not (b <= lab_abs_start or a >= lab_abs_end)
        if not overlap:
            labels[t_idx] = -100
    
    for t_idx in range(n, len(labels)):
        labels[t_idx] = -100

    enc["labels"] = labels
    return enc

@dataclass
class SimpleDataCollator:
    tokenizer: AutoTokenizer
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True, return_tensors="pt"
        )
        
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for l in labels:
            pad_len = max_len - len(l)
            padded = l + [-100] * pad_len
            padded_labels.append(padded)
        
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

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
    if "{" not in text or "}" not in text:
        return None
    
    json_starts = []
    for i, char in enumerate(text):
        if char == '{':
            json_starts.append(i)
    
    for start in reversed(json_starts):
        for end in range(len(text), start, -1):
            if text[end-1] == '}':
                try:
                    candidate = text[start:end]
                    return json.loads(candidate)
                except:
                    continue
    
    import re
    label_match = re.search(r'"label"\s*:\s*"([^"]*)"', text)
    if label_match:
        return {"label": label_match.group(1), "explanation": ""}
    
    return None

def generate_one(flow_record: Dict[str, Any], tok, model, max_new_tokens=256, temperature=0.0) -> Dict[str, Any]:
    prompt = build_prompt(flow_record)
    inputs = tok(prompt, return_tensors="pt")
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    
    generated_text = tok.decode(out[0], skip_special_tokens=True)
    prompt_len = len(prompt)
    if len(generated_text) > prompt_len:
        response = generated_text[prompt_len:].strip()
    else:
        response = generated_text
    
    js = extract_json_from_text(response)
    return js if isinstance(js, dict) else {"label": "", "explanation": ""}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)

    # CPU优化：使用更小的模型
    parser.add_argument("--base_model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # 1.1B参数，CPU友好
    parser.add_argument("--out_dir", type=str, default="lora-netflow-cpu")
    parser.add_argument("--adapter", type=str, default=None)

    # LoRA参数
    parser.add_argument("--r", type=int, default=8)  # 降低rank
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    
    # 训练参数 - CPU优化
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_bs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)  # 减少累积
    parser.add_argument("--max_len", type=int, default=1024)  # 大幅减少序列长度
    parser.add_argument("--max_train_samples", type=int, default=1000)  # 限制训练样本
    parser.add_argument("--max_val_samples", type=int, default=200)

    parser.add_argument("--mode", type=str, default="train", choices=["train","eval","predict"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    print(f"[*] CPU Training Mode - Optimized Configuration")
    print(f"[*] Model: {args.base_model}")
    print(f"[*] Max sequence length: {args.max_len}")
    print(f"[*] Max training samples: {args.max_train_samples}")

    # Tokenizer & Model
    print("[*] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[*] Loading base model (CPU mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,  # CPU使用float32
        low_cpu_mem_usage=True,
    )

    if args.mode == "train":
        assert args.train_path and args.val_path, "Need --train_path and --val_path for training."

        print("[*] Loading datasets (limited samples for CPU)...")
        train_ds_raw = make_hf_dataset(args.train_path, max_samples=args.max_train_samples)
        val_ds_raw = make_hf_dataset(args.val_path, max_samples=args.max_val_samples)
        
        print(f"[*] Train samples: {len(train_ds_raw)}, Val samples: {len(val_ds_raw)}")

        def _map_fn(ex): 
            return tokenize_with_label_mask(ex, tok, max_len=args.max_len)
        
        print("[*] Tokenizing datasets...")
        train_ds = train_ds_raw.map(_map_fn, remove_columns=train_ds_raw.column_names, desc="Tokenizing train")
        val_ds = val_ds_raw.map(_map_fn, remove_columns=val_ds_raw.column_names, desc="Tokenizing val")

        print("[*] Setting up LoRA...")
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.r, 
            lora_alpha=args.alpha, 
            lora_dropout=args.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

        args_tr = TrainingArguments(
            output_dir=args.out_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.per_device_bs,
            per_device_eval_batch_size=args.per_device_bs,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=50,
            logging_steps=10,
            fp16=False,  # CPU不支持fp16
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            dataloader_num_workers=0,  # CPU模式设为0
        )

        collator = SimpleDataCollator(tokenizer=tok)
        trainer = Trainer(
            model=model,
            args=args_tr,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator
        )

        print("[*] Start training (CPU mode - this will be slower)...")
        trainer.train()
        print("[*] Saving adapter to:", args.out_dir)
        model.save_pretrained(args.out_dir)
        tok.save_pretrained(args.out_dir)
        print("[*] Training completed.")

    elif args.mode == "eval":
        assert args.eval_path and args.adapter, "Need --eval_path and --adapter"
        
        print(f"[*] Loading LoRA adapter from {args.adapter}...")
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_cfg)
        model.load_adapter(args.adapter, adapter_name="default")
        model.set_adapter("default")

        print("[*] Evaluating...")
        eval_rows = read_jsonl(args.eval_path)[:100]  # 限制评估样本
        preds, refs = [], []
        
        for i, row in enumerate(eval_rows):
            if i % 10 == 0:
                print(f"[*] Processing {i+1}/{len(eval_rows)}")
            
            ref = row["label"]
            js = generate_one(row, tok, model, temperature=0.0)
            pred = js.get("label", "")
            preds.append(pred)
            refs.append(ref)

        acc = simple_accuracy(preds, refs)
        f1 = macro_f1(preds, refs)
        result = {"accuracy": acc, "macro_f1": f1, "total_samples": len(eval_rows)}
        print(json.dumps(result, indent=2))

    else:  # predict
        assert args.predict_path and args.adapter, "Need --adapter and --predict_path"
        
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_cfg)
        model.load_adapter(args.adapter, adapter_name="default")
        model.set_adapter("default")

        with open(args.predict_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        print("[*] Generating prediction...")
        out = generate_one(obj, tok, model, temperature=0.0)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        
if __name__ == "__main__":
    main()

