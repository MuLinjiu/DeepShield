"""
Minimal inference script for the DeepShield LoRA checkpoint.

Features:
- Loads the base model and your LoRA adapter from a checkpoint directory (default: checkpoint-900).
- Accepts a raw prompt string or a JSON flow record and returns the generated text.
- Uses the same system prompt and chat template style as the training script.

Usage examples:
  python inference.py --prompt "Describe this traffic: ..."                          \
         --checkpoint checkpoint-900 --base-model meta-llama/Llama-3.1-8B-Instruct

  python inference.py --flow-json sample_flow.json --checkpoint checkpoint-900

  HF_TOKEN=your_token python inference.py --prompt "..."  # if the base model is gated
"""

import argparse
import json
import os
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Same system prompt used during training to keep behavior consistent
SYSTEM_PROMPT = (
    "You are a network traffic security analyst.\n"
    "Analyze the network flow record and classify it as BENIGN or a specific attack type.\n"
    "Output a JSON with two fields:\n"
    '- "label": The security classification (BENIGN, Web Attack - XSS, Web Attack - Brute Force, '
    'Web Attack - Sql Injection, Infiltration, DDoS, PortScan, Bot, etc.)\n'
    '- "explanation": A brief analysis citing specific features from the flow that support your classification. '
    "IMPORTANT: The explanation MUST describe the evidence, such as unusual packet counts, suspicious protocols, "
    "abnormal payload patterns, TCP flag combinations, port numbers, or timing patterns.\n\n"
    "Example output format:\n"
    '{"label": "Web Attack - XSS", "explanation": "High packet count (1250 pkts) with HTTP traffic '
    'to port 80, payload contains script tags and suspicious ASCII patterns (ratio 0.85)"}\n\n'
    "IMPORTANT: You MUST provide a non-empty explanation describing the key features that led to your classification.\n\n"
    "Analyze the following flow:\n"
)


def build_prompt_from_flow(flow_record: Dict[str, Any]) -> str:
    """Create a concise, readable prompt from a flow JSON record (mirrors training logic)."""
    features = flow_record.get("features", {})
    enriched = flow_record.get("enriched", {})
    tuple5 = flow_record.get("tuple5", [])

    summary = []
    if len(tuple5) >= 5:
        src_ip, dst_ip, dst_port, src_port, protocol = tuple5
        summary.append(f"Connection: {src_ip}:{src_port} -> {dst_ip}:{dst_port} (proto={protocol})")

    if features:
        summary.append(
            f"Packets: {features.get('packet_count', 0)} total "
            f"(fwd={features.get('pkt_cnt_fwd', 0)}, bwd={features.get('pkt_cnt_bwd', 0)})"
        )
        summary.append(
            f"Bytes: {features.get('byte_count', 0)} total "
            f"(fwd={features.get('byte_cnt_fwd', 0)}, bwd={features.get('byte_cnt_bwd', 0)})"
        )
        summary.append(f"Flow duration: {features.get('flow_dur_ms', 0):.2f}ms")

        tcp_info = []
        for key, label in [
            ("tcp_syn_ratio", "SYN"),
            ("tcp_ack_ratio", "ACK"),
            ("tcp_psh_ratio", "PSH"),
            ("tcp_fin_ratio", "FIN"),
            ("tcp_rst_ratio", "RST"),
        ]:
            val = features.get(key, 0)
            if val and val > 0:
                tcp_info.append(f"{label}={val:.2f}")
        if tcp_info:
            summary.append(f"TCP flags: {', '.join(tcp_info)}")

        if features.get("payload_total_len", 0) > 0:
            summary.append(
                f"Payload: {features['payload_total_len']} bytes, "
                f"entropy={features.get('payload_entropy', 0):.2f}, "
                f"ASCII ratio={features.get('payload_ascii_ratio', 0):.2f}"
            )

    if enriched:
        protocols = enriched.get("protocols", [])
        if protocols:
            summary.append(f"Protocols: {', '.join(protocols[:5])}")
        context = enriched.get("context_summary", "")
        if context:
            summary.append(f"Context: {context[:150]}")

    return "\n".join(summary)


def format_chat(prompt: str, tok: AutoTokenizer, add_system: bool = True) -> str:
    """Render the chat-style prompt that matches training."""
    if add_system:
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n" + prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]

    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + "\n\n"


def load_tokenizer_and_model(
    base_model: str,
    adapter_dir: str,
    load_in_4bit: bool = True,
    bf16: bool = True,
) -> tuple[AutoTokenizer, torch.nn.Module]:
    """Load base model + LoRA adapter."""
    hf_token = os.getenv("HF_TOKEN", None)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        token=hf_token,
    )
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    return tok, model


def generate(
    prompt_text: str,
    tok: AutoTokenizer,
    model: torch.nn.Module,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Run generation with safe prompt truncation."""
    ctx_window = getattr(tok, "model_max_length", 4096)
    max_prompt_len = max(ctx_window - max_new_tokens - 32, 64)

    inputs = tok(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_len,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "repetition_penalty": 1.1,
    }

    # Remove None to avoid HF warnings
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated = out[0][input_len:]
    return tok.decode(generated, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the DeepShield LoRA checkpoint.")
    parser.add_argument("--prompt", type=str, help="Raw text prompt.")
    parser.add_argument("--prompt-file", type=str, help="File that contains the prompt text.")
    parser.add_argument("--flow-json", type=str, help="Path to a JSON file of a single flow record.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-900", help="LoRA checkpoint directory.")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model used during fine-tuning.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-system-prompt", action="store_true", help="Do not prepend the training system prompt.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading (use full precision).")
    parser.add_argument("--no-bf16", action="store_true", help="Force fp16 when not using 4-bit.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not any([args.prompt, args.prompt_file, args.flow_json]):
        raise SystemExit("Please provide --prompt, --prompt-file, or --flow-json.")

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            user_prompt = f.read().strip()
    elif args.flow_json:
        with open(args.flow_json, "r", encoding="utf-8") as f:
            flow = json.load(f)
        user_prompt = build_prompt_from_flow(flow)
    else:
        user_prompt = args.prompt.strip()

    tok, model = load_tokenizer_and_model(
        base_model=args.base_model,
        adapter_dir=args.checkpoint,
        load_in_4bit=not args.no_4bit,
        bf16=not args.no_bf16,
    )

    chat_prompt = format_chat(user_prompt, tok, add_system=not args.no_system_prompt)
    response = generate(
        chat_prompt,
        tok,
        model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(response)


if __name__ == "__main__":
    main()
