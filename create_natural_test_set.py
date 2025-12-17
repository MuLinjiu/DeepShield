#!/usr/bin/env python3
"""
创建按原始分布采样的测试集
从 llm_input_enriched_test.jsonl 中采样1000条，保持原始分布（~99% BENIGN）
"""

import json
import random
from collections import Counter

random.seed(42)

def main():
    # 读取完整测试集
    print("[*] Loading full test set...")
    test_data = []
    with open("data/llm_input_enriched_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    print(f"[*] Total test samples: {len(test_data)}")

    # 统计原始分布
    labels = [sample["label"] for sample in test_data]
    counter = Counter(labels)
    print("\n[*] Original distribution:")
    for label, count in counter.most_common():
        pct = count / len(test_data) * 100
        print(f"  {label:30s}: {count:6d} ({pct:5.2f}%)")

    # 随机采样5000条（保持原始分布）
    print("\n[*] Sampling 5000 samples (natural distribution)...")
    sampled = random.sample(test_data, 5000)

    # 统计采样后的分布
    sampled_labels = [sample["label"] for sample in sampled]
    sampled_counter = Counter(sampled_labels)
    print("\n[*] Sampled distribution:")
    for label, count in sampled_counter.most_common():
        pct = count / len(sampled) * 100
        print(f"  {label:30s}: {count:6d} ({pct:5.2f}%)")

    # 保存
    output_path = "data/llm_input_enriched_test_natural5k.jsonl"
    print(f"\n[*] Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in sampled:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[*] Done! Saved {len(sampled)} samples")

if __name__ == "__main__":
    main()
