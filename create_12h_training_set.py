#!/usr/bin/env python3
"""
创建12小时内可以完成训练的数据集
基于类别平衡采样策略
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

def estimate_training_time(num_samples, batch_size=1, grad_accum=16, epochs=3,
                          time_per_batch_sec=2.0):
    """
    估算训练时间

    参数:
    - num_samples: 训练样本数
    - batch_size: per_device_batch_size
    - grad_accum: gradient_accumulation_steps
    - epochs: 训练轮数
    - time_per_batch_sec: 每个batch的平均时间(秒)
    """
    effective_batch_size = batch_size * grad_accum
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = steps_per_epoch * epochs
    total_time_sec = total_steps * time_per_batch_sec
    total_time_hours = total_time_sec / 3600

    return {
        'samples': num_samples,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'total_time_hours': total_time_hours,
        'total_time_minutes': total_time_sec / 60
    }

def sample_training_data(input_train, input_val, output_train, output_val,
                        target_samples=20000, target_hours=12, seed=42):
    """
    采样训练数据,确保在目标时间内完成

    策略:
    1. 攻击类样本: 全部保留(稀有且重要)
    2. BENIGN样本: 采样到合适数量
    3. 保持类别平衡(攻击类不能被BENIGN淹没)
    """
    random.seed(seed)

    print("="*80)
    print("Creating 12-Hour Training Dataset")
    print("="*80)

    # 读取训练数据
    print(f"\n[1] Reading training data from {input_train}...")
    data_by_label = defaultdict(list)
    total_count = 0

    with open(input_train, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                print(f"    Processed {i:,} lines...")
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = obj.get('label', 'UNKNOWN')
            data_by_label[label].append(obj)
            total_count += 1

    print(f"\n[2] Original training set: {total_count:,} samples")
    for label in sorted(data_by_label.keys()):
        count = len(data_by_label[label])
        pct = 100.0 * count / total_count
        print(f"    {label:35s}: {count:8,} ({pct:5.2f}%)")

    # 采样策略
    print(f"\n[3] Sampling strategy for ~{target_samples:,} samples:")

    # 分离攻击类和BENIGN类
    attack_samples = []
    benign_samples = data_by_label.get('BENIGN', [])

    for label, items in data_by_label.items():
        if label != 'BENIGN':
            attack_samples.extend(items)
            print(f"    {label:35s}: Keep all {len(items):,} samples")

    num_attack = len(attack_samples)
    print(f"\n    Total attack samples: {num_attack:,}")

    # 计算BENIGN样本数量
    # 策略: 攻击类占20-30%,BENIGN占70-80%
    target_attack_ratio = 0.25  # 攻击类占25%
    num_benign_needed = int(num_attack * (1 - target_attack_ratio) / target_attack_ratio)

    # 确保不超过总样本数限制
    if num_attack + num_benign_needed > target_samples:
        num_benign_needed = target_samples - num_attack

    # 确保不超过实际BENIGN样本数
    num_benign_needed = min(num_benign_needed, len(benign_samples))

    print(f"    BENIGN: Sample {num_benign_needed:,} from {len(benign_samples):,} samples")

    # 采样BENIGN
    sampled_benign = random.sample(benign_samples, num_benign_needed)

    # 合并所有样本
    final_samples = attack_samples + sampled_benign
    random.shuffle(final_samples)

    # 统计最终分布
    final_counts = defaultdict(int)
    for obj in final_samples:
        final_counts[obj['label']] += 1

    print(f"\n[4] Final training set: {len(final_samples):,} samples")
    for label in sorted(final_counts.keys()):
        count = final_counts[label]
        pct = 100.0 * count / len(final_samples)
        print(f"    {label:35s}: {count:8,} ({pct:5.2f}%)")

    # 估算训练时间
    print(f"\n[5] Training time estimation:")

    # 测试不同配置
    configs = [
        {'batch_size': 1, 'grad_accum': 16, 'time_per_batch_sec': 2.0, 'epochs': 3},
        {'batch_size': 2, 'grad_accum': 8, 'time_per_batch_sec': 2.5, 'epochs': 3},
        {'batch_size': 1, 'grad_accum': 16, 'time_per_batch_sec': 2.0, 'epochs': 2},
    ]

    for i, config in enumerate(configs):
        est = estimate_training_time(len(final_samples), **config)
        print(f"\n    Config {i+1}: bs={config['batch_size']}, "
              f"grad_accum={config['grad_accum']}, epochs={config['epochs']}")
        print(f"      Steps/epoch: {est['steps_per_epoch']:,}")
        print(f"      Total steps: {est['total_steps']:,}")
        print(f"      Estimated time: {est['total_time_hours']:.1f}h ({est['total_time_minutes']:.0f}min)")

    # 写入训练集
    print(f"\n[6] Writing training set to {output_train}...")
    Path(output_train).parent.mkdir(parents=True, exist_ok=True)
    with open(output_train, 'w', encoding='utf-8') as f:
        for obj in final_samples:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # 处理验证集 - 采样10%的训练集大小
    print(f"\n[7] Processing validation set...")
    val_target = max(1000, len(final_samples) // 10)  # 至少1000个样本

    val_data_by_label = defaultdict(list)
    with open(input_val, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = obj.get('label', 'UNKNOWN')
            val_data_by_label[label].append(obj)

    # 按比例采样验证集
    val_samples = []
    for label, items in val_data_by_label.items():
        if label == 'BENIGN':
            # BENIGN采样
            ratio = num_benign_needed / len(benign_samples)
            num_val = int(len(items) * ratio)
            num_val = min(num_val, len(items))
        else:
            # 攻击类全部保留或采样
            num_val = min(len(items), val_target // (len(val_data_by_label) - 1))

        if num_val > 0:
            sampled = random.sample(items, num_val) if num_val < len(items) else items
            val_samples.extend(sampled)

    # 限制验证集大小
    if len(val_samples) > val_target:
        val_samples = random.sample(val_samples, val_target)

    random.shuffle(val_samples)

    print(f"    Validation set: {len(val_samples):,} samples")

    # 写入验证集
    print(f"\n[8] Writing validation set to {output_val}...")
    with open(output_val, 'w', encoding='utf-8') as f:
        for obj in val_samples:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print("Done! Dataset created successfully.")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Train with:")
    print(f"   python3 train_lora_netflow_refined.py \\")
    print(f"     --train_path {output_train} \\")
    print(f"     --val_path {output_val} \\")
    print(f"     --out_dir lora-netflow-12h \\")
    print(f"     --base_model meta-llama/Llama-2-7b-hf \\")
    print(f"     --epochs 3")
    print(f"\n2. Expected training time: ~{est['total_time_hours']:.1f} hours")

    return final_samples, val_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create 12-hour training dataset")
    parser.add_argument("--input_train", type=str,
                       default="data/llm_input_enriched_train.jsonl",
                       help="Input training data")
    parser.add_argument("--input_val", type=str,
                       default="data/llm_input_enriched_val.jsonl",
                       help="Input validation data")
    parser.add_argument("--output_train", type=str,
                       default="data/train_12h.jsonl",
                       help="Output training data")
    parser.add_argument("--output_val", type=str,
                       default="data/val_12h.jsonl",
                       help="Output validation data")
    parser.add_argument("--target_samples", type=int, default=20000,
                       help="Target number of training samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    sample_training_data(
        args.input_train,
        args.input_val,
        args.output_train,
        args.output_val,
        target_samples=args.target_samples,
        seed=args.seed
    )
