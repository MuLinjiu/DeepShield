# DeepShield - Network Traffic Security Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于LoRA（Low-Rank Adaptation）技术的网络流量安全分类器，使用大语言模型进行网络攻击检测和分类。

## ✨ 特性

- 🎯 **精确分类**：支持多种网络攻击类型识别（Web Attack、Brute Force、Infiltration等）
- 🧠 **可解释性**：生成分类原因说明
- 💡 **参数高效**：使用LoRA技术，只训练少量参数
- 🚀 **灵活模型**：支持Llama-3.1、Qwen2.5/3等多种基座模型
- 📊 **丰富指标**：详细的per-class precision/recall/f1评估

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置HuggingFace Token（可选）

对于公开模型（如Llama-3.1），不需要token。如果使用gated模型，需要设置token：

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，填入你的token
# HF_TOKEN=hf_xxxxxxxxxx

# 或直接设置环境变量
export HF_TOKEN=your_token_here
```

### 3. 准备数据

创建平衡训练集（从原始数据集采样20K样本）：

```bash
python3 create_12h_training_set.py \
  --input_train data/processed/llm_input_enriched_train.jsonl \
  --input_val data/processed/llm_input_enriched_val.jsonl \
  --output_train data/processed/train_12h.jsonl \
  --output_val data/processed/val_12h.jsonl
```

创建自然分布测试集（10K样本，保持原始99% BENIGN比例）：

```bash
python3 create_natural_test_set.py
```

### 4. 训练模型

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_lora_netflow_refined.py \
  --train_path data/processed/train_12h.jsonl \
  --val_path data/processed/val_12h.jsonl \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --out_dir lora-llama31-12h \
  --epochs 3 \
  --max_len 1536 \
  --per_device_bs 2 \
  --grad_accum 8 \
  --load_in_4bit \
  --bf16
```

### 5. 评估模型

平衡测试集（100样本）：
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_lora_netflow_refined.py \
  --eval_path data/processed/llm_input_enriched_test_sample100.jsonl \
  --adapter lora-llama31-12h/checkpoint-800 \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --mode eval
```

自然分布测试集（10K样本，~99% BENIGN）：
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_lora_netflow_refined.py \
  --eval_path data/processed/llm_input_enriched_test_natural10k.jsonl \
  --adapter lora-llama31-12h/checkpoint-800 \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --mode eval
```

评估Base Model（无LoRA）：
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_lora_netflow_refined.py \
  --eval_path data/processed/llm_input_enriched_test_sample100.jsonl \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --mode eval
```

---

## 📊 数据格式

### 输入数据格式（JSONL）

```json
{
  "flow_id": 1,
  "tuple5": ["192.168.1.1", "10.0.0.1", 45123, 80, 6],
  "window": [1234567890.0, 1234567895.0],
  "features": {
    "packet_count": 150,
    "byte_count": 75000,
    "flow_dur_ms": 5000.0,
    "tcp_syn_ratio": 0.02,
    "payload_entropy": 6.5,
    ...
  },
  "enriched": {
    "protocols": ["HTTP", "TCP"]
  },
  "label": "Web Attack - XSS"
}
```

### 模型输出格式

```json
{
  "label": "Web Attack - XSS",
  "explanation": "HTTP traffic on port 80 with high packet count (150 packets) and large payload (75KB) showing high ASCII ratio (0.85) indicating text-based content"
}
```

---

## 🎮 支持的模型

### Llama系列
- `meta-llama/Llama-3.1-8B-Instruct` （推荐）
- `meta-llama/Llama-2-7b-hf`

### Qwen系列
- `Qwen/Qwen3-8B-Instruct` （最新）
- `Qwen/Qwen2.5-7B-Instruct`

### 其他
- `mistralai/Mistral-7B-Instruct-v0.2`

---

## 📁 项目结构

```
DeepShield/
├── train_lora_netflow_refined.py      # 主训练/评估脚本
├── create_12h_training_set.py         # 创建平衡训练集
├── create_natural_test_set.py         # 创建自然分布测试集
├── requirements.txt                    # Python依赖
├── .env.example                        # 环境变量模板
├── .gitignore                          # Git忽略配置
└── data/processed/                     # 数据目录
    ├── train_12h.jsonl                 # 平衡训练集（20K）
    ├── val_12h.jsonl                   # 验证集（2K）
    ├── llm_input_enriched_test_sample100.jsonl  # 测试集（100）
    └── llm_input_enriched_test_natural10k.jsonl # 自然分布测试集（10K）
```

---

## ⚙️ 训练参数说明

### 基础参数
- `--base_model`: 基座模型名称
- `--train_path`: 训练数据路径
- `--val_path`: 验证数据路径
- `--out_dir`: 输出目录
- `--adapter`: LoRA adapter路径（eval时使用）

### LoRA配置
- `--r`: LoRA rank（默认16）
- `--alpha`: LoRA alpha（默认32）
- `--dropout`: LoRA dropout（默认0.05）

### 训练配置
- `--epochs`: 训练轮数（默认3）
- `--lr`: 学习率（默认2e-4）
- `--per_device_bs`: 每设备batch size（默认1）
- `--grad_accum`: 梯度累积步数（默认16）
- `--max_len`: 最大序列长度（默认4096）

### 实验性参数
- `--remove_eos_from_training`: 去掉训练序列的EOS token，鼓励模型生成explanation
- `--disable_explanation_fallback`: 禁用hard-coded explanation生成fallback

---

## 📈 评估指标

评估时会输出：

### 总体指标
- Accuracy
- Macro Precision/Recall/F1
- Weighted Precision/Recall/F1

### Per-Class指标
- 每个类别的Precision、Recall、F1、Support

示例输出：
```
============================================================
Overall Metrics:
============================================================
Accuracy:          0.9850
Macro Precision:   0.8234
Macro Recall:      0.7891
Macro F1:          0.8058
Weighted Precision: 0.9823
Weighted Recall:    0.9850
Weighted F1:        0.9836
Total Samples:      10000

============================================================
Per-Class Metrics:
============================================================
Class                                Precision     Recall         F1    Support
--------------------------------------------------------------------------------
BENIGN                                  0.9900     0.9990     0.9945       9911
Web Attack - Brute Force                0.7500     0.7500     0.7500         37
Web Attack - XSS                        0.8333     0.8333     0.8333         27
...
```

---

## 🛠️ 技术细节

### 训练策略

训练数据格式：
```json
{"label": "Web Attack - XSS", "explanation": "
```

- Completion在引号未闭合处结束
- 不包含EOS token（使用 `--remove_eos_from_training`）
- 鼓励模型在eval时继续生成explanation

### LoRA配置
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- 4-bit量化（`load_in_4bit`）
- BF16混合精度训练

### 学习率调度
- Scheduler: Cosine with warmup
- Warmup steps: 100
- Learning rate: 2e-4

---

## 🐛 常见问题

### 1. HuggingFace Token错误
```bash
# 设置环境变量
export HF_TOKEN=your_token_here

# 或使用.env文件
cp .env.example .env
# 编辑.env填入token
```

### 2. CUDA Out of Memory
- 减少 `--per_device_bs`（尝试1）
- 增加 `--grad_accum`（尝试16或32）
- 减少 `--max_len`（尝试1024或512）
- 使用4-bit量化（`--load_in_4bit`）

### 3. 训练太慢
- 增加 `--per_device_bs`（如果显存允许）
- 减少 `--grad_accum`
- 使用更少数据或更少epochs

---

## 📄 License

MIT License

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## ⚠️ 免责声明

本工具仅用于安全研究和教育目的。使用者需遵守当地法律法规。
