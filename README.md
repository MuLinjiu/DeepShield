# DeepShield LoRA - Network Flow Security Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于LoRA（Low-Rank Adaptation）技术的网络流量安全分类器，使用大语言模型进行网络攻击检测和分类。

## ✨ 特性

- 🎯 **精确分类**：支持多种网络攻击类型识别（DDoS、端口扫描、暴力破解等）
- 🧠 **可解释性**：自动生成分类原因说明
- 💡 **参数高效**：使用LoRA技术，只训练少量参数
- 🖥️ **灵活部署**：支持CPU和GPU训练
- 📊 **丰富特征**：整合网络流量统计特征和载荷分析

---

## 🚀 快速开始

### GPU训练（推荐）

如果您有NVIDIA GPU（推荐16GB+显存）：

```bash
# 克隆仓库
git clone https://github.com/yourusername/DeepShield.git
cd DeepShield

# 一键训练
bash setup_and_train.sh
```

### CPU训练（低资源环境）

⚠️ **注意**：CPU训练需要至少8GB RAM，4GB内存环境建议使用云GPU服务。

```bash
# 使用CPU优化版本
bash setup_cpu_training.sh
```

---

## 📊 数据格式

训练数据格式（JSONL）：
```json
{
  "flow_id": 1,
  "tuple5": ["src_ip", "dst_ip", src_port, dst_port, proto],
  "window": [start_time, end_time],
  "features": {...},
  "enriched": {...},
  "label": "BENIGN"
}
```

---

## 📁 项目文件

```
DeepShield/
├── train_cpu_optimized.py          # CPU优化训练程序（当前使用）
├── setup_cpu_training.sh           # CPU一键训练脚本
├── train_lora_netflow_refined.py  # GPU版本训练程序
├── setup_and_train.sh              # GPU一键训练脚本
├── requirements_cpu.txt            # CPU依赖
├── requirements.txt                # GPU依赖
└── data/processed/                 # 训练数据（3.5GB）
```

---

## 💻 CPU训练（当前配置）

### 方式1：一键启动
```bash
bash setup_cpu_training.sh
```

### 方式2：后台运行（推荐）
```bash
nohup bash setup_cpu_training.sh > training.log 2>&1 &
tail -f training.log  # 查看进度
```

### 方式3：快速测试（5分钟）
```bash
bash setup_cpu_training.sh --quick-test
```

---

## 🎮 GPU训练（如有GPU服务器）

### 一键启动
```bash
bash setup_and_train.sh
```

### 手动训练
```bash
python3 train_lora_netflow_refined.py \
    --train_path data/processed/llm_input_enriched_train.jsonl \
    --val_path data/processed/llm_input_enriched_val.jsonl \
    --out_dir lora-netflow-gpu \
    --epochs 3
```

---

## 📈 训练完成后

### 评估模型（CPU版本）
```bash
python3 train_cpu_optimized.py \
    --eval_path data/cpu_test_100.jsonl \
    --adapter lora-netflow-cpu \
    --mode eval
```

### 评估模型（GPU版本）
```bash
python3 train_lora_netflow_refined.py \
    --eval_path data/processed/llm_input_enriched_test.jsonl \
    --adapter lora-netflow-gpu \
    --mode eval
```

### 预测单个样本
```bash
python3 train_cpu_optimized.py \
    --predict_path sample.json \
    --adapter lora-netflow-cpu \
    --mode predict
```

---

## ⚙️ CPU vs GPU 对比

| 特性 | CPU版本 | GPU版本 |
|------|---------|---------|
| 模型 | TinyLlama-1.1B | Mistral-7B |
| 训练样本 | 1000条 | 全部（数十万） |
| 序列长度 | 1024 | 4096 |
| 训练时间 | 2-4小时 | 4-8小时 |
| 硬件要求 | 8GB RAM | 16GB+ GPU |
| 模型效果 | 较低但可用 | 更好 |

---

## 🛠️ 技术细节

### 训练策略
- 训练时输出：`{"label": "BENIGN", "explanation": ""}`
- 只对label值进行监督学习
- 推理时自动生成explanation

### LoRA配置
- CPU: rank=8, alpha=16
- GPU: rank=16, alpha=32

### 优化措施
- CPU版本使用小模型和少量数据
- 精确的token级损失掩码
- 梯度累积减少内存占用

---

## 📝 更多信息

- **CPU训练详情**: 查看 `START_HERE_CPU.md`
- **GPU训练详情**: 查看 `README_refined.md`

---

## 🐛 问题排查

### CPU训练太慢
- 减少数据：修改脚本中的`head -n 1000`为`head -n 100`
- 减少轮数：`--epochs 1`
- 减少序列长度：`--max_len 512`

### 内存不足
- 降低batch size（已经是1）
- 减少序列长度
- 减少训练样本

### 依赖安装失败
```bash
# 单独安装CPU版PyTorch
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install transformers datasets peft accelerate
```

---

## 🏗️ 架构说明

### 训练策略

本项目采用创新的**部分监督训练**策略：

1. **训练阶段**：输出格式为 `{"label": "DDoS", "explanation": ""}`
   - 只对`label`字段的值进行损失计算
   - `explanation`字段为空，不参与训练

2. **推理阶段**：模型自动补全 `explanation`
   - 输出：`{"label": "DDoS", "explanation": "检测到大量SYN包..."}`

这种方法使模型既学会分类，又能生成解释。

### 模型选择

| 环境 | 模型 | 参数量 | 显存/内存 |
|------|------|--------|----------|
| GPU | Mistral-7B-Instruct | 7B | 16GB+ |
| CPU | TinyLlama-1.1B-Chat | 1.1B | 8GB+ |

---

## 📂 项目结构

```
DeepShield/
├── train_lora_netflow_refined.py  # GPU训练程序
├── train_cpu_optimized.py         # CPU训练程序
├── setup_and_train.sh              # GPU一键脚本
├── setup_cpu_training.sh           # CPU一键脚本
├── requirements.txt                # GPU依赖
├── requirements_cpu.txt            # CPU依赖
├── .gitignore                      # Git忽略配置
└── data/                           # 数据目录（需自备）
    ├── processed/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── test.jsonl
    └── sample_*.jsonl             # 示例数据
```

---

## 📝 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{deepshield2025,
  title={DeepShield: Network Flow Security Classifier with LoRA},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/DeepShield}
}
```

---

## 📄 License

本项目采用 MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## ⚠️ 免责声明

本工具仅用于安全研究和教育目的。使用者需遵守当地法律法规。

---

**推荐环境：GPU服务器（Google Colab / AWS / 云服务器）** 🚀
