#!/bin/bash
# DeepShield LoRA 训练 - 完整设置和训练脚本

set -e  # 遇到错误立即退出

echo "=================================================="
echo "DeepShield LoRA 网络流量分类器 - 自动设置和训练"
echo "=================================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否有sudo权限
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        echo -e "${YELLOW}注意: 需要sudo权限来安装系统包${NC}"
        echo "请运行: sudo -v"
        exit 1
    fi
}

# 步骤1: 安装pip
install_pip() {
    echo -e "\n${GREEN}[1/5] 检查并安装 pip...${NC}"
    if ! python3 -m pip --version &>/dev/null; then
        echo "安装 pip..."
        check_sudo
        sudo apt update
        sudo apt install -y python3-pip
    else
        echo "✓ pip 已安装"
    fi
}

# 步骤2: 安装Python依赖
install_dependencies() {
    echo -e "\n${GREEN}[2/5] 安装 Python 依赖包...${NC}"
    echo "这可能需要10-30分钟，具体取决于网络速度..."
    
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt --no-cache-dir
    
    echo "✓ 依赖包安装完成"
}

# 步骤3: 验证安装
verify_installation() {
    echo -e "\n${GREEN}[3/5] 验证安装...${NC}"
    
    python3 -c "
import torch
import transformers
import datasets
import peft
print('✓ PyTorch version:', torch.__version__)
print('✓ Transformers version:', transformers.__version__)
print('✓ Datasets version:', datasets.__version__)
print('✓ PEFT version:', peft.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA devices:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('⚠ CUDA not available - 训练将在CPU上进行（非常慢）')
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 所有依赖验证通过${NC}"
    else
        echo -e "${RED}✗ 依赖验证失败${NC}"
        exit 1
    fi
}

# 步骤4: 验证数据
verify_data() {
    echo -e "\n${GREEN}[4/5] 验证数据文件...${NC}"
    
    python3 verify_setup.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 数据验证通过${NC}"
    else
        echo -e "${RED}✗ 数据验证失败${NC}"
        exit 1
    fi
}

# 步骤5: 开始训练
start_training() {
    echo -e "\n${GREEN}[5/5] 开始模型训练...${NC}"
    echo "训练参数："
    echo "  - 训练数据: data/processed/llm_input_enriched_train.jsonl"
    echo "  - 验证数据: data/processed/llm_input_enriched_val.jsonl"
    echo "  - 输出目录: lora-netflow"
    echo "  - 基础模型: mistralai/Mistral-7B-Instruct-v0.2"
    echo "  - Batch size: 1 (梯度累积: 16)"
    echo "  - 学习率: 2e-4"
    echo "  - 训练轮数: 3"
    echo ""
    
    # 创建输出目录
    mkdir -p lora-netflow
    
    # 开始训练
    python3 train_lora_netflow_refined.py \
        --train_path data/processed/llm_input_enriched_train.jsonl \
        --val_path data/processed/llm_input_enriched_val.jsonl \
        --out_dir lora-netflow \
        --epochs 3 \
        --per_device_bs 1 \
        --grad_accum 16 \
        --lr 2e-4 \
        --max_len 4096 \
        --seed 42
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}=================================================="
        echo "🎉 训练完成！"
        echo "=================================================="
        echo "模型已保存到: lora-netflow/"
        echo ""
        echo "下一步操作："
        echo "1. 评估模型:"
        echo "   python3 train_lora_netflow_refined.py \\"
        echo "       --eval_path data/processed/llm_input_enriched_test.jsonl \\"
        echo "       --adapter lora-netflow --mode eval"
        echo ""
        echo "2. 预测单个样本:"
        echo "   python3 train_lora_netflow_refined.py \\"
        echo "       --predict_path your_sample.json \\"
        echo "       --adapter lora-netflow --mode predict"
        echo "=================================================="
        echo -e "${NC}"
    else
        echo -e "\n${RED}训练过程出错，请检查日志${NC}"
        exit 1
    fi
}

# 主流程
main() {
    cd /home/ubuntu/Workspace/DeepShield
    
    # 如果传入了 --skip-install 参数，跳过安装步骤
    if [ "$1" == "--skip-install" ]; then
        echo "跳过安装步骤，直接开始训练..."
        verify_data
        start_training
    else
        # 完整流程
        install_pip
        install_dependencies
        verify_installation
        verify_data
        start_training
    fi
}

# 运行主流程
main "$@"
