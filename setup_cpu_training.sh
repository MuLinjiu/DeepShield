#!/bin/bash
# CPU训练配置和启动脚本

set -e

echo "=================================================="
echo "DeepShield LoRA - CPU优化版本训练"
echo "=================================================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 步骤1: 安装依赖
install_dependencies() {
    echo -e "\n${GREEN}[1/4] 安装CPU版本依赖...${NC}"
    echo "这将安装纯CPU版本的PyTorch（无CUDA支持）"
    
    # 检查pip
    if ! python3 -m pip --version &>/dev/null; then
        echo "安装 pip..."
        sudo apt update
        sudo apt install -y python3-pip
    fi
    
    # 安装CPU版本的PyTorch
    echo "安装PyTorch (CPU版本)..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # 安装其他依赖
    echo "安装其他依赖..."
    python3 -m pip install transformers datasets peft accelerate scipy numpy tqdm sentencepiece protobuf
    
    echo -e "${GREEN}✓ 依赖安装完成${NC}"
}

# 步骤2: 验证安装
verify_installation() {
    echo -e "\n${GREEN}[2/4] 验证安装...${NC}"
    
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
print('✓ CPU threads:', torch.get_num_threads())

if torch.cuda.is_available():
    print('⚠️  检测到GPU，但我们将使用CPU版本训练')
else:
    print('✓ 将使用CPU进行训练（速度较慢但可行）')
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 环境验证通过${NC}"
    else
        echo -e "${RED}✗ 环境验证失败${NC}"
        exit 1
    fi
}

# 步骤3: 创建小数据集
prepare_small_dataset() {
    echo -e "\n${GREEN}[3/4] 准备训练数据...${NC}"
    echo "为了在CPU上合理的时间内完成训练，我们将使用较小的数据集"
    
    # 创建小数据集（减少到100条避免内存不足）
    head -n 100 data/processed/llm_input_enriched_train.jsonl > data/cpu_train_1k.jsonl
    head -n 20 data/processed/llm_input_enriched_val.jsonl > data/cpu_val_200.jsonl
    head -n 20 data/processed/llm_input_enriched_test.jsonl > data/cpu_test_100.jsonl
    
    echo -e "${GREEN}✓ 数据集准备完成${NC}"
    echo "  - 训练集: 100 条记录（内存优化）"
    echo "  - 验证集: 20 条记录"
    echo "  - 测试集: 20 条记录"
}

# 步骤4: 开始训练
start_training() {
    echo -e "\n${GREEN}[4/4] 开始CPU训练...${NC}"
    echo "配置信息："
    echo "  - 模型: TinyLlama-1.1B (小型模型，CPU友好)"
    echo "  - 训练样本: 100条（内存优化）"
    echo "  - 序列长度: 512 (大幅减少内存)"
    echo "  - Batch size: 1"
    echo "  - 训练轮数: 2"
    echo ""
    echo -e "${YELLOW}⚠️  CPU训练预计需要数小时，请耐心等待${NC}"
    echo ""
    
    mkdir -p lora-netflow-cpu
    
    # CPU优化训练（极度内存优化）
    python3 train_cpu_optimized.py \
        --train_path data/cpu_train_1k.jsonl \
        --val_path data/cpu_val_200.jsonl \
        --out_dir lora-netflow-cpu \
        --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --epochs 2 \
        --per_device_bs 1 \
        --grad_accum 2 \
        --lr 3e-4 \
        --max_len 512 \
        --r 4 \
        --alpha 8
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}=================================================="
        echo "🎉 训练完成！"
        echo "=================================================="
        echo "模型已保存到: lora-netflow-cpu/"
        echo ""
        echo "评估模型:"
        echo "python3 train_cpu_optimized.py \\"
        echo "    --eval_path data/cpu_test_100.jsonl \\"
        echo "    --adapter lora-netflow-cpu --mode eval"
        echo "=================================================="
        echo -e "${NC}"
    else
        echo -e "\n${RED}训练出错${NC}"
        exit 1
    fi
}

# 主流程
main() {
    cd /home/ubuntu/Workspace/DeepShield
    
    if [ "$1" == "--skip-install" ]; then
        echo "跳过安装，直接训练..."
        prepare_small_dataset
        start_training
    elif [ "$1" == "--quick-test" ]; then
        echo "快速测试模式（仅10条数据）..."
        head -n 10 data/processed/llm_input_enriched_train.jsonl > data/cpu_test_10.jsonl
        python3 train_cpu_optimized.py \
            --train_path data/cpu_test_10.jsonl \
            --val_path data/cpu_test_10.jsonl \
            --out_dir lora-cpu-test \
            --epochs 1 \
            --max_len 512
    else
        install_dependencies
        verify_installation
        prepare_small_dataset
        start_training
    fi
}

main "$@"
