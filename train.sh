# PYTHONUNBUFFERED=1 python3 -u train_lora_netflow_refined.py \
#     --train_path data/train_12h.jsonl \
#     --val_path  data/val_12h.jsonl \
#     --base_model meta-llama/Llama-3.1-8B-Instruct \
#     --out_dir lora-llama31-12h-incomplete-exp-short2 \
#     --epochs 5 --max_len 1536 --per_device_bs 20 --grad_accum 8 \
#     --load_in_4bit --bf16 --remove_eos_from_training | tee logs/train.log
direnv exec . torchrun --nproc_per_node=4 train_lora_netflow_refined.py \
    --train_path data/train_12h.jsonl \
    --val_path data/val_12h.jsonl \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --out_dir lora-qwen3-12h \
    --epochs 3 --max_len 2048 \
    --per_device_bs 2 --grad_accum 4 \
    --bf16 --dataloader_num_workers 20 \
    --remove_eos_from_training \
    --ip_mask_rate 0.8 \
    --eval_steps 100 --save_steps 100 --logging_steps 50
    # --resume_from_checkpoint lora-qwen3-12h/checkpoint-900 \
