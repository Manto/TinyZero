#!/bin/bash
# alias python='/home/weiji/anaconda3/envs/zero/bin/python'
# alias python3='python3'
# alias pip='/home/weiji/anaconda3/envs/zero/bin/pip'

export N_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
ray stop --force && ray start --head --dashboard-host=0.0.0.0 --include-dashboard=True
export BASE_MODEL="Qwen/Qwen2.5-3B"
export DATA_DIR="data/countdown"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero_a100_ppo.sh