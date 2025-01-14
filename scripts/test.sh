export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=0

accelerate launch --num_processes 1 inference.py