export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=0

accelerate launch --num_processes 1 train.py --stage1_ckpt "./output/generator_best.pth"