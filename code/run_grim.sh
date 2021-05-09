GPU_ID=$1
CUDA_VISIBLE_DEVICES=$GPU_ID python grim.py \
    --split \
    --seed 1 \
    --split_ratio 0.001 \
    --model_name boychaboy/mnli_roberta-base
