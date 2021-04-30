CUDA_VISIBLE_DEVICE=$1 python metric.py \
    --input_file data/crows_pairs_anonymized.csv \
    --lm_model bert-large \
    --output_file bert-large-cased
