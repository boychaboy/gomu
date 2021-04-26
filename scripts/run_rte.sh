OUTPUT_DIR=$1
export TASK_NAME=rte

CUDA_VISIBLE_DEVICE=$2 python text-classification/run_glue.py \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir runs/$OUTPUT_DIR/
