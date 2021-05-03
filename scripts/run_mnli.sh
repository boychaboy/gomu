OUTPUT_DIR=$1
export TASK_NAME=mnli

CUDA_VISIBLE_DEVICE=$2 python text-classification/run_glue.py \
  --model_name_or_path bert-large-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir runs/$OUTPUT_DIR/
