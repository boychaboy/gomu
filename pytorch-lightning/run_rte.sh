# Install example requirements
#pip install -r requirements.txt

# Download glue data
#python3 ../../../utils/download_glue_data.py --tasks RTE,MNLI

export TASK=rte
export DATA_DIR=../glue_data/RTE
export MAX_LENGTH=128
export LEARNING_RATE=1e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=64
export NUM_EPOCHS=10
export SEED=1
export OUTPUT_DIR_NAME=rte-bert
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python3 run_glue.py --gpus 1 --data_dir $DATA_DIR \
--task $TASK \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train \
--do_predict
