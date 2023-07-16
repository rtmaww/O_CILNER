LOSS_NAME1="supcon_ce"
LOSS_NAME2="supcon_o_bce"
CLS_NAME="ncm_dot"
DATA_DIR=../data/tasks/
LABEL_DIR=../data/labels.txt
NUM_TRAIN_EPOCHS=16
START_STEP=0
NB_TASK=11
PER_TYPES=6
START_TRAIN_O_EPOCH=10
OUTPUT_DIR=../output_nerd
LOG_DIR=../log/nerd/results.txt

CUDA_VISIBLE_DEVICES=0 python3 -u run_incremental_proto.py --data_dir $DATA_DIR \
--model_type bert \
--model_type_create bert_create \
--model_type_eval bert_eval \
--labels $LABEL_DIR \
--model_name_or_path $OUTPUT_DIR \
--do_lower_case \
--output_dir $OUTPUT_DIR \
--log_dir $LOG_DIR \
--overwrite_output_dir \
--max_seq_length  128 \
--evaluate_during_training \
--logging_steps 4000 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--save_steps 4000 \
--eval_all_checkpoints \
--num_train_epochs $NUM_TRAIN_EPOCHS \
--seed 1 \
--start_step $START_STEP \
--nb_tasks $NB_TASK \
--per_types $PER_TYPES \
--cls_name $CLS_NAME \
--feat_dim 128 \
--start_train_o_epoch $START_TRAIN_O_EPOCH \
--relabel_th 0.98 \
--relabels_th_reduction 0.05 \
--loss_name1 $LOSS_NAME1 \
--loss_name2 $LOSS_NAME2 \
--change_th \
--do_train \
--do_predict