#!/usr/bin/env bash
SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

LOG_DIR="/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1"
DATASET_DIR="/master/datasets/mscoco"
TEST_DATASET_DIR="/master/src/sparse_caption/test_data"
CACHE_FREE_RAM=0.3

export MPLCONFIGDIR="/tmp/matplotlib"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"


# `--id` is optional if `--log_dir` points to the experiment directory

python /master/src/sparse_caption/eval_model.py \
    --log_dir ${LOG_DIR}/RTrans__baseline \
    --beam_size_test 2 \
    --eval_dir_suffix TESTING \
    --model_file model_best.pth

python /master/src/sparse_caption/eval_model.py \
    --log_dir ${LOG_DIR} \
    --beam_size_test 2 \
    --eval_dir_suffix TESTING \
    --id RTrans__supermask__0.90__SCST_sample_baseline_s15_e10_C1B0001

python /master/src/sparse_caption/eval_model.py \
    --log_dir ${LOG_DIR} \
    --beam_size_test 2 \
    --eval_dir_suffix TESTING \
    --id RTrans__supermask__0.991__wg_120.0

python /master/src/sparse_caption/eval_model.py \
    --log_dir ${LOG_DIR} \
    --beam_size_test 2 \
    --eval_dir_suffix TESTING \
    --model_file model_best.pth \
    --id UpDownLSTM__baseline

python /master/src/sparse_caption/eval_model.py \
    --log_dir ${LOG_DIR} \
    --beam_size_test 2 \
    --eval_dir_suffix TESTING \
    --model_file model_best.pth \
    --id UpDownLSTM__baseline__SCST_sample_baseline_s60_e10_C1B0001

python /master/src/sparse_caption/eval_model.py \
    --log_dir ${LOG_DIR} \
    --beam_size_test 2 \
    --eval_dir_suffix TESTING \
    --id UpDownLSTM__supermask__0.991__wg_120.0



#######################
## TRAINING
#######################

# Pruning
PRUNE_TYPE="supermask"
PRUNE_SPARSITY_TARGET=0.9875
PRUNE_WEIGHT=120

# SCST
SCST_NUM_SAMPLES=10
SCST_SAMPLE="random"
SCST_BASELINE="sample"

#######################
## Up-Down LSTM
#######################

# Baseline dense
MODEL_TYPE="up_down_lstm"
MODEL_ID="UpDownLSTM"
SCHEDULER="cosine"


python /master/src/sparse_caption/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset mscoco_testing \
    --dataset_dir ${TEST_DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --learning_rate 0.01 \
    --optim_epsilon 0.01 \
    --batch_size 2 \
    --batch_size_eval 2 \
    --max_epochs 1 \
    --id TESTING_${MODEL_ID}__baseline \
    --save_checkpoint_every 10 \
    --cache_min_free_ram ${CACHE_FREE_RAM}

# Pruning
MODEL_TYPE="up_down_lstm_prune"

python /master/src/sparse_caption/train_n_prune_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset mscoco_testing \
    --dataset_dir ${TEST_DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --learning_rate 0.01 \
    --optim_epsilon 0.01 \
    --drop_prob_lm 0.1 \
    --batch_size 2 \
    --batch_size_eval 2 \
    --max_epochs 1 \
    --max_seq_length 10 \
    --prune_type ${PRUNE_TYPE} \
    --prune_sparsity_target ${PRUNE_SPARSITY_TARGET} \
    --prune_supermask_sparsity_weight ${PRUNE_WEIGHT} \
    --id TESTING_${MODEL_ID}__${PRUNE_TYPE}__${PRUNE_SPARSITY_TARGET} \
    --save_checkpoint_every 10 \
    --cache_min_free_ram ${CACHE_FREE_RAM}


# Fine-tune with mask frozen
BASELINE="${LOG_DIR}/${MODEL_ID}__supermask__${PRUNE_SPARSITY_TARGET}__wg_120.0/model_best_bin_mask.pth"

python /master/src/sparse_caption/train_n_prune_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --start_from ${BASELINE} \
    --lr_scheduler step \
    --learning_rate 5e-5 \
    --learning_rate_decay_start -1 \
    --batch_size 2 \
    --batch_size_eval 2 \
    --max_epochs 1 \
    --drop_prob_lm 0.1 \
    --prune_type mask_freeze \
    --prune_sparsity_target ${PRUNE_SPARSITY_TARGET} \
    --scst_start_epoch 0 \
    --scst_num_samples ${SCST_NUM_SAMPLES} \
    --scst_sample ${SCST_SAMPLE} \
    --scst_baseline ${SCST_BASELINE} \
    --scst_bleu_weight 0,0,0,1 \
    --id TESTING_${MODEL_ID}__supermask__${PRUNE_SPARSITY_TARGET}__SCST_${SCST_SAMPLE}_${SCST_BASELINE}_s${SCST_NUM_SAMPLES}_e${EPOCHS}_C1B0001 \
    --save_checkpoint_every 10 \
    --cache_min_free_ram ${CACHE_FREE_RAM}


######################
# Relation Transformer
######################

# Baseline dense
MODEL_TYPE="relation_transformer"
MODEL_ID="RTrans"
SCHEDULER="noam"

python /master/src/sparse_caption/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset mscoco_testing \
    --dataset_dir ${TEST_DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --batch_size 2 \
    --batch_size_eval 2 \
    --max_epochs 1 \
    --id TESTING_${MODEL_ID}__baseline \
    --save_checkpoint_every 10 \
    --cache_min_free_ram ${CACHE_FREE_RAM}

# Fine-tune
BASELINE="${LOG_DIR}/${MODEL_ID}__baseline/model_best.pth"

python /master/src/sparse_caption/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --start_from ${BASELINE} \
    --lr_scheduler step \
    --learning_rate 5e-5 \
    --learning_rate_decay_start -1 \
    --batch_size 2 \
    --batch_size_eval 2 \
    --max_epochs 1 \
    --drop_prob_src 0.1 \
    --scst_start_epoch 0 \
    --scst_num_samples ${SCST_NUM_SAMPLES} \
    --scst_sample ${SCST_SAMPLE} \
    --scst_baseline ${SCST_BASELINE} \
    --scst_bleu_weight 0,0,0,1 \
    --id TESTING_${MODEL_ID}__baseline__SCST_${SCST_SAMPLE}_${SCST_BASELINE}_s${SCST_NUM_SAMPLES}_e${EPOCHS}_C1B0001 \
    --save_checkpoint_every 10 \
    --cache_min_free_ram ${CACHE_FREE_RAM}


# Pruning
MODEL_TYPE="relation_transformer_prune"

python /master/src/sparse_caption/train_n_prune_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset mscoco_testing \
    --dataset_dir ${TEST_DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --drop_prob_src 0.1 \
    --batch_size 2 \
    --batch_size_eval 2 \
    --max_epochs 1 \
    --prune_type ${PRUNE_TYPE} \
    --prune_sparsity_target ${PRUNE_SPARSITY_TARGET} \
    --prune_supermask_sparsity_weight ${PRUNE_WEIGHT} \
    --id TESTING_${MODEL_ID}__${PRUNE_TYPE}__${PRUNE_SPARSITY_TARGET} \
    --save_checkpoint_every 10 \
    --cache_min_free_ram ${CACHE_FREE_RAM}

