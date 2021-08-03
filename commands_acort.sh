#!/usr/bin/env bash
SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

LOG_DIR="/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1"
DATASET_DIR="/master/datasets/mscoco"
CACHE_FREE_RAM=0.3

export STANZA_CACHE_DIR="${DATASET_DIR}/stanza_resources"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"


### Collect scores ###
python src/caption_vae/scripts/collect_scores.py --check_compiled_scores


### Eval ###
#    --eval_dir_suffix  \
#    --load_as_float16  \
#    --mscoco_online_test  \
python /master/src/caption_vae/eval_model.py \
    --log_dir ${LOG_DIR} \
    --beam_size_test 2 \
    --model_file model_best.pth \
    --id ACORT__base


######################
# ACORT
######################

MODEL_TYPE="relation_transformer"
MODEL_ID="ACORT"
SCHEDULER="noam"

# -base
python /master/src/caption_vae/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --tokenizer radix \
    --radix_base 768 \
    --max_seq_length 26 \
    --share_att_encoder kv \
    --share_att_decoder kv \
    --share_layer_encoder "(0, 0, 0, 1, 1, 1)" \
    --share_layer_decoder "(0, 0, 0, 1, 1, 1)" \
    --d_model 512 \
    --dim_feedforward 2048 \
    --id ${MODEL_ID}__base \
    --cache_min_free_ram ${CACHE_FREE_RAM}

# -small
python /master/src/caption_vae/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --tokenizer radix \
    --radix_base 768 \
    --max_seq_length 26 \
    --share_att_encoder kv \
    --share_att_decoder kv \
    --share_layer_encoder "(0, 0, 0, 1, 1, 1)" \
    --share_layer_decoder "(0, 0, 0, 1, 1, 1)" \
    --d_model 256 \
    --dim_feedforward 1024 \
    --id ${MODEL_ID}__small \
    --cache_min_free_ram ${CACHE_FREE_RAM}



######################
# Relation Transformer (ORT)
######################

MODEL_TYPE="relation_transformer"
MODEL_ID="ORT"
SCHEDULER="noam"

# -base
python /master/src/caption_vae/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --id ${MODEL_ID}__base \
    --cache_min_free_ram ${CACHE_FREE_RAM}

# -small
python /master/src/caption_vae/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --d_model 256 \
    --dim_feedforward 1024 \
    --id ${MODEL_ID}__small \
    --cache_min_free_ram ${CACHE_FREE_RAM}

# -xsmall
python /master/src/caption_vae/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --d_model 104 \
    --dim_feedforward 416 \
    --id ${MODEL_ID}__xsmall \
    --cache_min_free_ram ${CACHE_FREE_RAM}

