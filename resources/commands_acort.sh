#!/usr/bin/env bash
SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

LOG_DIR="/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1"
DATASET_DIR="/master/datasets/mscoco"
CACHE_FREE_RAM=0.3

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"


### Collect scores ###
python src/sparse_caption/scripts/collect_scores.py --check_compiled_scores


### Eval ###
#    --eval_dir_suffix  \
#    --load_as_float16  \
#    --mscoco_online_test  \
#    --beam_size_val 5 \
python /master/src/sparse_caption/eval_model.py \
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
python /master/src/sparse_caption/train_transformer.py \
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
python /master/src/sparse_caption/train_transformer.py \
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
python /master/src/sparse_caption/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --id ${MODEL_ID}__base \
    --cache_min_free_ram ${CACHE_FREE_RAM}

# -small
python /master/src/sparse_caption/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --d_model 256 \
    --dim_feedforward 1024 \
    --id ${MODEL_ID}__small \
    --cache_min_free_ram ${CACHE_FREE_RAM}

# -xsmall
python /master/src/sparse_caption/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --d_model 104 \
    --dim_feedforward 416 \
    --id ${MODEL_ID}__xsmall \
    --cache_min_free_ram ${CACHE_FREE_RAM}


######################
# SCST
######################

MODEL_ID="ACORT__small"
MODEL_TYPE="relation_transformer"
BASELINE="${LOG_DIR}/${MODEL_ID}/model_best.pth"
EPOCHS=10
SCST_NUM_SAMPLES=15
SCST_SAMPLE="random"
SCST_BASELINE="sample"

python /master/src/sparse_caption/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --start_from ${BASELINE} \
    --lr_scheduler step \
    --learning_rate 5e-5 \
    --learning_rate_decay_start -1 \
    --batch_size 5 \
    --max_epochs ${EPOCHS} \
    --drop_prob_src 0.1 \
    --losses_log_every 75 \
    --scst_start_epoch 0 \
    --scst_num_samples ${SCST_NUM_SAMPLES} \
    --scst_sample ${SCST_SAMPLE} \
    --scst_baseline ${SCST_BASELINE} \
    --scst_bleu_weight 0,0,0,1 \
    --tokenizer radix \
    --radix_base 768 \
    --max_seq_length 26 \
    --share_att_encoder kv \
    --share_att_decoder kv \
    --share_layer_encoder "(0, 0, 0, 1, 1, 1)" \
    --share_layer_decoder "(0, 0, 0, 1, 1, 1)" \
    --d_model 256 \
    --dim_feedforward 1024 \
    --id ${MODEL_ID}__SCST_${SCST_SAMPLE}_${SCST_BASELINE}_s${SCST_NUM_SAMPLES}_e${EPOCHS}_C1B0001 \
    --cache_min_free_ram ${CACHE_FREE_RAM}


######################
# Speed Tests
######################

MODEL_TYPE="relation_transformer"
MODEL_ID="ACORT"
SCHEDULER="noam"

for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/eval_model.py \
        --log_dir ${LOG_DIR} \
        --batch_size_eval 1 \
        --beam_size_test 2 \
        --model_file model_best.pth \
        --eval_dir_suffix "_b1_speedtest_run${x}" \
        --id Radix_b768_RTrans__baseline__ls_2layer_both__tied_kv_both
    sleep 5m
done

for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/eval_model.py \
        --log_dir ${LOG_DIR} \
        --batch_size_eval 1 \
        --beam_size_test 2 \
        --model_file model_best.pth \
        --eval_dir_suffix "_b1_speedtest_run${x}" \
        --id Radix_b768_RTrans__baseline__ls_2layer_both_r1__tied_kv_both
    sleep 5m
done

for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/eval_model.py \
        --log_dir ${LOG_DIR} \
        --batch_size_eval 1 \
        --beam_size_test 2 \
        --model_file model_best.pth \
        --eval_dir_suffix "_b1_speedtest_run${x}" \
        --id Radix_b768_RTrans__baseline__ls_2layer_both__tied_kv_both_4M
    sleep 5m
done

for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/eval_model.py \
        --log_dir /home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1/ \
        --batch_size_eval 1 \
        --beam_size_test 2 \
        --model_file model_best.pth \
        --eval_dir_suffix "_b1_speedtest_run${x}" \
        --id RTrans__baseline
    sleep 5m
done

for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/eval_model.py \
        --log_dir ${LOG_DIR} \
        --batch_size_eval 1 \
        --beam_size_test 2 \
        --model_file model_best.pth \
        --eval_dir_suffix "_b1_speedtest_run${x}" \
        --id RTrans__baseline__slim_17M
    sleep 5m
done

for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/eval_model.py \
        --log_dir ${LOG_DIR} \
        --batch_size_eval 1 \
        --beam_size_test 2 \
        --model_file model_best.pth \
        --eval_dir_suffix "_b1_speedtest_run${x}" \
        --id RTrans__baseline__slim_4M
    sleep 5m
done


MODEL_TYPE="relation_transformer"
MODEL_ID="ACORT"
SCHEDULER="noam"

# -base
for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/train_transformer.py \
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
        --id ${MODEL_ID}__base__speedtest_run${x} \
        --cache_min_free_ram ${CACHE_FREE_RAM}
    sleep 5m
done

# -base-AL
for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/train_transformer.py \
        --caption_model ${MODEL_TYPE} \
        --dataset_dir ${DATASET_DIR} \
        --log_dir ${LOG_DIR} \
        --lr_scheduler ${SCHEDULER} \
        --tokenizer radix \
        --radix_base 768 \
        --max_seq_length 26 \
        --share_att_encoder kv \
        --share_att_decoder kv \
        --share_layer_encoder "(0, 0, 0, 0, 0, 0)" \
        --share_layer_decoder "(0, 0, 0, 0, 0, 0)" \
        --d_model 512 \
        --dim_feedforward 2048 \
        --id ${MODEL_ID}__base-AL__speedtest_run${x} \
        --cache_min_free_ram ${CACHE_FREE_RAM}
    sleep 5m
done

# -small
for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/train_transformer.py \
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
        --id ${MODEL_ID}__small__speedtest_run${x} \
        --cache_min_free_ram ${CACHE_FREE_RAM}
    sleep 5m
done


MODEL_TYPE="relation_transformer"
MODEL_ID="ORT"
SCHEDULER="noam"

# -base
for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/train_transformer.py \
        --caption_model ${MODEL_TYPE} \
        --dataset_dir ${DATASET_DIR} \
        --log_dir ${LOG_DIR} \
        --lr_scheduler ${SCHEDULER} \
        --id ${MODEL_ID}__base__speedtest_run${x} \
        --cache_min_free_ram ${CACHE_FREE_RAM}
    sleep 5m
done

# -small
for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/train_transformer.py \
        --caption_model ${MODEL_TYPE} \
        --dataset_dir ${DATASET_DIR} \
        --log_dir ${LOG_DIR} \
        --lr_scheduler ${SCHEDULER} \
        --d_model 256 \
        --dim_feedforward 1024 \
        --id ${MODEL_ID}__small__speedtest_run${x} \
        --cache_min_free_ram ${CACHE_FREE_RAM}
    sleep 5m
done


# -xsmall
for x in 1 2 3 4 5; do
    python /master/src/sparse_caption/train_transformer.py \
        --caption_model ${MODEL_TYPE} \
        --dataset_dir ${DATASET_DIR} \
        --log_dir ${LOG_DIR} \
        --lr_scheduler ${SCHEDULER} \
        --d_model 104 \
        --dim_feedforward 416 \
        --id ${MODEL_ID}__xsmall__speedtest_run${x} \
        --cache_min_free_ram ${CACHE_FREE_RAM}
    sleep 5m
done

