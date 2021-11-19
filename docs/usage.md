# Usage

Refer to `resources/commands_pruning.sh` and `resources/commands_acort.sh` for more examples.

These paths below assume a Docker setup following [Get Started](get_started.md).


## Training

```bash
# Baseline dense
MODEL_TYPE="up_down_lstm"
MODEL_ID="UpDownLSTM"
SCHEDULER="cosine"

python /workspace/scripts/train_transformer.py \
    --caption_model ${MODEL_TYPE} \
    --dataset_dir ${DATASET_DIR} \
    --log_dir ${LOG_DIR} \
    --lr_scheduler ${SCHEDULER} \
    --learning_rate 0.01 \
    --optim_epsilon 0.01 \
    --id ${MODEL_ID}__baseline \
    --cache_min_free_ram ${CACHE_FREE_RAM}

# Pruning
MODEL_TYPE="up_down_lstm_prune"

for PRUNE_SPARSITY_TARGET in 0.9875 0.975 0.95; do
    for PRUNE_WEIGHT in 40 80 120; do
        python /workspace/scripts/train_n_prune_transformer.py \
            --caption_model ${MODEL_TYPE} \
            --dataset_dir ${DATASET_DIR} \
            --log_dir ${LOG_DIR} \
            --lr_scheduler ${SCHEDULER} \
            --learning_rate 0.01 \
            --optim_epsilon 0.01 \
            --drop_prob_lm 0.1 \
            --prune_type supermask \
            --prune_sparsity_target ${PRUNE_SPARSITY_TARGET} \
            --prune_supermask_sparsity_weight ${PRUNE_WEIGHT} \
            --id ${MODEL_ID}__${PRUNE_TYPE}__${PRUNE_SPARSITY_TARGET} \
            --cache_min_free_ram ${CACHE_FREE_RAM}
    done
done
```


## Inference
To evaluate the models, simply run:
```shell script
python /workspace/scripts/eval_model.py \
    --log_dir ${LOG_DIR} \
    --beam_size_test 2 \
    --id RTrans__supermask__0.9875__wg_80.0
```


## MS-COCO Online Evaluation

To perform online server evaluation:
1. Run `python /workspace/scripts/eval_model.py` with `--mscoco_online_test` option.
    For example:
    ```shell script
    python /workspace/scripts/eval_model.py \
        --log_dir ${LOG_DIR} \
        --beam_size_test 5 \
        --mscoco_online_test \
        --id ${ID}
    ```
2. Rename the JSON files to `captions_test2014__results.json` and `captions_val2014__results.json`.
    * `captions_val2014__results.json` will contain fake captions, just there to fulfil submission format
3. Zip the files and [submit](https://competitions.codalab.org/competitions/3221#participate).


## Visualisation

You can explore and visualise generated captions [using this Streamlit app](https://github.com/jiahuei/MSCOCO-caption-explorer).

