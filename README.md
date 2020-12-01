# Sparse Image Captioning with Transformer

## Features

* Captioning models built using PyTorch
    * Up-Down LSTM
    * Object Relation Transformer
* Unstructured weight pruning
    * Supermask (end-to-end pruning)
    * Gradual magnitude pruning
    * Lottery ticket
    * One-shot magnitude pruning
    * Single-shot Network Pruning (SNIP)
* Data caching during training
    * Training examples will be cached in memory to reduce disk I/O
    * With sufficient memory, the entire training set can be loaded from memory after the first epoch
    * Memory usage can be controlled via `cache_min_free_ram` flag
* `coco_caption` in Python 3
    * Based on [salaniz/pycocoevalcap](https://github.com/salaniz/pycocoevalcap/tree/ad63453cfab57a81a02b2949b17a91fab1c3df77)
* Multiple captions per image during teacher-forcing training
    * Reduce training time: run encoder once, optimize on multiple training captions
* Incremental decoding (Transformer with attention cache)
* Self-Critical Sequence Training (SCST)
    * Sampling: Random sample (>= 1 samples per image) or beam search _(untested)_
    * Baseline: Greedy search or random sample
    * Based on [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch/tree/3.2)
* Tokenizer based on `sentencepiece`
    * Word
    * _(untested)_ Unigram, BPE, Character
* Datasets
    * MS-COCO
    * _(to be added)_ Flickr8k, Flickr30k, InstaPIC-1.1M


## Main Requirements

* python >= 3.6
* pytorch >= 1.6
* sentencepiece >= 0.1.91
* torchvision >= 0.7.0

See the rest in `./docker/requirements.txt`


## Setup

For convenience, setup is done using Docker.

1. Run `bash docker/docker_build.sh` to build the Docker image.
2. Run `bash docker/docker_run_clean.sh` to launch a container. Edit paths in the script as needed.
3. Run `python caption_vae/scripts/setup.sh` to perform dataset pre-processing.
4. Done


## Usage

Refer to `commands.sh` for more examples.

### Training
```shell script
# Baseline dense
MODEL_TYPE="up_down_lstm"
MODEL_ID="UpDownLSTM"
SCHEDULER="cosine"

python /master/src/caption_vae/train_transformer.py \
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
        python /master/src/caption_vae/train_n_prune_transformer.py \
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

### Inference
To evaluate the models, simply run:
```shell script
python /master/src/caption_vae/eval_model.py \
    --log_dir ${LOG_DIR} \
    --beam_size_test 2 \
    --id RTrans__supermask__0.9875__wg_80.0
```


## Pre-trained Sparse Models

The checkpoints are [available at this repo](https://drive.google.com/drive/folders/1PN-oBoHLjdAkY9k1GePbkmQNMn9uR3e-?usp=sharing).


## MS-COCO Online Evaluation

To perform online server evaluation:
1. Run `python /master/src/caption_vae/eval_model.py` with `--mscoco_online_test` option.
    For example:
    ```shell script
    python /master/src/caption_vae/eval_model.py \
        --log_dir ${LOG_DIR} \
        --beam_size_test 5 \
        --mscoco_online_test \
        --id ${ID}
    ```
2. Rename the JSON files to `captions_test2014__results.json` and `captions_val2014__results.json`.
    * `captions_val2014__results.json` will contain fake captions, just there to fulfill submission format
3. Zip the files and [submit](https://competitions.codalab.org/competitions/3221#participate).


