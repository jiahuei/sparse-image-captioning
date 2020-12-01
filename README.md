# Image Captioning with Transformer

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


