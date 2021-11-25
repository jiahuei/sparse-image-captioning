# Learning to Prune Image Captioning Models

![Tests](https://github.com/jiahuei/sparse-image-captioning/actions/workflows/tests.yml/badge.svg)
![Black](https://github.com/jiahuei/sparse-image-captioning/actions/workflows/black.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/sparse-image-captioning/badge/?version=latest)](https://sparse-image-captioning.readthedocs.io/en/latest/?badge=latest)


[PDF](https://www.sciencedirect.com/science/article/pii/S003132032100546X) | [ArXiv](https://arxiv.org/abs/2110.03298)

### Official pytorch implementation of the paper: "End-to-End Supermask Pruning: Learning to Prune Image Captioning Models"

#### Published at Pattern Recognition, Elsevier

Released on July 20, 2021


## Description

This work explores model pruning for image captioning task at the first time. Empirically, we show that 80% to 95% sparse networks can either match or even slightly outperform their dense counterparts. In order to promote Green Computer Vision, we release the pre-trained sparse models for UD and ORT that are capable of achieving CIDEr scores >120 on MS-COCO dataset; yet are only 8.7 MB (reduction of 96% compared to dense UD) and 14.5 MB (reduction of 94% compared to dense ORT) in model size.

<p align="center"> <img src="resources/pr2021.jpg" width="35%"> </p>
<p align="center"> Figure 1: We show that our deep captioning networks with 80% to 95% sparse are capable to either match or even slightly outperform their dense counterparts.</p>


## Get Started

[Please refer to the documentation](https://sparse-image-captioning.readthedocs.io/en/latest/).


## Features

* Captioning models built using PyTorch
    * [Up-Down LSTM](http://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.html)
    * [Object Relation Transformer](https://papers.nips.cc/paper/9293-image-captioning-transforming-objects-into-words.pdf)
    * A Compact Object Relation Transformer (ACORT)
* Unstructured weight pruning
    * [Supermask Pruning (SMP, end-to-end pruning)](https://arxiv.org/abs/2110.03298)
    * [Gradual magnitude pruning](https://arxiv.org/abs/1710.01878)
    * [Lottery ticket](https://arxiv.org/abs/1803.03635)
    * One-shot magnitude pruning ([paper 1](https://arxiv.org/abs/1506.02626), [paper 2](https://arxiv.org/abs/1606.09274))
    * [Single-shot Network Pruning (SNIP)](https://arxiv.org/abs/1810.02340)
* Self-Critical Sequence Training (SCST)]()
    * Random sampling + Greedy search baseline: [vanilla SCST](https://openaccess.thecvf.com/content_cvpr_2017/html/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.html)
    * Beam search sampling + Greedy search baseline: à la [Up-Down](http://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.html)
    * Random sampling + Sample mean baseline: [arxiv paper](https://arxiv.org/abs/2003.09971)
    * Beam search sampling + Sample mean baseline: à la [M2 Transformer](http://openaccess.thecvf.com/content_CVPR_2020/html/Cornia_Meshed-Memory_Transformer_for_Image_Captioning_CVPR_2020_paper.html)
    * Optimise CIDEr and/or BLEU scores with custom weightage
    * Based on [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch/tree/3.2)
* Multiple captions per image during teacher-forcing training
    * Reduce training time: run encoder once, optimize on multiple training captions
* Incremental decoding (Transformer with attention cache)
* Data caching during training
    * Training examples will be cached in memory to reduce disk I/O
    * With sufficient memory, the entire training set can be loaded from memory after the first epoch
    * Memory usage can be controlled via `cache_min_free_ram` flag
* `coco_caption` in Python 3
    * Based on [salaniz/pycocoevalcap](https://github.com/salaniz/pycocoevalcap/tree/ad63453cfab57a81a02b2949b17a91fab1c3df77)
* Tokenizer based on `sentencepiece`
    * Word
    * [Radix encoding](https://github.com/jiahuei/COMIC-Compact-Image-Captioning-with-Attention)
    * _(untested)_ Unigram, BPE, Character
* Datasets
    * MS-COCO
    * _(contributions welcome)_ Flickr8k, Flickr30k, InstaPIC-1.1M


## Pre-trained Sparse and ACORT Models

The checkpoints are [available at this repo](https://github.com/jiahuei/sparse-captioning-checkpoints).

Soft-attention models implemented in TensorFlow 1.9 are available at [this repo](https://github.com/jiahuei/tf-sparse-captioning).


## CIDEr score of pruning methods

### Up-Down (UD)

| Sparsity | NNZ | Dense Baseline | SMP | Lottery ticket (class-blind) | Lottery ticket (class-uniform) | Lottery ticket (gradual) | Gradual pruning | Hard pruning (class-blind) | Hard pruning (class-distribution) | Hard pruning (class-uniform) | SNIP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.950 | 2.7 M | 111.3 | **112.5** | - | 107.7 | 109.5 | 109.7 | - | 110.0 | 110.2 | 38.2 |
| 0.975 | 1.3 M | 111.3 | **110.6** | - | 103.8 | 106.6 | 107.0 | - | 105.9 | 105.4 | 34.7 |
| 0.988 | 0.7 M | 111.3 | **109.0** | - | 99.3 | 102.2 | 103.4 | - | 101.3 | 100.5 | 32.6 |
| 0.991 | 0.5 M | 111.3 | **107.8** |  |  |  |  |  |  |  |  |

### Object Relation Transformer (ORT)

| Sparsity | NNZ | Dense Baseline | SMP | Lottery ticket (gradual) | Gradual pruning | Hard pruning (class-blind) | Hard pruning (class-distribution) | Hard pruning (class-uniform) | SNIP |
|---|---|---|---|---|---|---|---|---|---|
| 0.950 | 2.8 M | 114.7 | 113.7 | **115.7** | 115.3 | 4.1 | 112.5 | 113.0 | 47.2 |
| 0.975 | 1.4 M | 114.7 | **113.7** | 112.9 | 113.2 | 0.7 | 106.6 | 106.9 | 44.0 |
| 0.988 | 0.7 M | 114.7 | **110.7** | 109.8 | 110.0 | 0.9 | 96.9 | 59.8 | 37.3 |
| 0.991 | 0.5 M | 114.7 | **109.3** | 107.1 | 107.0 |  |  |  |  |


## Acknowledgements

* SCST, Up-Down: [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch/tree/3.2)
* Object Relation Transformer: [yahoo/object_relation_transformer](https://github.com/yahoo/object_relation_transformer)
* `coco_caption` in Python 3: [salaniz/pycocoevalcap](https://github.com/salaniz/pycocoevalcap/tree/ad63453cfab57a81a02b2949b17a91fab1c3df77)


## Citation

If you find this work useful for your research, please cite
```
@article{tan2021end,
  title={End-to-End Supermask Pruning: Learning to Prune Image Captioning Models},
  author={Tan, Jia Huei and Chan, Chee Seng and Chuah, Joon Huang},
  journal={Pattern Recognition},
  pages={108366},
  year={2021},
  publisher={Elsevier},
  doi={10.1016/j.patcog.2021.108366}
}
```

## Feedback
Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the authors by sending an email to
`tan.jia.huei at gmail.com` or `cs.chan at um.edu.my`.

## License and Copyright
The project is open source under BSD-3 license (see the ``` LICENSE ``` file).

&#169;2021 Universiti Malaya.


## Dev Info

Run Black linting:
```bash
black --line-length=120 --safe sparse_caption
black --line-length=120 --safe tests
black --line-length=120 --safe scripts
```
