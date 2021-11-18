# -*- coding: utf-8 -*-
"""
Created on 31 Mar 2020 23:02:57

@author: jiahuei
"""

import os
from pycocotools.coco import COCO
from .pycocoevalcap.eval import COCOEvalCap

COCO_DIR = os.path.dirname(os.path.realpath(__file__))


def evaluate_caption_json(res_file, ann_file):
    """
    Given file paths to the caption prediction and annotation JSON files,
    will compute captioning metric scores as a `dict`.

    The file paths can be absolute or relative. If relative, the files are assumed to be in
    `coco_caption/results` and `coco_caption/annotations` respectively.

    Args:
        res_file: File path to the caption prediction JSON file.
        ann_file: File path to the caption annotation JSON file.
    Returns:
        A `dict` with scores, with the following structure:
            {
            'Bleu_1': x.x,
            'Bleu_2': x.x,
            'Bleu_3': x.x,
            'Bleu_4': x.x,
            'METEOR': x.x,
            'ROUGE_L': x.x,
            'CIDEr': x.x,
            'SPICE': x.x,
            }

        A list of scores for each image:
            [
                {
                'image_id': 2090545563,
                'Bleu_1': x.x,
                'Bleu_2': x.x,
                'Bleu_3': x.x,
                'Bleu_4': x.x,
                'METEOR': x.x,
                'ROUGE_L': 0.0,
                'CIDEr': 0.0,
                'SPICE':
                    {
                    'All': {'pr': 0.0, 're': 0.0, 'f': 0.0, 'fn': 24.0, 'numImages': 1.0, 'fp': 7.0, 'tp': 0.0},
                    'Relation': {'pr': 0.0, 're': 0.0, 'f': 0.0, 'fn': 7.0, 'numImages': 1.0, 'fp': 0.0, 'tp': 0.0},
                    'Cardinality': {'pr': 0.0, 're': 0.0, 'f': 0.0, 'fn': 3.0, 'numImages': 1.0, 'fp': 0.0, 'tp': 0.0},
                    'Attribute': {'pr': 0.0, 're': 0.0, 'f': 0.0, 'fn': 7.0, 'numImages': 1.0, 'fp': 5.0, 'tp': 0.0},
                    'Size': {'pr': 0.0, 're': 0.0, 'f': 0.0, 'fn': 1.0, 'numImages': 1.0, 'fp': 0.0, 'tp': 0.0},
                    'Color': {'pr': 0.0, 're': 0.0, 'f': 0.0, 'fn': 1.0, 'numImages': 1.0, 'fp': 0.0, 'tp': 0.0},
                    'Object': {'pr': 0.0, 're': 0.0, 'f': 0.0, 'fn': 10.0, 'numImages': 1.0, 'fp': 2.0, 'tp': 0.0}
                    }
                },
                ...
            ]
        COCO Eval object
    """
    assert ann_file.endswith(".json"), "`ann_file` should end with `.json`, saw `{}` instead.".format(ann_file)
    assert res_file.endswith(".json"), "`res_file` should end with `.json`, saw `{}` instead.".format(res_file)
    default_ann_dir = os.path.join(COCO_DIR, "annotations")
    default_res_dir = os.path.join(COCO_DIR, "results")
    # create coco object and cocoRes object
    coco = COCO(os.path.join(default_ann_dir, ann_file))
    coco_res = coco.loadRes(os.path.join(default_res_dir, res_file))

    # create cocoEval object by taking coco and coco_res
    coco_eval = COCOEvalCap(coco, coco_res)

    # evaluate on a subset of images
    coco_eval.params["image_id"] = coco_res.getImgIds()

    # evaluate results
    coco_eval.evaluate()

    results = {}
    for metric, score in coco_eval.eval.items():
        # print '%s: %.3f' % (metric, score)
        results[metric] = score
    return results, coco_eval.evalImgs, coco_eval
