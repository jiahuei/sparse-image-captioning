# -*- coding: utf-8 -*-
"""
Created on 06 Jan 2021 16:57:17
@author: jiahuei

python -m unittest coco_caption/test_coco_caption.py
"""
import unittest
import os
from sparse_caption.coco_caption.eval import evaluate_caption_json
from sparse_caption.data.mscoco import MscocoDataset
from .paths import TEST_DATA_DIRPATH


class TestCocoCaption(unittest.TestCase):
    METRICS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]

    def test_mscoco_score(self):
        scores, scores_detailed, coco_eval = evaluate_caption_json(
            res_file=os.path.join(TEST_DATA_DIRPATH, "caption_00156000.json"), ann_file=MscocoDataset.ANNOTATION_FILE
        )
        scores = [round(scores[_], 3) for _ in self.METRICS]
        self.assertEqual(
            scores, [0.806, 0.655, 0.514, 0.398, 0.288, 0.584, 1.311, 0.220], "Scores are different from expected."
        )


if __name__ == "__main__":
    unittest.main()
