# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:57:43 2019

@author: jiahuei
"""
import logging
import numpy as np
from typing import Union, Tuple, List
from ..coco_caption.pycocoevalcap.bleu.bleu import Bleu
from .cider.pyciderevalcap.ciderD.ciderD import CiderD


logger = logging.getLogger(__name__)


class CaptionScorer(object):
    """
    An object that encapsulates the different scorers to provide a unified
    interface.
    """

    def __init__(
            self, path_to_cached_tokens: str,
            cider_weight: float = 1.0,
            bleu_weight: Union[List, Tuple] = None
    ):
        assert isinstance(cider_weight, float)
        if bleu_weight is None:
            bleu_weight = [0.] * 4
        else:
            assert isinstance(bleu_weight, (list, tuple))
        assert len(bleu_weight) == 4
        self.path_to_cached_tokens = path_to_cached_tokens
        self.scorers = None
        self.weights = {
            "ciderD": cider_weight,
            "bleu": bleu_weight,
        }

    @staticmethod
    def input_check(inputs, same_sub_len=True):
        assert isinstance(inputs, (list, tuple))
        assert all(isinstance(_, (list, tuple)) for _ in inputs)
        if same_sub_len:
            lens = set(len(_) for _ in inputs)
            assert len(lens) == 1, (
                f"Each image should have the same number of captions."
                f"Received captions per image: {lens}"
            )

    def __call__(self, refs, sample, greedy=None):
        if self.scorers is None:
            self.scorers = {
                "ciderD": CiderD(df=self.path_to_cached_tokens),
                "bleu": BleuSilent(4),
            }
        self.input_check(refs, same_sub_len=False)
        self.input_check(sample)
        assert len(refs) == len(sample), (
            f"`ref` and `sample` have different lengths: "
            f"refs = {len(refs)}, sample = {len(sample)}"
        )
        if greedy:
            self.input_check(greedy)
            assert len(sample) == len(greedy), (
                f"`sample` and `greedy` have different lengths: "
                f"sample = {len(sample)}, greedy = {len(greedy)}"
            )
        else:
            assert greedy is None, "`greedy` should be one of: None, list or tuple."

        weights = self.weights
        num_greedy = len(greedy) if greedy else 0
        num_sample_per_img = len(sample[0])
        gts = {}
        res = {}
        item_id = 0
        for i in range(num_greedy):
            gts[item_id], res[item_id] = refs[i], greedy[i]
            item_id += 1
        for i in range(len(sample)):
            for j in range(num_sample_per_img):
                gts[item_id], res[item_id] = refs[i], sample[i][j: j + 1]
                item_id += 1
        num_items = item_id
        assert (len(sample) * num_sample_per_img + num_greedy) == num_items
        assert len(gts.keys()) == num_items

        scores = {}
        for metric in self.scorers:
            wg = weights[metric]
            if isinstance(wg, (float, int)) and wg <= 0:
                continue
            if isinstance(wg, (list, tuple)) and max(wg) <= 0:
                continue
            _, sc = self.scorers[metric].compute_score(gts, res)
            if isinstance(wg, (list, tuple)):
                for i, w in enumerate(wg):
                    scores[f"{metric}_{i}"] = np.array(sc[i]) * w
            else:
                scores[metric] = sc * wg

        scores = sum(scores.values())  # Sum across metrics
        assert len(scores) == num_items
        sc_greedy = scores[:num_greedy] if greedy else np.array([0])
        sc_sample = scores[num_greedy:]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{self.__class__.__name__}: Captions: greedy = `{greedy}`    sampled = `{sample}`"
            )
            logger.debug(
                f"{self.__class__.__name__}: "
                f"Average scores: greedy = `{sc_greedy.mean()}`    sampled = `{sc_sample.mean()}`"
            )
            greedy_num_words = 0
            sample_num_words = 0
            for i in range(num_items):
                if i < num_greedy:
                    greedy_num_words += len(res[i][0].split(" "))
                else:
                    sample_num_words += len(res[i][0].split(" "))
            logger.debug(
                f"{self.__class__.__name__}: Average # of words: "
                f"greedy = `{greedy_num_words / num_greedy if greedy else 0:.2f}`    "
                f"sampled = `{sample_num_words / (num_items - num_greedy):.2f}`"
            )

        return sc_sample, sc_greedy


class BleuSilent(Bleu):
    # noinspection PyMethodOverriding
    def compute_score(self, gts, res):
        return super().compute_score(gts=gts, res=res, verbose=0)


"""

Cross-entropy loss derivative is p_i - y_i,
    where p is the output of softmax and y is the one-hot label.
    This means XE loss grad is prob of class i minus 1.0 if true or 0 if false.

SCST loss derivative is 
    [r(sampled) - r(greedy)] * [p(sample @ t) - oneHot(sample @ t)]
    This means it is equivalent to a weighted version of XE loss, where
    the labels are sampled captions, and the weights are baselined rewards.
        dec_log_ppl = tf.contrib.seq2seq.sequence_loss(
                                        logits=sampled_logits,
                                        targets=sampled_onehot,
                                        weights=sampled_masks,
                                        average_across_batch=False)
        dec_log_ppl = tf.reduce_mean(dec_log_ppl * rewards)


"""
