# -*- coding: utf-8 -*-
"""
Created on 21 Apr 2020 22:25:24
@author: jiahuei
"""

import logging
import os
import random
import numpy as np
import multiprocessing.managers as mp
import torch
import torchvision.transforms as transforms
from argparse import ArgumentParser, _ArgumentGroup
from typing import Union, Optional, Dict, List
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from ..utils.model_utils import sequence_from_numpy
from ..utils.misc import get_memory_info
from ..tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def _asserts(img_transform, tokenizer):
    assert isinstance(img_transform, transforms.Compose)
    assert isinstance(tokenizer, Tokenizer)


class ListDataset(Dataset):
    """Basically a `list` but is a subclass ofm `Dataset`."""

    def __init__(self, data: List):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class BasicCollate:
    def __init__(self, config, img_transform: Compose, tokenizer: Tokenizer):
        self.config = config
        self.img_transform = img_transform
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # config = self.config
        # if config.use_dummy_inputs:
        #     image = torch.normal(0, 1, size=(3, 224, 224), dtype=torch.float32)
        #     target = torch.randint(0, max(config.eos_token_id, 4000), size=(20,), dtype=torch.int64)
        #     target[-1] = config.eos_token_id
        #     return image, os.path.join(config.dataset_dir, "img_01.jpg"), target

        image_paths, captions, image_ids = zip(*batch)

        images = [self.img_transform(Image.open(_).convert("RGB")) for _ in image_paths]
        images = torch.stack(images, 0)

        input_ids = [torch.as_tensor(self.tokenizer.encode(_, add_bos_eos=True), dtype=torch.int64) for _ in captions]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.config.pad_token_id)
        assert input_ids[:, 0].eq(self.config.bos_token_id).all().item()
        labels = input_ids[:, 1:]
        labels_len = torch.sum(labels.ne(self.config.pad_token_id), dim=1)
        return {
            "images": images,
            "image_names": [os.path.basename(_) for _ in image_paths],
            "input_ids": input_ids[:, :-1],
            "labels": labels,
            "labels_len": labels_len,
        }


class UpDownCollate:
    def __init__(self, config, tokenizer: Tokenizer, cache_dict: Optional[Dict] = None):
        self.config = config
        self.tokenizer = tokenizer
        # noinspection PyUnresolvedReferences
        self.cache_dict = cache_dict if isinstance(cache_dict, mp.DictProxy) else None
        if self.cache_dict is not None:
            logger.info(f"{self.__class__.__name__}: Using multiprocessing cache dict.")
        if self.config.input_att_dir is None:
            self.config.input_att_dir = self.join_default_bu_dir("cocobu_att")
        assert self.config.seq_per_img > 0, "`self.config.seq_per_img` should be greater than 0"

    def join_default_bu_dir(self, dirname):
        return os.path.join(self.config.dataset_dir, "bu", dirname)

    def _cache_data(self, key, key_value_fn):
        if self.cache_dict is None:
            return key_value_fn(key)
        try:
            data = self.cache_dict[key]
            logger.debug(f"{self.__class__.__name__}: Cache hit: {key}")
        except KeyError:
            data = key_value_fn(key)
            logger.debug(f"{self.__class__.__name__}: Cache miss: {key}")
            mem_info = get_memory_info()
            if mem_info["free"] / mem_info["total"] > max(0.2, self.config.cache_min_free_ram):
                self.cache_dict[key] = data
                logger.debug(f"{self.__class__.__name__}: Cache add: {key}")
        return data

    @staticmethod
    def _get_att_feats(path):
        data = np.load(path)
        data = data.reshape(-1, data.shape[-1]).astype("float32")
        # data = np.random.normal(size=(36, 2048)).astype("float32")
        return data

    def _debug_logging(self, data):
        if logger.isEnabledFor(logging.DEBUG):
            _batch = "\n".join(f"{k}: {v.size() if isinstance(v, torch.Tensor) else v}" for k, v in data.items())
            logger.debug(f"{self.__class__.__name__}: Batch input: \n{_batch}")

    def __call__(self, batch):
        config = self.config

        # image_paths, image_ids, captions, all_captions, all_gts, pos = zip(*batch)
        image_paths, image_ids, captions, all_captions, all_gts = zip(*batch)

        att_feats = [
            self._cache_data(os.path.join(config.input_att_dir, f"{imgid}.npy"), self._get_att_feats)
            for imgid in image_ids
        ]

        att_masks = [torch.ones(_.shape[0]) for _ in att_feats]
        att_masks = torch.nn.utils.rnn.pad_sequence(att_masks, batch_first=True, padding_value=0)

        labels = [
            torch.as_tensor(
                self.tokenizer.encode(_, add_bos_eos=True, max_seq_length=config.max_seq_length), dtype=torch.int64
            )
            for gt in all_captions
            for _ in random.sample(gt, min(config.seq_per_img, len(gt)))
        ]
        label_masks = [torch.ones(_.shape[0]) for _ in labels]

        # Parts of Speech
        # pos = [
        #     torch.as_tensor(
        #         self.tokenizer.encode_tokenized(
        #             _.split(" "), add_bos_eos=False, max_seq_length=config.max_seq_length
        #         ),
        #         dtype=torch.int64
        #     )
        #     for _ in pos
        # ]

        data = {
            "att_feats": torch.nn.utils.rnn.pad_sequence(
                sequence_from_numpy(att_feats), batch_first=True, padding_value=0.0
            ),
            "att_masks": att_masks,
            "seqs": torch.nn.utils.rnn.pad_sequence(sequence_from_numpy(labels), batch_first=True, padding_value=0),
            "masks": torch.nn.utils.rnn.pad_sequence(
                sequence_from_numpy(label_masks), batch_first=True, padding_value=0
            ),
            "gts": all_gts,
            "image_paths": image_paths,
            "image_ids": image_ids,
            # "pos": torch.nn.utils.rnn.pad_sequence(
            #     sequence_from_numpy(pos), batch_first=True, padding_value=0
            # ),
        }
        return data

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        parser.add_argument(
            "--max_seq_length", type=int, default=18,
            help="int: Maximum sequence length including <BOS> and <EOS>.",
        )
        parser.add_argument(
            "--seq_per_img", type=int, default=5,
            help="Number of captions to sample for each image during training. "
                 "Can reduce CNN forward passes / Reduce disk read load."
        )
        parser.add_argument(
            "--input_att_dir", type=str, default=None,
            help="str: path to the directory containing the preprocessed att feats"
        )
        # fmt: on


class ObjectRelationCollate(UpDownCollate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.input_rel_box_dir is None:
            self.config.input_rel_box_dir = self.join_default_bu_dir("cocobu_box_relative")

    @staticmethod
    def _get_boxes(path):
        data = np.load(path).astype("float32")
        # data = np.random.normal(size=(36, 4)).astype("float32")
        return data

    def __call__(self, batch):
        config = self.config
        image_paths, image_ids, captions, all_captions, all_gts = zip(*batch)
        data = super().__call__(batch)

        # if config.get("norm_att_feat", False):
        #     att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)

        boxes = [
            self._cache_data(os.path.join(config.input_rel_box_dir, f"{imgid}.npy"), self._get_boxes)
            for imgid in image_ids
        ]
        data["boxes"] = torch.nn.utils.rnn.pad_sequence(sequence_from_numpy(boxes), batch_first=True, padding_value=0.0)
        self._debug_logging(data)
        return data

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        UpDownCollate.add_argparse_args(parser)
        parser.add_argument(
            "--input_rel_box_dir", type=str, default=None,
            help="str: this directory contains the bounding boxes in relative coordinates "
                 "for the corresponding image features in --input_att_dir"
        )
        # fmt: on


class AttCollate(UpDownCollate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.input_fc_dir is None:
            self.config.input_fc_dir = self.join_default_bu_dir("cocobu_fc")

    @staticmethod
    def _get_fc_feats(path):
        data = np.load(path).astype("float32")
        # data = np.random.normal(size=(2048,)).astype("float32")
        return data

    def __call__(self, batch):
        config = self.config
        image_paths, image_ids, captions, all_captions, all_gts = zip(*batch)
        data = super().__call__(batch)
        fc_feats = [
            self._cache_data(os.path.join(config.input_fc_dir, f"{imgid}.npy"), self._get_fc_feats)
            for imgid in image_ids
        ]
        data["fc_feats"] = torch.tensor(fc_feats)
        self._debug_logging(data)
        return data

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        UpDownCollate.add_argparse_args(parser)
        parser.add_argument(
            "--input_fc_dir", type=str, default=None,
            help="str: path to the directory containing the preprocessed fc feats"
        )
        # fmt: on
