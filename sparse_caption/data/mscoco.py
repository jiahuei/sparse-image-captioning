# -*- coding: utf-8 -*-
"""
Created on 21 Apr 2020 00:25:38
@author: jiahuei
"""

import os
import logging
from typing import Union
from shutil import copyfile
from argparse import ArgumentParser, _ArgumentGroup
from . import KarpathyDataset, register_dataset
from ..utils import file as file_utils

logger = logging.getLogger(__name__)


@register_dataset("mscoco")
class MscocoDataset(KarpathyDataset):
    """COCO data and inputs."""

    ANNOTATION_FILE = "captions_val2014.json"
    # RAW_JSON_FILE = "dataset_coco_pos.json"
    RAW_JSON_FILE = "dataset_coco.json"

    def __init__(self, config):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            config: Configuration object.
        """
        logger.info(f"{self.__class__.__name__}: Dataset initializing ...")
        super().__init__(config)

    def prepare_data(self):
        self.download_and_process_karpathy_json()
        if self.config.get("mscoco_online_test", False):
            # Use karpathy test as validation split
            self.data = {
                "train": self.data["train"] + self.data["val"],
                "val": self.data["test"],
                "test": self.get_test2014_split(),
            }
        else:
            self.data = {
                "train": self.data["train"],
                "val": self.data["val"],
                "test": self.data["test"],
            }
        annot_fpath = os.path.join(self.DEFAULT_ANNOT_DIR, self.ANNOTATION_FILE)
        # Maybe download and copy `val2014` annot file to `coco_caption/annotations`
        if not os.path.isfile(annot_fpath):
            file_utils.get_file(
                fname="annotations_trainval2014.zip",
                origin=r"http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                dest_dir=self.dataset_dir,
                extract=True,
            )
            copyfile(src=os.path.join(self.dataset_dir, "annotations", self.ANNOTATION_FILE), dst=annot_fpath)
        self.train_captions_txt_dump()

    def get_test2014_split(self):
        img_paths = file_utils.list_dir(os.path.join(self.dataset_dir, "test2014"))
        img_ids = [self.image_filename_to_id(os.path.basename(_)) for _ in img_paths]
        data = [
            {
                "img_path": p,
                "img_id": i,
                "caption": "",
                "all_captions": [""],
                "all_gts": [""],
            }
            for p, i in zip(img_paths, img_ids)
        ]
        return data

    @staticmethod
    def image_filename_to_id(filename: str) -> int:
        # Example: "COCO_val2014_000000522418.jpg"
        if filename.endswith(".jpg"):
            filename = filename.replace(".jpg", "")
        else:
            raise ValueError(f"Expected all MS-COCO images to be of `.jpg`, saw `{filename}` instead.")
        return int(filename.split("_")[-1])

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        """Adds dataset arguments to `ArgumentParser`."""
        # fmt: off
        # assert isinstance(parser, ArgumentParser)
        KarpathyDataset.add_argparse_args(parser)
        parser.add_argument(
            "--mscoco_online_test",
            action="store_true",
            help="bool: If `True`, include both `val` and `restval` into `train` split, and use `test` for validation.",
        )
        # fmt: on
        # return parser


@register_dataset("mscoco_testing")
class MscocoTesting(MscocoDataset):
    RAW_JSON_FILE = "dataset_coco_testing.json"
