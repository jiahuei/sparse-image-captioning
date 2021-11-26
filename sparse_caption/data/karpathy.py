# -*- coding: utf-8 -*-
"""
Created on 01 Aug 2020 18:19:23
@author: jiahuei
"""
import logging
import os
import json
import random
from tqdm import tqdm
from typing import Tuple, Iterable, Union
from abc import ABC, abstractmethod
from collections import defaultdict
from argparse import ArgumentParser, _ArgumentGroup
from .collate import ListDataset
from ..utils import misc as misc_utils, file as file_utils
from ..utils.config import Config

# from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

logger = logging.getLogger(__name__)


class KarpathyDataset(ABC):
    """Captioning data and inputs."""

    ANNOTATION_FILE = RAW_JSON_FILE = ""
    DEFAULT_ANNOT_DIR = os.path.join(misc_utils.PACKAGE_DIR, "coco_caption", "annotations")

    def __init__(self, config: Config) -> None:
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            config: Configuration object.
        """
        self.data = None
        self.config = config
        self.dataset_dir = self.config.dataset_dir

    @property
    def train_size(self):
        return len(self.data["train"])

    @staticmethod
    @abstractmethod
    def image_filename_to_id(filename: str) -> int:
        """Given the file name of an image, return its image ID (integer)."""
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self):
        """Download, process, tokenize, etc."""
        raise NotImplementedError

    def get_split(self, split: str, generation_mode: bool = False):
        if split not in ("train", "val", "test"):
            error_mssg = f"Invalid split `{split}`, please pass in one of ('train', 'val', 'test')."
            raise ValueError(error_mssg)

        if generation_mode:
            # Avoid repeated image since most datasets have > 1 captions per image
            # We cannot simply index with strides (eg [::5]), because some MSCOCO images have captions > 5
            data = {}
            for d in self.data[split]:
                data[d["img_id"]] = d
            data = list(data.values())
        else:
            data = self.data[split]
        data = [
            (
                _["img_path"],
                _["img_id"],
                _["caption"],
                _["all_captions"],
                _["all_gts"],
                # _["pos"],
            )
            for _ in data
        ]
        return ListDataset(data)

    def download_and_process_karpathy_json(self):
        # Read Karpathy's caption JSON file
        raw_json = os.path.join(self.dataset_dir, self.RAW_JSON_FILE)
        if os.path.isfile(raw_json):
            logger.debug(f"{self.__class__.__name__}: Found caption JSON file: `{self.RAW_JSON_FILE}`")
        else:
            file_utils.get_file(
                fname="caption_datasets.zip",
                origin=r"https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
                dest_dir=self.dataset_dir,
                extract=True,
            )
        with open(raw_json, "r") as f:
            data = json.load(f)

        # Process and tokenize
        self.data = defaultdict(list)
        all_img_id = set()
        all_filename = set()
        for d in tqdm(data["images"], desc="Processing dataset"):
            # Process image ID
            img_id = self.image_filename_to_id(d["filename"])
            all_img_id.add(img_id)
            all_filename.add(d["filename"])
            # The rest
            img_path = os.path.join(self.dataset_dir, d.get("filepath", "images"), d["filename"])
            split = "train" if d["split"] == "restval" else d["split"]
            # For SCST: coco_caption scorers require tokenized GT captions
            all_gts = [" ".join(sent["tokens"]) for sent in d["sentences"]]
            all_captions = [
                sent["raw"] if self.config.retokenize_captions else " ".join(sent["tokens"]) for sent in d["sentences"]
            ]
            for cap, sent in zip(all_captions, d["sentences"]):
                tmp_dict = {
                    "split": split,
                    "img_path": img_path,
                    "img_id": img_id,
                    "caption": cap,
                    "all_captions": all_captions,
                    "all_gts": all_gts,
                    # "pos": sent["pos"],
                }
                self.data[split].append(tmp_dict)

        # Assert that all image have unique IDs
        if len(all_img_id) != len(all_filename):
            error_mssg = "The number of unique image IDs does not match that of unique image file names."
            raise ValueError(error_mssg)
        # if save_processed_json:
        #     # Need a way to make sure the file has a unique name for every tokenizer / config.
        #     with open(os.path.join(self.dataset_dir, self.processed_json_file), 'w') as f:
        #         json.dump(self.data, f)
        self.random_image_check()

    def random_image_check(self, num_samples: int = 5) -> None:
        # Randomly check some image files
        passed = all(
            os.path.isfile(curr_data["img_path"]) for curr_data in random.sample(self.data["train"], num_samples)
        )
        if not passed:
            raise FileNotFoundError(
                "One or more training images are missing. " "Perhaps you need to re-download the dataset images."
            )

    def train_captions_txt_dump(self) -> None:
        """
        Generate a text file, one sentence per line. Used to train tokenizer.
        """
        config = self.config
        tokenizer_dir = os.path.join(config.log_dir, "tokenizer")
        train_txt = os.path.join(tokenizer_dir, "train_captions.txt")
        config.tokenizer_train_files = train_txt
        # if os.path.isfile(train_txt):
        #     logger.debug(f"{self.__class__.__name__}: Found tokenizer training file at `{train_txt}`.")
        #     return
        if os.path.isdir(os.path.dirname(train_txt)):
            logger.debug(f"{self.__class__.__name__}: Found tokenizer dir at `{os.path.dirname(train_txt)}`.")
            return
        os.makedirs(tokenizer_dir, exist_ok=True)
        with open(train_txt, "w") as f:
            f.write("\n".join([sent["caption"] for sent in self.data["train"]]))

    def coco_annot_json_dump(self) -> None:
        """
        Generate COCO-style annotation file, if it does not exist.
        """
        assert self.ANNOTATION_FILE.endswith(
            ".json"
        ), f"`self.ANNOTATION_FILE` should end with `.json`, saw `{self.ANNOTATION_FILE}` instead."
        json_fpath = os.path.join(self.DEFAULT_ANNOT_DIR, self.ANNOTATION_FILE)
        if os.path.isfile(json_fpath):
            logger.debug(f"{self.__class__.__name__}: Found annotation file at `{json_fpath}`.")
            return
        logger.debug(f"{self.__class__.__name__}: Generating COCO-style annotation file at `{json_fpath}`.")
        annot = dict(
            images=[],
            annotations=[],
            info="",
            type="captions",
            licenses="",
        )
        for split in ("val", "test"):
            assert split in self.data, f"Split `{split}` not found in `self.data`."
            for d in self.data[split]:
                annot["images"].append({"id": d["img_id"]})
                annot["annotations"].append({"caption": d["raw"], "id": 0, "image_id": d["img_id"]})

        os.makedirs(os.path.split(json_fpath)[0], exist_ok=True)
        with open(json_fpath, "w") as f:
            json.dump(annot, f)

    def coco_caption_json_dump(
        self,
        img_fname_caption_pair: Iterable[Tuple[str, ...]],
        output_fpath: str,
    ) -> None:
        """
        Takes in `[(img_fname_str, caption_str), ...]` as `img_fname_caption_pair`,
        and saves the results as a JSON file compatible with `coco_caption` evaluation format.

        Args:
            img_fname_caption_pair: An `Iterable` of (image name, caption string).
            output_fpath: The path for the output file.
        Returns:
            The file path of the JSON file.
        """
        assert isinstance(img_fname_caption_pair, Iterable)
        assert output_fpath.endswith(".json"), f"`output_fpath` should end with `.json`, saw `{output_fpath}` instead."
        # Get image IDs
        coco_json = []
        for img_fname, caption in img_fname_caption_pair:
            image_id = self.image_filename_to_id(os.path.basename(img_fname))
            assert isinstance(caption, str), "Caption must be a string."
            # caption = caption.encode('ascii', 'ignore').decode()
            coco_json.append({"image_id": image_id, "caption": caption})

        # Save results to JSON file
        os.makedirs(os.path.split(output_fpath)[0], exist_ok=True)
        with open(output_fpath, "w") as f:
            json.dump(coco_json, f)

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        # assert isinstance(parser, _ArgumentGroup)
        parser.add_argument(
            "--dataset_dir",
            type=str,
            default=None,
            help="str: Dataset directory.",
        )
        parser.add_argument(
            "--retokenize_captions",
            action="store_true",
            help="bool: If `True`, retokenize. Otherwise use captions tokenized by Karpathy",
        )
        # fmt: on
        # return parser
