# -*- coding: utf-8 -*-
"""
Created on 11 Sep 2020 13:36:27
@author: jiahuei
"""
import os
# import sys
# CURR_DIR = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(1, os.path.dirname(CURR_DIR))

import json
from tqdm import tqdm
from time import time
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from tokenizer import PosTagger


def parse_arguments() -> Namespace:
    # fmt: off
    # noinspection PyTypeChecker
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--json_path', type=str, required=True,
        help='The path to the JSON file.')
    parser.add_argument(
        "--retokenize_captions", action="store_true",
        help="bool: If `True`, retokenize. Otherwise use captions tokenized by Karpathy",
    )
    args = parser.parse_args()
    # fmt: on
    return args


def main(args):
    raw_json = args.json_path
    assert raw_json.endswith(".json"), f"Must be a JSON file, instead saw `{os.path.basename(raw_json)}`"
    with open(raw_json, "r") as f:
        data = json.load(f)

    tagger = PosTagger(pretokenized=not args.retokenize_captions)
    # Process and tokenize
    data["pos_retokenized"] = args.retokenize_captions
    captions = []
    for d in data["images"]:
        for sent in d["sentences"]:
            if args.retokenize_captions:
                caption = sent["raw"]
            else:
                caption = " ".join(sent["tokens"])
            assert not caption.endswith("\n")
            captions.append(caption)

    batch_size = 200
    pos = []
    for i in tqdm(range(0, len(captions), batch_size), desc="Tagging Part-of-Speech"):
        cap = "\n\n".join(captions[i: i + batch_size])
        pos += tagger(cap)

    assert len(captions) == len(pos)
    idx = 0
    for d in tqdm(data["images"], desc="Processing dataset"):
        for sent in d["sentences"]:
            sent["pos"] = " ".join(pos[idx])
            idx += 1

    with open(raw_json.replace(".json", "_pos2.json"), "w") as f:
        json.dump(data, f)


if __name__ == '__main__':
    main(parse_arguments())
