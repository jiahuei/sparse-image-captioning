# -*- coding: utf-8 -*-
"""
Created on 29 Oct 2020 19:28:34
@author: jiahuei
"""
import os
import logging
import torch
from sparse_caption.pruning import prune
from sparse_caption.pruning.sampler import rounding_sigmoid
from sparse_caption.utils.config import Config
from sparse_caption.utils.misc import configure_logging
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter

logger = logging.getLogger(__name__)


def parse_opt() -> Namespace:
    # noinspection PyTypeChecker
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="str: Logging level.",
    )
    parser.add_argument("--log_dir", type=str, default="", help="str: Logging / Saving directory.")
    parser.add_argument("--id", type=str, default="", help="An id identifying this run/job.")
    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, f"{args.id}")
    return args


def main(args):
    config = Config.load_config_json(os.path.join(args.log_dir, "config.json"))
    assert config.prune_type == prune.REGULAR, f"Expected mask_type to be `{prune.REGULAR}`, saw `{config.prune_type}`."

    ckpt_path = os.path.join(args.log_dir, "model_best.pth")
    out_path = ckpt_path.replace(".pth", "_bin_mask.pth")
    state_dict = torch.load(ckpt_path)
    logger.info(f"Model weights loaded from `{ckpt_path}`")

    pruning_masks = set(filter(lambda k: k.endswith("_pruning_mask"), state_dict.keys()))
    assert len(pruning_masks) > 0, "Checkpoint file does not contain any pruning mask."
    state_dict = {k: rounding_sigmoid(v) if k in pruning_masks else v for k, v in state_dict.items()}
    torch.save(state_dict, out_path)
    logger.info(f"Model weights with binarized masks saved to `{out_path}`")
    return True


if __name__ == "__main__":
    opt = parse_opt()
    logger = configure_logging(opt.logging_level)
    main(opt)
