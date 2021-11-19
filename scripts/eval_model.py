# -*- coding: utf-8 -*-
"""
Created on 26 Oct 2020 23:09:28
@author: jiahuei
"""
import os
import logging
import torch
from sparse_caption.utils.config import Config
from sparse_caption.utils.misc import configure_logging, replace_from_right
from sparse_caption.utils.model_utils import densify_state_dict
from sparse_caption.utils.training import TrainingModule
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
    parser.add_argument("--id", type=str, default="", help="str: An id identifying this run/job.")
    parser.add_argument("--log_dir", type=str, default="", help="str: Logging / Saving directory.")
    parser.add_argument(
        "--model_file", type=str, default="model_best_pruned_sparse.pth", help="str: Model checkpoint file."
    )
    parser.add_argument("--eval_dir_suffix", type=str, default="", help="str: Eval directory name suffix.")
    parser.add_argument("--beam_size_test", type=int, default=0, help="int: Beam size used for test set.")
    parser.add_argument("--beam_size_val", type=int, default=0, help="int: Beam size used for validation set.")
    parser.add_argument("--batch_size_eval", type=int, default=50, help="int: Batch size for evaluation.")
    parser.add_argument(
        "--load_as_float16",
        action="store_true",
        help="bool: If `True`, load model weights as `float16` (but run in float32).",
    )
    parser.add_argument(
        "--mscoco_online_test",
        action="store_true",
        help="bool: If `True`, run inference on MS-COCO `test2014` split.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="str: Dataset name. If not provided, load from config.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="str: Dataset directory. If not provided, load from config.",
    )
    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.id)
    return args


def main(args):
    config = Config.load_config_json(os.path.join(args.log_dir, "config.json"))
    if config.caption_model.endswith("_prune"):
        config.caption_model = replace_from_right(config.caption_model, "_prune", "", 1)
    config.update({k: v for k, v in vars(args).items() if v is not None})
    # config.max_seq_length = 3

    ckpt_path = os.path.join(args.log_dir, args.model_file)
    state_dict = torch.load(ckpt_path)
    if args.load_as_float16:
        config.eval_dir_suffix = f"{config.eval_dir_suffix}_float16" if config.eval_dir_suffix else "float16"
        state_dict = {k: v.to(torch.float16) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
        torch.save(state_dict, ckpt_path.replace(".pth", "_float16.pth"))
    state_dict = densify_state_dict(state_dict)
    logger.info(f"Model weights loaded from `{ckpt_path}`")

    if args.beam_size_val > 0 and args.beam_size_test > 0:
        raise ValueError("`beam_size_val` and `beam_size_test` cannot both be > 0")
    if args.beam_size_val > 0:
        split = "val"
    elif args.beam_size_test > 0:
        split = "test"
    else:
        raise ValueError("One of `beam_size_val` or `beam_size_test` must be > 0")
    return TrainingModule.eval_model(state_dict=state_dict, config=config, split=split)


if __name__ == "__main__":
    opt = parse_opt()
    logger = configure_logging(opt.logging_level)
    main(opt)
