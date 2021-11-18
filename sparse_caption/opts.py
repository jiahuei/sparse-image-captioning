# -*- coding: utf-8 -*-
"""
Created on 16 Sep 2020 17:21:25
@author: jiahuei
"""
import os
import logging
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from .utils.training import TrainingModule
from .data import get_dataset, DATASET_REGISTRY
from .tokenizer import get_tokenizer, TOKENIZER_REGISTRY
from .models import get_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)


def parse_opt(arguments=None) -> Namespace:
    if arguments is not None:
        assert isinstance(arguments, list), f"Expected `arguments` to be a list, received: {type(arguments)}"

    # fmt: off
    # noinspection PyTypeChecker
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mscoco",
        choices=DATASET_REGISTRY.keys(),
        help="str: Dataset name.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="word",
        choices=TOKENIZER_REGISTRY.keys(),
        help="str: Tokenizer name.",
    )
    parser.add_argument(
        "--caption_model", type=str, default="relation_transformer",
        choices=MODEL_REGISTRY.keys(),
        help="str: Model name."
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="str: Logging level.",
    )
    # parser.add_argument(
    #     "--cache_dir",
    #     type=str,
    #     default=None,
    #     help="str: Cache directory for `torchvision`."
    # )
    # fmt: on
    args, unknown = parser.parse_known_args(arguments)
    TrainingModule.add_argparse_args(
        parser.add_argument_group("Training", "Arguments for model training, logging and evaluation.")
    )
    get_dataset(args.dataset).add_argparse_args(
        parser.add_argument_group("Dataset", "Arguments for dataset and dataloaders.")
    )
    get_tokenizer(args.tokenizer).add_argparse_args(
        parser.add_argument_group(
            "Tokenizer",
            "Arguments for tokenization, defaulting to values defined in each tokenizer class.",
        )
    )
    get_model(args.caption_model).add_argparse_args(
        parser.add_argument_group("Model", "Arguments for model, hyperparameters.")
    )
    args = parser.parse_args(arguments)

    # Paths
    args.log_dir = os.path.join(args.log_dir, f"{args.id}")
    return args
