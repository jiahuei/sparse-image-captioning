# -*- coding: utf-8 -*-
"""
Created on 20 Apr 2020 19:00:29
@author: jiahuei

Adapted from:
    https://raw.githubusercontent.com/pytorch/fairseq/v0.9.0/fairseq/models/__init__.py
    http://scottlobdell.me/2015/08/using-decorators-python-automatic-registration/

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import logging
import importlib
from typing import Type
from .karpathy import KarpathyDataset

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {}


def register_dataset(name):
    """
    New datasets can be added with the :func:`register_dataset` function decorator.

    For example::

        @register_dataset('mscoco')
        class MscocoDataset:
            (...)

    Args:
        name (str): the name of the model
    """

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Cannot register duplicate dataset: `{name}`.")
        if not issubclass(cls, KarpathyDataset):
            raise ValueError(f"Dataset ({name}: {cls.__name__}) must extend `KarpathyDataset`.")
        DATASET_REGISTRY[name.lower()] = cls
        return cls

    return register_dataset_cls


def get_dataset(name: str) -> Type[KarpathyDataset]:
    name = name.lower()
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        _list = "\n".join(DATASET_REGISTRY.keys())
        error_mssg = f"Dataset specified `{name}` is invalid. Available options are: \n{_list}"
        raise ValueError(error_mssg)


# automatically import any Python files in the current directory
curr_dir = os.path.dirname(__file__)
for file in os.listdir(curr_dir):
    path = os.path.join(curr_dir, file)
    if not file.startswith("_") and not file.startswith(".") and (file.endswith(".py") or os.path.isdir(path)):
        module_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(f"sparse_caption.data.{module_name}")
