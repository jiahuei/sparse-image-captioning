# -*- coding: utf-8 -*-
"""
Created on 28 Aug 2020 12:43:22
@author: jiahuei
"""
import os
import logging
import importlib
from typing import Any

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {}


def register_model(name):
    """
    New models can be added with the :func:`register_model` function decorator.

    For example::

        @register_model('relation_transformer')
        class RelationTransformerModel:
            (...)

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model: `{name}`.")
        MODEL_REGISTRY[name.lower()] = cls
        return cls

    return register_model_cls


def get_model(name: str) -> Any:
    name = name.lower()
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        _list = "\n".join(MODEL_REGISTRY.keys())
        error_mssg = f"Model specified `{name}` is invalid. Available options are: \n{_list}"
        raise ValueError(error_mssg)


# automatically import any Python files in the current directory
curr_dir = os.path.dirname(__file__)
for file in os.listdir(curr_dir):
    path = os.path.join(curr_dir, file)
    if not file.startswith("_") and not file.startswith(".") and (file.endswith(".py") or os.path.isdir(path)):
        module_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(f"sparse_caption.models.{module_name}")
