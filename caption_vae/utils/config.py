# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:48:09 2017

@author: jiahuei
"""
import os
import pickle
import logging
import json
from typing import Union, Type, TypeVar
from argparse import ArgumentParser, Namespace
from copy import deepcopy

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Config:
    """ Configuration object."""

    @classmethod
    def from_argument_parser(cls: Type[T], parser_or_args: Union[ArgumentParser, Namespace]) -> T:
        """
        Initialize a `Config` object from `ArgumentParser`,
        replacing `None` with default values defined in the constructor.
        Args:
            parser_or_args: `ArgumentParser` or `NameSpace` object.
        Returns:
            `Config` object.
        """
        if isinstance(parser_or_args, ArgumentParser):
            args = parser_or_args.parse_args()
        elif isinstance(parser_or_args, Namespace):
            args = parser_or_args
        else:
            raise TypeError("`parser_or_args` must be either `ArgumentParser` or `NameSpace` object.")
        return cls(**vars(args))

    # @classmethod
    # def load_config(cls, config_filepath):
    #     with open(config_filepath, 'rb') as f:
    #         c_dict = pickle.load(f)
    #     logger.info(f"{cls.__name__}: Loaded from `{config_filepath}`.")
    #     return cls(**c_dict)

    @classmethod
    def load_config_json(cls, config_filepath):
        with open(config_filepath, 'r') as f:
            c_dict = json.load(f)
        logger.info(f"{cls.__name__}: Loaded from `{config_filepath}`.")
        return cls(**c_dict)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # noinspection PyUnresolvedReferences
    def save_config(self, exist_ok=True):
        """
        Save this instance to a json file.

        Args:
            exist_ok (:obj:`bool`): If set to True, allow overwrites.
        """
        assert os.path.isdir(self.log_dir), f"Invalid logging path: {self.log_dir}"
        output_fp = os.path.join(self.log_dir, "config")
        if exist_ok is False and os.path.isfile(output_fp + ".json"):
            raise FileExistsError(f"Found existing config file at `{output_fp}`")

        config_dict = vars(self)
        with open(output_fp + ".json", "w") as f:
            json.dump(config_dict, fp=f, indent=2, sort_keys=True, ensure_ascii=False)
        # with open(output_fp + ".pkl", "wb") as f:
        #     pickle.dump(config_dict, f, pickle.HIGHEST_PROTOCOL)
        logger.info(f"{self.__class__.__name__}: Saved as `{output_fp + '.json'}`")
        return output_fp + ".json"

    def get(self, key, default_value):
        if logger.isEnabledFor(logging.DEBUG) and key not in vars(self):
            logger.debug(f"{self.__class__.__name__}: `{key}` not found, returning `{default_value}`")
        return vars(self).get(key, default_value)

    def update(self, kv_mapping):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.__class__.__name__}: Updating config with `{kv_mapping}`")
        return vars(self).update(kv_mapping)

    def deepcopy(self):
        return deepcopy(self)
