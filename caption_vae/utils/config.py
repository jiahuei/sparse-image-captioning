# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:48:09 2017

@author: jiahuei
"""
import os
import pickle
import logging
import json
from typing import Union, Type, TypeVar, Dict
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from utils.file import read_json, dump_json

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Config:
    """ Configuration object."""
    VERSION = "0.2.0"

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

    @classmethod
    def load_config_json(cls, config_filepath):
        c_dict = read_json(config_filepath)
        loaded_version = c_dict.get("version", None)
        if loaded_version != cls.VERSION:
            logger.warning(
                f"{cls.__name__}: Version mismatch: "
                f"Current is `{cls.VERSION}`, loaded config is `{loaded_version}`"
            )
            c_dict = cls.compat(c_dict)
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
        config_dict["version"] = self.VERSION
        dump_json(output_fp + ".json", config_dict, indent=2, sort_keys=True, ensure_ascii=False)
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

    @classmethod
    def compat(cls, config_dict: Dict):
        loaded_version = config_dict.get("version", None)

        def get_key(key):
            if key in config_dict:
                return config_dict[key]
            else:
                raise KeyError(
                    f"{cls.__name__}: Expected loaded config to have key `{key}` but not found."
                )

        if loaded_version is None:
            # SCST_MODES = ["greedy_baseline", "sample_baseline", "beam_search"]
            if "scst_mode" in config_dict:
                scst_mode = config_dict["scst_mode"]
                del config_dict["scst_mode"]
            elif "scst_beam_search" in config_dict:
                scst_mode = "beam_search" if config_dict["scst_beam_search"] else "greedy_baseline"
                del config_dict["scst_beam_search"]
            else:
                raise KeyError(
                    f"{cls.__name__}: Expected loaded config to have one of keys: "
                    f"`scst_mode` or `scst_beam_search`."
                )
            if scst_mode == "greedy_baseline":
                scst_baseline = "greedy"
                scst_sample = "random"
            elif scst_mode == "beam_search":
                scst_baseline = "greedy"
                scst_sample = "beam_search"
            else:
                scst_baseline = "sample"
                scst_sample = "random"
            config_dict["scst_baseline"] = scst_baseline
            config_dict["scst_sample"] = scst_sample
        else:
            raise ValueError(
                f"{cls.__name__}: Compatibility error, "
                f"unable to convert config from version `{loaded_version}`."
            )
        return config_dict
