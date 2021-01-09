# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:48:09 2017

@author: jiahuei
"""
import os
import logging
from packaging import version
from typing import Union, Type, TypeVar, Dict
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from utils.file import read_json, dump_json
from version import __version__

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Config:
    """ Configuration object."""
    VERSION = __version__

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
        c_dict = cls.compat(read_json(config_filepath))
        logger.info(f"{cls.__name__}: Loaded from `{config_filepath}`.")
        return cls(**c_dict)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        self_dict = vars(self)
        if logger.isEnabledFor(logging.DEBUG) and key in self_dict and self_dict[key] != value:
            logger.warning(
                f"{self.__class__.__name__}: "
                f"Overwriting `self.{key}` from `{self_dict[key]}` to `{value}`"
            )
        super().__setattr__(key, value)

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
        loaded_version = version.parse(config_dict.get("version", "0.1.0"))
        if loaded_version == version.parse(cls.VERSION):
            return config_dict

        logger.warning(
            f"{cls.__name__}: Version mismatch: "
            f"Current is `{cls.VERSION}`, loaded config is `{loaded_version}`"
        )

        if loaded_version < version.parse("0.3.0"):
            config_dict["vocab_size"] += 1
            if "transformer" in config_dict["caption_model"]:
                config_dict["d_model"] = config_dict["input_encoding_size"]
                config_dict["dim_feedforward"] = config_dict["rnn_size"]
                config_dict["drop_prob_src"] = config_dict["drop_prob_lm"]
                del config_dict["input_encoding_size"]
                del config_dict["rnn_size"]
                del config_dict["drop_prob_lm"]

        if loaded_version == version.parse("0.1.0"):
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

        if loaded_version == version.parse("0.0.0"):
            raise ValueError(
                f"{cls.__name__}: Compatibility error, "
                f"unable to convert config from version `{loaded_version}`."
            )
        return config_dict
