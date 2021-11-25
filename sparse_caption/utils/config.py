# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:48:09 2017

@author: jiahuei
"""
import os
import json
import logging
from datetime import datetime
from pkg_resources import packaging
from copy import deepcopy
from .file import read_json, dumps_file
from ..version import __version__

logger = logging.getLogger(__name__)
version = packaging.version


class Config:
    """Configuration object."""

    @classmethod
    def load_config_json(cls, config_filepath, verbose=True):
        config = cls(**read_json(config_filepath)).compat()
        if verbose:
            logger.info(f"{cls.__name__}: Loaded from `{config_filepath}`.")
        return config

    def __init__(self, x: str = None, **kwargs):
        self.datetime = str(datetime.now())
        if x is not None:
            if not isinstance(x, str):
                raise TypeError(f"Positional argument must be a string, saw {type(x)}")
            kwargs.update(json.loads(x))
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        self_dict = vars(self)
        if logger.isEnabledFor(logging.DEBUG) and key in self_dict and self_dict[key] != value:
            logger.warning(f"{self.__class__.__name__}: Overwriting `self.{key}` from `{self_dict[key]}` to `{value}`")
        super().__setattr__(key, value)

    def __repr__(self):
        return self.json()

    def dict(self):
        d = {k: vars(v) if isinstance(v, Config) else v for k, v in vars(self).items()}
        d["version"] = __version__
        return d

    def json(self, **kwargs):
        kwargs["indent"] = kwargs.get("indent", 2)
        kwargs["sort_keys"] = kwargs.get("sort_keys", True)
        kwargs["ensure_ascii"] = kwargs.get("ensure_ascii", False)
        return json.dumps(self.dict(), **kwargs)

    # noinspection PyUnresolvedReferences
    def save_config(self, exist_ok=True):
        """
        Save this instance to a json file.

        Args:
            exist_ok (:obj:`bool`): If set to True, allow overwrites.
        """
        assert os.path.isdir(self.log_dir), f"Invalid logging path: {self.log_dir}"
        output_fp = os.path.join(self.log_dir, "config.json")
        if exist_ok is False and os.path.isfile(output_fp):
            raise FileExistsError(f"Found existing config file at `{output_fp}`")

        dumps_file(output_fp, self.json())
        logger.info(f"{self.__class__.__name__}: Saved as `{output_fp}`")
        return output_fp

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

    # noinspection PyAttributeOutsideInit
    def compat(self):
        loaded_version = version.parse(self.get("version", "0.1.0"))
        self.check_loaded_version(self.get("version", "0.1.0"))
        if loaded_version == version.parse(__version__):
            return self

        if loaded_version < version.parse("0.6.0"):
            self.share_att_encoder = self.share_att_decoder = None
            self.share_layer_encoder = self.share_layer_decoder = None
            if "relation_transformer" in self.caption_model:
                self.no_box_trigonometric_embedding = not self.box_trigonometric_embedding
            if "transformer" in self.caption_model:
                self.num_heads = self.get("num_heads", 8)

        if loaded_version < version.parse("0.3.0"):
            self.vocab_size += 1
            if "transformer" in self.caption_model:
                self.d_model = self.input_encoding_size
                self.dim_feedforward = self.rnn_size
                self.drop_prob_src = self.drop_prob_lm
                del self.input_encoding_size
                del self.rnn_size
                del self.drop_prob_lm

        if loaded_version == version.parse("0.1.0"):
            # SCST_MODES = ["greedy_baseline", "sample_baseline", "beam_search"]
            if "scst_mode" in vars(self):
                scst = self.scst_mode
                del self.scst_mode
            elif "scst_beam_search" in vars(self):
                scst = "beam_search" if self.scst_beam_search else "greedy_baseline"
                del self.scst_beam_search
            else:
                raise KeyError(
                    f"{self.__name__}: Expected loaded self to have one of keys: `scst_mode` or `scst_beam_search`."
                )
            if scst == "greedy_baseline":
                scst_baseline = "greedy"
                scst_sample = "random"
            elif scst == "beam_search":
                scst_baseline = "greedy"
                scst_sample = "beam_search"
            else:
                scst_baseline = "sample"
                scst_sample = "random"
            self.scst_baseline = scst_baseline
            self.scst_sample = scst_sample

        if loaded_version == version.parse("0.0.0"):
            raise ValueError(
                f"{self.__class__.__name__}: Compatibility error, "
                f"unable to convert config from version `{loaded_version}`."
            )
        return self

    def check_loaded_version(self, loaded_version):
        def _warn(expected_ver):
            if loaded_version not in expected_ver:
                logger.warning(
                    f"{self.__class__.__name__}: Version mismatch: "
                    f"Expected `{expected_ver}`, loaded config is `{loaded_version}`"
                )

        try:
            _ = self.scst_baseline
            _ = self.scst_sample
        except AttributeError:
            _warn(["0.1.0"])
            return

        try:
            if "transformer" in self.caption_model:
                _ = self.d_model
                _ = self.dim_feedforward
                _ = self.drop_prob_src
        except AttributeError:
            _warn(["0.2.0"])
            return

        try:
            if "relation_transformer" in self.caption_model:
                _ = self.no_box_trigonometric_embedding
            if "transformer" in self.caption_model:
                _ = self.num_heads
                _ = self.share_att_encoder
                _ = self.share_att_decoder
                _ = self.share_layer_encoder
                _ = self.share_layer_decoder
        except AttributeError:
            _warn(["0.3.0", "0.4.0", "0.5.0"])
            return

        return
