# -*- coding: utf-8 -*-
"""
Created on 29 Apr 2020 15:55:21
@author: jiahuei

Utility functions.
"""
import logging
import argparse
import os
import functools
import time
import itertools
from typing import Union

logger = logging.getLogger(__name__)

up_dir = os.path.dirname
CURR_DIR = up_dir(os.path.realpath(__file__))
BASE_DIR = up_dir(CURR_DIR)
REPO_DIR = up_dir(BASE_DIR)
assert os.path.isdir(
    os.path.join(REPO_DIR, "caption_vae")
), f"`REPO_DIR` should point to `caption-vae-transformer`, saw `{REPO_DIR}` instead."


def configure_logging(
        logging_level: Union[int, str] = logging.INFO,
        logging_fmt: str = "%(levelname)s: %(name)s: %(funcName)s: %(message)s",
        logger_obj: Union[None, logging.Logger] = None,
) -> logging.Logger:
    """
    Setup logging on the root logger, because `transformers` calls `logger.info` upon import.

    Adapted from:
        https://stackoverflow.com/a/54366471/5825811
        Configures a simple console logger with the given level.
        A use-case is to change the formatting of the default handler of the root logger.

    Format variables:
        https://docs.python.org/3/library/logging.html#logrecord-attributes
    """
    logger_obj = logger_obj or logging.getLogger()  # either the given logger or the root logger
    # logger_obj.removeHandler(stanza.log_handler)
    logger_obj.handlers.clear()
    logger_obj.setLevel(logging_level)
    # If the logger has handlers, we configure the first one. Otherwise we add a handler and configure it
    if logger_obj.handlers:
        console = logger_obj.handlers[0]  # we assume the first handler is the one we want to configure
    else:
        console = logging.StreamHandler()
        logger_obj.addHandler(console)
    console.setFormatter(logging.Formatter(logging_fmt))
    console.setLevel(logging_level)
    # # Work around to update TensorFlow's absl.logging threshold which alters the
    # # default Python logging output behavior when present.
    # # see: https://github.com/abseil/abseil-py/issues/99
    # # and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
    # try:
    #     import absl.logging
    # except ImportError:
    #     pass
    # else:
    #     absl.logging.set_verbosity("info")
    #     absl.logging.set_stderrthreshold("info")
    #     absl.logging._warn_preinit_stderr = False
    return logger_obj


def get_memory_info():
    """
    Get node total memory and memory usage
    https://stackoverflow.com/a/17718729
    """
    with open("/proc/meminfo", "r") as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == "MemTotal:":
                ret["total"] = int(sline[1])
            elif str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                tmp += int(sline[1])
        ret["free"] = tmp
        ret["used"] = int(ret["total"]) - int(ret["free"])
    return ret


def time_function(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        if logger.isEnabledFor(logging.DEBUG):
            elapsed_time = time.perf_counter() - tic
            logger.debug(
                f"{func.__name__}: Elapsed time: {elapsed_time:.6f} sec"
            )
        return value

    return wrapper_timer


def grouper(iterable, group_n, fill_value=0):
    args = [iter(iterable)] * group_n
    return list(itertools.zip_longest(fillvalue=fill_value, *args))


def replace_from_right(string: str, old: str, new: str, count: int = -1):
    """
    String replacement from right to left.
    https://stackoverflow.com/a/3679215
    """
    assert isinstance(string, str)
    assert isinstance(old, str)
    assert isinstance(new, str)
    assert isinstance(count, int)
    string = string.rsplit(old, count)
    return new.join(string)


def str_to_none(input_string: str):
    assert isinstance(input_string, str)
    if input_string.lower() == "none":
        return None
    else:
        return input_string


def csv_to_int_list(input_string: str):
    if input_string is None or input_string == "":
        return None
    assert isinstance(input_string, str)
    try:
        return [int(_) for _ in input_string.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)


def csv_to_float_list(input_string: str):
    if input_string is None or input_string == "":
        return None
    assert isinstance(input_string, str)
    try:
        return [float(_) for _ in input_string.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)


def csv_to_str_list(input_string: str):
    if input_string is None or input_string == "":
        return None
    else:
        assert isinstance(input_string, str)
        try:
            return [_.strip() for _ in input_string.split(",")]
        except ValueError as e:
            raise argparse.ArgumentTypeError(e)


def str_to_bool(input_string: str):
    if input_string.lower() in ("true", "1"):
        return True
    elif input_string.lower() in ("false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ChoiceList:
    """
    A Type for ArgParse for validation of choices.
    https://mail.python.org/pipermail/tutor/2011-April/082825.html
    """

    def __init__(self, choices):
        self.choices = choices

    def __repr__(self):
        return f"{self.__class__.__name__}({self.choices})"

    def __call__(self, csv):
        try:
            args = csv.split(",")
            remainder = sorted(set(args) - set(self.choices))
            if remainder:
                raise ValueError(f"Invalid choices: {remainder} (choose from {self.choices})")
            return args
        except ValueError as e:
            raise argparse.ArgumentTypeError(e)
