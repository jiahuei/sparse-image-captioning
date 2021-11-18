# -*- coding: utf-8 -*-
"""
Created on 19 Oct 2020 23:57:58
@author: jiahuei

cd sparse_caption
python -m scripts.collect_scores
"""
from __future__ import annotations
import os
import logging
import json
import pandas as pd
from io import StringIO
from tqdm import tqdm
from functools import reduce
from collections import defaultdict
from copy import deepcopy
from time import localtime, strftime
from decimal import Decimal
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from sparse_caption.utils.file import list_files
from sparse_caption.utils.config import Config


class Score:
    DELIMITER: str = ","
    METRICS: list = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
    value: pd.DataFrame
    model_name: str

    def __init__(self, model_name, data, header="Experiment"):
        self.value = self._make_df(header, data)
        self.model_name = model_name

    def add_data(self, header, data):
        self.value = pd.concat([self.value, self._make_df(header, data)], axis=1)

    def add_test_csv(self, csv_fp):
        self.value = pd.concat([self.value, self._pd_read_csv(csv_fp)], axis=1)

    def add_validation_csv(self, csv_fp, best_ckpt=None):
        if best_ckpt is None:
            raise ValueError("Invalid checkpoint.")
        df = self._pd_read_csv(csv_fp)
        self.value = pd.concat([self.value, df[df["Step"] == best_ckpt].reset_index(drop=True)], axis=1)

    def shift(self, shift: int = 2, new_precision: int = 1):
        """
        Returns a new Score instance with metric scores shifted by `shift` amount,
        with `new_precision` decimal places.

        Args:
            shift: int, shift the digits by the amount specified.
            new_precision: int, number of decimal places.
        Return:
            A new Score instance.
        """
        score = deepcopy(self)
        try:
            new_precision = Decimal(Decimal("1").as_tuple()._replace(exponent=-new_precision))
            score.value[score.METRICS] = score.value[score.METRICS].applymap(
                lambda x: str(Decimal(x).shift(shift).quantize(new_precision))
            )
        except KeyError:
            logger.info(f"{self.__class__.__name__}: shift() skipped: No metric scores found for {str(self)}.")
        return score

    def merge(self, score: "Score"):
        new_score = deepcopy(self)
        new_score.value = pd.merge(new_score.value, score.value, how="outer")
        return new_score

    @staticmethod
    def _pd_read_csv(csv_fp):
        return pd.read_csv(csv_fp, dtype=str, delimiter=",")

    def _make_df(self, header, data):
        return self._pd_read_csv(StringIO(f"{header}\n{data}"))

    def __repr__(self):
        return self.value.to_string()

    def __str__(self):
        # noinspection PyTypeChecker
        return "\n".join(",".join(_) for _ in self.value.fillna("").to_numpy(str).tolist())

    @property
    def header(self):
        return ",".join(self.value.columns.to_list())

    @property
    def best_checkpoint(self):
        try:
            return self.value["Step"].to_numpy()[0]
        except TypeError as e:
            raise ValueError("More than 1 test checkpoint set. Please reload data.") from e
        except KeyError as e:
            raise ValueError("Best checkpoint not set. Please load data from test score CSV first.") from e


class ScoreCollector:
    def __init__(self, config, check_train_captions=False):
        self.config = config
        train_caption_files = list(filter(lambda x: x.endswith("train_captions.txt"), list_files(self.config.log_dir)))
        if check_train_captions:
            train_captions = [
                set(self.read_file(_))
                for _ in tqdm(train_caption_files, desc=f"{self.__class__.__name__}: Checking train caption files")
            ]
            if self.list_items_equal(train_captions):
                logger.info(f"{self.__class__.__name__}: All train captions are identical.")
            else:
                logger.warning(f"{self.__class__.__name__}: Some train captions are different.")
        self.train_captions_set = set(self.read_file(train_caption_files[0]))
        self.experiments = list(filter(os.path.isdir, self.list_dir(self.config.log_dir)))

    def collect_scores(self):
        all_scores = defaultdict(list)
        num_exp = len(self.experiments)
        for i, exp_dir in enumerate(self.experiments):
            exp_name = os.path.basename(exp_dir)
            logger.info(f"{self.__class__.__name__}: Reading directory ({i + 1}/{num_exp}): `{exp_name}`")
            try:
                model_config = Config.load_config_json(os.path.join(exp_dir, "config.json"), verbose=False)
            except FileNotFoundError:
                logger.warning(f"{self.__class__.__name__}: No config JSON file found at:\n`{exp_dir}`")
                continue
            # Load test data
            test_dirs = list(filter(self.is_test_dir, self.list_dir(exp_dir)))
            if len(test_dirs) == 0:
                all_scores["no_test"].append(Score(model_config.caption_model, exp_name))
                continue
            best_checkpoint = None
            for test_dir in test_dirs:
                score = Score(model_config.caption_model, exp_name)
                # Collect scores
                try:
                    score.add_test_csv(os.path.join(test_dir, "scores.csv"))
                except TypeError as e:
                    # https://stackoverflow.com/a/46091127
                    raise TypeError(f"Invalid score format in `{test_dir}`") from e
                score.add_data(*self.compute_caption_stats(test_dir, model_config))
                score.add_data("Test dir", os.path.basename(test_dir))
                self._load_model_params(exp_dir, score)
                # Maybe collect sparsities, prioritise file in test_dir
                try:
                    score.add_test_csv(os.path.join(test_dir, "sparsities.csv"))
                except FileNotFoundError:
                    try:
                        score.add_test_csv(os.path.join(exp_dir, "sparsities.csv"))
                    except FileNotFoundError:
                        pass
                best_checkpoint = score.best_checkpoint
                all_scores[os.path.basename(test_dir)].append(score)
            # Load validation data
            if best_checkpoint is None:
                continue
            val_dirs = list(filter(self.is_val_dir, self.list_dir(exp_dir)))
            for val_dir in val_dirs:
                score = Score(model_config.caption_model, exp_name)
                score.add_validation_csv(os.path.join(val_dir, "scores.csv"), best_checkpoint)
                score.add_data("Val dir", os.path.basename(val_dir))
                self._load_model_params(exp_dir, score)
                all_scores[os.path.basename(val_dir)].append(score)

        # Write to CSV
        self._write_output_csv(all_scores)
        # Scale score by 100
        self._write_output_csv({k: [_.shift() for _ in v] for k, v in all_scores.items()}, "compiled_scores_100x.csv")
        logger.info(f"{self.__class__.__name__}: Done.")

    def _write_output_csv(self, all_scores: dict[str, list[Score]], filename="compiled_scores.csv"):
        # Filter / sort
        out_file = os.path.join(self.config.log_dir, filename)
        try:
            existing_data = set(self.read_file(out_file))
        except FileNotFoundError:
            existing_data = set()

        out_str = ""
        for score_dir, scores in sorted(all_scores.items()):
            scores = list(filter(lambda x: str(x) not in existing_data, scores))
            if len(scores) == 0:
                continue
            out_str += f"\n>>> {score_dir}\n"
            model_set = set(_.model_name for _ in scores)
            for model in sorted(model_set):
                # Merge DataFrames
                sc = reduce(lambda x, y: x.merge(y), filter(lambda x: x.model_name == model, scores))
                # Compute ORT param groups
                if model == "relation_transformer" and "Params" in sc.header:
                    df = sc.value
                    cols = df.columns.to_list()
                    att_cols = list(filter(lambda x: "_attn." in x, cols))
                    emb_cols = list(filter(lambda x: ".generator" in x or ".tgt_embed" in x, cols))
                    df["Attention params"] = df[att_cols].fillna(0).astype(int).sum(axis=1)
                    df["Embedding params"] = df[emb_cols].fillna(0).astype(int).sum(axis=1)
                    cols.insert(cols.index("Params") + 1, "Attention params")
                    cols.insert(cols.index("Params") + 1, "Embedding params")
                    sc.value = df[cols]
                out_str += f"\n{sc.header}\n"
                out_str += str(sc)
                out_str += "\n"
            out_str += "\n"

        if out_str == "":
            logger.info(f"{self.__class__.__name__}: No difference with existing file: `{out_file}`")
        else:
            logger.info(f"{self.__class__.__name__}: Writing to file: `{out_file}`")
            current_time_str = strftime("%Y/%m/%d %H:%M:%S", localtime())
            out_str = f"\n--------\nCompiled: {current_time_str}\n--------\n{out_str}"
            with open(out_file, "a") as f:
                f.write(out_str)

    @staticmethod
    def list_items_equal(lst):
        """
        Returns True if all items in list are equal. Returns False for empty list.
        https://stackoverflow.com/a/3844832
        """
        return lst and lst.count(lst[0]) == len(lst)

    @staticmethod
    def is_test_dir(path):
        return "test_" in os.path.basename(path) and os.path.isdir(path)

    @staticmethod
    def is_val_dir(path):
        return "val_" in os.path.basename(path) and os.path.isdir(path)

    def _load_model_params(self, exp_dir, score):
        # Load model params if available
        try:
            model_params = self.read_json(os.path.join(exp_dir, "model_params.json"))
            score.add_data("Params", model_params["total"])
            for k, v in model_params["breakdown"].items():
                score.add_data(k, v)
        except FileNotFoundError:
            pass

    def compute_caption_stats(self, test_dir, model_config):
        # Load vocab size from config
        vocab_size = model_config.vocab_size

        # Find caption file
        caption_file = list(
            filter(
                lambda x: os.path.basename(x).startswith("caption_") and x.endswith(".json"), self.list_dir(test_dir)
            )
        )
        if len(caption_file) == 1:
            pass
        elif len(caption_file) > 1:
            logger.warning(
                f"{self.__class__.__name__}: "
                f"More than one caption JSON file found in `{test_dir}`, using the first one."
            )
        else:
            raise ValueError(
                f"{self.__class__.__name__}: "
                f"Expected at least one caption JSON file in `{test_dir}`, saw {len(caption_file)}"
            )

        caption_data = self.read_json(caption_file[0])
        caption_data = [d["caption"] for d in caption_data]

        # Calculate stats
        appear_in_train = 0
        counts = defaultdict(int)
        caption_length = []
        assert isinstance(self.train_captions_set, set)
        for caption in caption_data:
            # Unique
            if caption in self.train_captions_set:
                appear_in_train += 1
            # Vocab
            caption = caption.split(" ")
            for w in caption:
                counts[w] += 1
            # Length
            caption_length.append(len(caption))

        vocab_coverage = (len(counts) / (vocab_size - 2)) * 100.0  # Exclude <GO> and <EOS>
        average_length = sum(caption_length) / len(caption_length)
        percent_unique = (1.0 - (appear_in_train / len(caption_data))) * 100.0

        header = "Vocab coverage,Pct unique,Avg len,Num captions"
        data = f"{vocab_coverage:.1f},{percent_unique:.1f},{average_length:.2f},{len(caption_data):d}"
        return header, data

    def check_compiled_scores(self):
        out_file = os.path.join(self.config.log_dir, "compiled_test_scores.csv")
        compiled_scores = list(
            filter(lambda x: len(x.split(",")) > 1 and not x.startswith("Experiment,"), self.read_file(out_file))
        )
        for line in tqdm(compiled_scores, desc=f"{self.__class__.__name__}: Checking `compiled_test_scores.csv`"):
            line = line.split(",")
            exp_dir = line[0]
            scores = ",".join(line[1:10])
            test_dir = line[14]
            try:
                _, ref_scores = self.read_file(os.path.join(self.config.log_dir, exp_dir, test_dir, "scores.csv"))
                assert scores == ref_scores, (
                    f"Score mismatch for `{os.path.join(exp_dir, test_dir)}`: "
                    f"Compiled = {scores}    Reference = {ref_scores}"
                )
            except FileNotFoundError:
                logger.warning(f"{self.__class__.__name__}: `{os.path.join(exp_dir, test_dir)}` not found.")
        logger.info(f"{self.__class__.__name__}: `compiled_test_scores.csv` check complete.")
        return True

    @staticmethod
    def list_dir(path):
        return sorted(os.path.join(path, _) for _ in os.listdir(path))

    @staticmethod
    def read_file(path):
        with open(path, "r") as f:
            data = [_.rstrip() for _ in f.readlines()]
        return data

    @staticmethod
    def read_json(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def parse_opt() -> Namespace:
        # fmt: off
        # noinspection PyTypeChecker
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            "--log_dir",
            type=str,
            default="/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1",
            help="str: Directory path.",
        )
        parser.add_argument(
            "--logging_level",
            type=str,
            default="INFO",
            choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
            help="str: Logging level.",
        )
        parser.add_argument(
            "--skip_check_train_file", "-s",
            action="store_true",
            help="bool: If True, skip tokenizer train file check.",
        )
        parser.add_argument(
            "--check_compiled_scores", "-c",
            action="store_true",
            help="bool: If True, check compiled metric scores against original CSV files.",
        )
        # fmt: on
        return parser.parse_args()


if __name__ == "__main__":
    c = ScoreCollector.parse_opt()
    logging.basicConfig(level=c.logging_level)
    logger = logging.getLogger(__name__)
    collector = ScoreCollector(c, check_train_captions=not c.skip_check_train_file)
    collector.collect_scores()
    if c.check_compiled_scores:
        collector.check_compiled_scores()
