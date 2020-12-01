# -*- coding: utf-8 -*-
"""
Created on 19 Oct 2020 23:57:58
@author: jiahuei


python caption_vae/scripts/collect_scores.py
"""
import os
import logging
import json
from tqdm import tqdm
from itertools import chain
from collections import defaultdict
from time import localtime, strftime
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter


class Score:
    def __init__(self, data, header="Experiment"):
        self.type_check(header, data)
        self.header = header
        self.data = data

    def add_data(self, header, data):
        self.type_check(header, data)
        self.header = f"{self.header},{header}"
        self.data = f"{self.data},{data}"

    def __str__(self):
        return self.data

    def __repr__(self):
        return self.header + " --- " + self.data

    @staticmethod
    def type_check(header, data):
        assert isinstance(header, str), f"Expected header of type str, saw {type(header)}"
        assert isinstance(data, str), f"Expected data of type str, saw {type(data)}"


class ScoreCollector:
    def __init__(self, config, check_train_captions=True):
        self.config = config
        train_caption_files = list(filter(
            lambda x: x.endswith("train_captions.txt"), self.list_files(self.config.log_dir)
        ))
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

    def collect_scores(self):
        all_scores = defaultdict(list)
        for i, exp in enumerate(self.experiments):
            exp_name = os.path.basename(exp)
            logger.info(f"{self.__class__.__name__}: Reading directory: `{exp_name}`")
            subdirs = list(filter(self.is_test_dir, self.list_dir(exp)))
            if len(subdirs) == 0:
                all_scores["no_test"].append(Score(exp_name))
                continue
            for test_dir in subdirs:
                if not os.path.isfile(os.path.join(exp, "config.json")):
                    logger.warning(f"{self.__class__.__name__}: No config JSON file found at:\n`{exp}`")
                    continue
                score = Score(exp_name)
                # Collect scores
                try:
                    score.add_data(*self.read_file(os.path.join(test_dir, "scores.csv")))
                except TypeError as e:
                    # https://stackoverflow.com/a/46091127
                    raise TypeError(f"Invalid score format in `{test_dir}`") from e
                score.add_data(*self.compute_caption_stats(test_dir))
                score.add_data("Test dir", os.path.basename(test_dir))
                # Maybe collect sparsities, prioritise file in test_dir
                try:
                    score.add_data(*self.read_file(os.path.join(test_dir, "sparsities.csv")))
                except FileNotFoundError:
                    try:
                        score.add_data(*self.read_file(os.path.join(exp, "sparsities.csv")))
                    except FileNotFoundError:
                        pass
                all_scores[os.path.basename(test_dir)].append(score)

        # Filter / sort
        out_file = os.path.join(self.config.log_dir, "compiled_test_scores.csv")
        try:
            existing_data = set(self.read_file(out_file))
        except FileNotFoundError:
            existing_data = {}

        out_str = ""
        for test_dir, scores in sorted(all_scores.items()):
            scores = list(filter(lambda x: str(x) not in existing_data, scores))
            if len(scores) == 0:
                continue
            out_str += f"\n>>> {test_dir}\n"
            header_set = set(_.header for _ in scores)
            for header in sorted(header_set):
                out_str += f"\n{header}\n"
                out_str += "\n".join(map(str, filter(lambda x: x.header == header, scores)))
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
        logger.info(f"{self.__class__.__name__}: Done.")

    def compute_caption_stats(self, test_dir):
        # Load vocab size from config
        exp_config = self.read_json(os.path.join(os.path.dirname(test_dir), "config.json"))
        vocab_size = exp_config["vocab_size"]

        # Find caption file
        caption_file = list(filter(
            lambda x: os.path.basename(x).startswith("caption_") and x.endswith(".json"),
            self.list_dir(test_dir)
        ))
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
        caption_data = [d['caption'] for d in caption_data]

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

        vocab_coverage = (len(counts) / (vocab_size - 2)) * 100.  # Exclude <GO> and <EOS>
        average_length = sum(caption_length) / len(caption_length)
        percent_unique = (1. - (appear_in_train / len(caption_data))) * 100.

        header = "Vocab coverage,Pct unique,Avg len,Num captions"
        data = f"{vocab_coverage:.1f},{percent_unique:.1f},{average_length:.2f},{len(caption_data):d}"
        return header, data

    def check_compiled_scores(self):
        out_file = os.path.join(self.config.log_dir, "compiled_test_scores.csv")
        compiled_scores = list(filter(
            lambda x: len(x.split(",")) > 1 and not x.startswith("Experiment,"),
            self.read_file(out_file)
        ))
        for line in tqdm(compiled_scores, desc=f"{self.__class__.__name__}: Checking `compiled_test_scores.csv`"):
            line = line.split(",")
            exp_dir = line[0]
            scores = ",".join(line[1:10])
            test_dir = line[14]
            try:
                _, ref_scores = self.read_file(
                    os.path.join(self.config.log_dir, exp_dir, test_dir, "scores.csv")
                )
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
    def list_files(path):
        files = chain.from_iterable(
            (os.path.join(root, f) for f in files) for root, subdirs, files in os.walk(path)
        )
        return files

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
            "--check_compiled_scores",
            action="store_true",
            help="bool: If True, check compiled metric scores against original CSV files.",
        )
        # fmt: on
        return parser.parse_args()


if __name__ == "__main__":
    config = ScoreCollector.parse_opt()
    logging.basicConfig(level=config.logging_level)
    logger = logging.getLogger(__name__)
    collector = ScoreCollector(config)
    collector.collect_scores()
    if config.check_compiled_scores:
        collector.check_compiled_scores()
