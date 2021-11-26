# -*- coding: utf-8 -*-
"""
Created on 01 May 2020 15:00:54
@author: jiahuei

https://github.com/huggingface/transformers/blob/v2.9.0/examples/lightning_base.py
"""
import os
import logging
import json
import torch
from time import perf_counter
from tqdm import tqdm
from argparse import ArgumentParser, _ArgumentGroup
from typing import Dict, Callable, Union
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from .config import Config
from .misc import csv_to_float_list, ROOT_DIR
from .optim import ALL_OPTIMIZERS, ALL_SCHEDULERS
from .model_utils import map_to_cuda
from ..data import get_dataset, KarpathyDataset
from ..tokenizer import get_tokenizer, Tokenizer
from ..models import get_model
from ..coco_caption.eval import evaluate_caption_json
from ..scst.scorers import CaptionScorer

logger = logging.getLogger(__name__)


# noinspection PyAttributeOutsideInit
class TrainingModule:
    """
    Base class for training and evaluation.
    """

    ALL_METRICS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
    SCST_SAMPLE = ["beam_search", "random"]
    SCST_BASELINE = ["greedy", "sample"]
    config: Config
    data: KarpathyDataset
    collate_fn: Dict[str, Callable]
    model: nn.Module
    optimizer: torch.optim.Optimizer
    tokenizer: Tokenizer
    scst_scorer: CaptionScorer
    checkpoint_path = str

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        os.makedirs(config.log_dir, exist_ok=True)
        # Get dataset
        self.data = get_dataset(config.dataset)(config)
        self.data.prepare_data()
        # Get tokenizer, train and save
        self.tokenizer = get_tokenizer(config.tokenizer)(config)
        logger.info(f"{self.__class__.__name__}: Vocab size = {config.vocab_size}")
        logger.info(
            f"{self.__class__.__name__}: Token IDs: "
            f"BOS = {config.bos_token_id}, "
            f"EOS = {config.eos_token_id}, "
            f"UNK = {config.unk_token_id}, "
            f"PAD = {config.pad_token_id}, "
            # f"MASK = {config.mask_token_id}"
        )
        # Build model
        model_cls = get_model(config.caption_model)
        self.model = model_cls(config)
        if config.get("cache_min_free_ram", 1) < 1:
            from multiprocessing import Manager

            manager = Manager()
            cache_dict = manager.dict()
        else:
            cache_dict = None
        self.collate_fn = {
            "train": model_cls.COLLATE_FN(config=config, tokenizer=self.tokenizer, cache_dict=cache_dict),
            "eval": model_cls.COLLATE_FN(config=config, tokenizer=self.tokenizer),
        }
        self.checkpoint_path = os.path.join(config.log_dir, "model_{}.pth")
        self.optimizer_path = os.path.join(config.log_dir, "optimizer_{}.pth")

    def train_dataloader(self):
        logger.debug(f"{self.__class__.__name__}: Setting up dataloader for train split")
        return self.get_dataloader("train", collate_fn=self.collate_fn["train"], generation_mode=True)

    def val_dataloader(self):
        logger.debug(f"{self.__class__.__name__}: Setting up dataloader for validation split")
        return self.get_dataloader("val", collate_fn=self.collate_fn["eval"], generation_mode=True)

    def test_dataloader(self):
        logger.debug(f"{self.__class__.__name__}: Setting up dataloader for test split")
        return self.get_dataloader("test", collate_fn=self.collate_fn["eval"], generation_mode=True)

    def get_dataloader(self, split: str, collate_fn: Callable, generation_mode: bool = False):
        if split not in ("train", "val", "test"):
            error_mssg = f"Invalid split `{split}`, please pass in one of ('train', 'val', 'test')."
            raise ValueError(error_mssg)
        is_training = split == "train"
        if is_training:
            batch_size = self.config.batch_size
        else:
            batch_size = self.config.get("batch_size_eval", self.config.batch_size)

        data_loader = DataLoader(
            dataset=self.data.get_split(split, generation_mode),
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=is_training,
        )
        return data_loader

    def prepare(self):
        config = self.config

        assert config.max_epochs > 0, "`config.max_epochs` should be > 0"
        assert config.beam_size_val > 0, "`config.beam_size_val` should be > 0"
        assert config.save_checkpoint_every > 0, "`config.save_checkpoint_every` should be > 0"
        assert config.losses_log_every > 0, "`config.losses_log_every` should be > 0"
        if config.cached_tokens is None:
            config.cached_tokens = os.path.join(config.dataset_dir, "bu", "coco-train-words")

        self.config_path = self.config.save_config(exist_ok=False)
        self.train_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        # self.test_loader = self.test_dataloader()
        self.tb_summary_writer = SummaryWriter(config.log_dir)
        self.scst_scorer = CaptionScorer(
            config.cached_tokens, cider_weight=config.scst_cider_weight, bleu_weight=config.scst_bleu_weight
        )
        self.global_step = 0
        self.max_train_step = config.max_train_step = config.max_epochs * len(self.train_loader)
        self.best_val_score = 0.0
        config.best_global_step = 0

    def maybe_load_checkpoint(self, strict=True):
        config = self.config
        model = self.model
        if not config.start_from:
            return None
        if os.path.isfile(config.start_from):
            restore_dir = os.path.dirname(config.start_from)
            model_file = config.start_from
        elif os.path.isdir(config.start_from):
            restore_dir = config.start_from
            if config.get("resume_training", False):
                model_file = os.path.join(config.start_from, "model_last.pth")
            else:
                model_file = os.path.join(config.start_from, "model_best.pth")
        else:
            raise ValueError(
                f"{self.__class__.__name__}: "
                f"Argument `start_from` must be either directory path or model checkpoint path."
            )

        # Check if models are compatible
        old_config = Config.load_config_json(os.path.join(restore_dir, "config.json"))
        checklist = ("caption_model", "rnn_type", "rnn_size", "num_layers")
        old_config_vars, config_vars = vars(old_config), vars(config)
        for check in checklist:
            if check in old_config_vars:
                if old_config_vars[check] != config_vars[check]:
                    logger.warning(
                        f"{self.__class__.__name__}: "
                        f"Argument provided and loaded config disagree on `{check}`: "
                        f"Provided: `{config_vars[check]}`    Loaded: `{old_config_vars[check]}`"
                    )
            else:
                if check in config_vars:
                    logger.warning(
                        f"{self.__class__.__name__}: "
                        f"Argument `{check}` is provided but is missing from loaded config."
                    )

        # If resume training, load last model and optimizer checkpoints
        if config.get("resume_training", False):
            opt_file = os.path.join(restore_dir, "optimizer_last.pth")
            self.optimizer.load_state_dict(torch.load(opt_file))
            logger.info(f"{self.__class__.__name__}: Optimizer weights loaded from `{opt_file}`")
            config.optimizer_restored = True

        missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_file), strict=strict)
        logger.info(f"{self.__class__.__name__}: Model weights loaded from `{model_file}`")
        restore_log = os.path.join(config.log_dir, "restore_log.txt")
        if len(missing_keys) > 0:
            _log = f"{self.__class__.__name__}: Checkpoint `{model_file}` is missing one or more parameters"
            with open(restore_log, "a") as f:
                f.write(f"{_log}:\n" + "\n".join(missing_keys) + "\n\n")
            logger.info(f"{_log}. See `{restore_log}` for more info.")
        if len(unexpected_keys) > 0:
            _log = f"{self.__class__.__name__}: Checkpoint `{model_file}` contains extra parameters"
            with open(restore_log, "a") as f:
                f.write(f"{_log}:\n" + "\n".join(unexpected_keys) + "\n\n")
            logger.info(f"{_log}. See `{restore_log}` for more info.")
        config.model_restored = True

    def compute_scst_loss(self, model_inputs, gts, loss_fn):
        config = self.config
        model = self.model
        assert isinstance(model_inputs, dict), f"Expected `model_inputs` to be dict, saw {type(model_inputs)}"
        assert (
            config.scst_num_samples > 0
        ), f"Expected `config.scst_num_samples` to be > 0, saw {config.scst_num_samples}"
        assert (
            config.scst_sample in self.SCST_SAMPLE
        ), f"Expected `config.scst_sample` to be one of `{self.SCST_SAMPLE}`, saw {config.scst_sample}"
        assert (
            config.scst_baseline in self.SCST_BASELINE
        ), f"Expected `config.scst_baseline` to be one of `{self.SCST_BASELINE}`, saw {config.scst_baseline}"

        if config.scst_baseline == "greedy":
            # Greedy decoding baseline
            model.eval()
            with torch.no_grad():
                greedy_res, _ = model(**model_inputs, mode="sample")
        else:
            assert config.scst_baseline == "sample"
            greedy_res = None
        model.train()
        if config.scst_sample == "beam_search":
            sample_res, sample_logprobs = model(
                **model_inputs,
                mode="sample",
                opt={"beam_size": config.scst_num_samples},
            )
        else:
            assert config.scst_sample == "random"
            sample_res, sample_logprobs = model(
                **model_inputs,
                mode="sample",
                opt={"num_random_sample": config.scst_num_samples, "beam_size": 0},
            )

        # Decode into sentences
        if greedy_res is None:
            greedy_decoded = None
        else:
            greedy_decoded = [[self.tokenizer.decode(_[0])] for _ in greedy_res.cpu().numpy()]
        sample_decoded = [[self.tokenizer.decode(__) for __ in _] for _ in sample_res.cpu().numpy()]

        # Compute reward
        if config.scst_baseline == "greedy":
            assert greedy_decoded is not None
        else:
            assert greedy_decoded is None
        sc_sample, sc_baseline = self.scst_scorer(refs=gts, sample=sample_decoded, baseline=greedy_decoded)
        reward = map_to_cuda(torch.from_numpy(sc_sample - sc_baseline).type_as(sample_logprobs))
        mask = sample_res.view(sample_res.size(0) * sample_res.size(1), -1) != model.pad_idx
        loss = loss_fn(sample_logprobs, mask=mask, reward=reward)
        return loss, reward, sc_sample, sc_baseline

    def eval_on_split(self, loader, split):
        assert loader.drop_last is False, "`drop_last` must be False for eval dataloader`."
        config = self.config
        model = self.model
        # Make sure in the evaluation mode
        model.eval()
        config.beam_size = config.get(f"beam_size_{split}", 1)

        t0 = perf_counter()
        image_paths = []
        predictions = []
        for batch_idx, data in enumerate(tqdm(loader, desc="Evaluating model")):
            data = map_to_cuda(data)
            with torch.no_grad():
                seq = model(**data, opt=config, mode="sample")[0]

            predictions += [self.tokenizer.decode(_[0]) for _ in seq]
            image_paths += data["image_paths"]
        print(f"Speed: {len(image_paths) / (perf_counter() - t0):.2f} img/sec")
        # Switch back to training mode
        model.train()

        # Save caption prediction JSON file
        is_test2014_split = config.get("mscoco_online_test", False) and split == "test"
        if is_test2014_split:
            out_dir = os.path.join(config.log_dir, f"test2014_beam_{config.beam_size}")
        else:
            out_dir = os.path.join(config.log_dir, f"{split}_beam_{config.beam_size}")

        if config.get("eval_dir_suffix", None):
            out_dir += f"_{config.eval_dir_suffix}"
        json_fpath = os.path.join(out_dir, f"caption_{self.global_step:08d}.json")
        self.data.coco_caption_json_dump(zip(image_paths, predictions), json_fpath)

        if is_test2014_split:
            val_img_paths = os.listdir(os.path.join(config.dataset_dir, "val2014"))
            fake_preds = ["an example caption" for _ in val_img_paths]
            self.data.coco_caption_json_dump(
                zip(val_img_paths, fake_preds), json_fpath.replace(".json", "_val2014.json")
            )
            scores = None
        else:
            # MS-COCO test2014 split has no GT
            scores, scores_detailed, coco_eval = evaluate_caption_json(
                res_file=json_fpath, ann_file=self.data.ANNOTATION_FILE
            )
            # Save score JSON
            score_fpath = os.path.join(out_dir, f"score_{self.global_step:08d}.json")
            with open(score_fpath, "w") as f:
                json.dump(scores, fp=f, indent=2, sort_keys=True, ensure_ascii=False)
            with open(score_fpath.replace(".json", "_detailed.json"), "w") as f:
                json.dump(scores_detailed, fp=f, indent=2, sort_keys=True, ensure_ascii=False)
            # Save score CSV
            score_csv_fpath = os.path.join(out_dir, "scores.csv")
            if os.path.isfile(score_csv_fpath):
                score_str = ""
            else:
                score_str = f"Step,{','.join(str(k) for k in self.ALL_METRICS)}\n"
            with open(score_csv_fpath, "a") as f:
                score_str += f"{self.global_step:08d},"
                score_str += ",".join(f"{scores[k]:.3f}" for k in self.ALL_METRICS)
                f.write(f"{score_str}\n")

        # if image_root:
        #     # Save cocoEval and any other relevant information into a pickle to be used
        #     # later for generating a report and visualizing results.
        #     report_data = ReportData(cocoEval, preds, image_root, model_id, split)
        #     pickle_file_name = REPORT_DATA_PKL_FILE_TEMPLATE % (model_id, split)
        #     pickle_path = os.path.join(results_dir, pickle_file_name)
        #     report_data.save_to_pickle(pickle_path)
        return predictions, scores, out_dir

    @classmethod
    def eval_model(cls, state_dict, config, split="test"):
        assert isinstance(
            config, Config
        ), f"`config` should be an instance of `utils.config.Config`, saw {type(config)}"
        self = cls(config)
        self.model.load_state_dict(state_dict)
        map_to_cuda(self.model)
        if split == "val":
            self.test_loader = self.val_dataloader()
        elif split == "test":
            self.test_loader = self.test_dataloader()
        else:
            raise ValueError(f"{self.__class__.__name__}: `split` must be one of ('val', 'train'), saw: {split}")
        self.global_step = self.config.get("best_global_step", 0)
        return self.eval_on_split(self.test_loader, split=split)

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        # General
        parser.add_argument(
            "--seed", type=int, default=8888,
            help="int: Random number generator (RNG) seed."
        )
        # parser.add_argument(
        #     "--gpus", type=misc_utils.csv_to_int_list, default=[0],
        #     help="List of ints: List of GPUs. Pass a CSV string.",
        # )
        parser.add_argument(
            "--cache_min_free_ram", type=float, default=0.4,
            help="float: Minimum free RAM when caching training data. Set to 1.0 to disable."
        )
        parser.add_argument(
            "--num_workers", type=int, default=4,
            help="int: Number of workers for each `DataLoader`."
        )
        parser.add_argument(
            "--cached_tokens", type=str, default=None,
            help="str: Cached token file for calculating cider score during self critical training."
        )
        # Checkpoint
        parser.add_argument(
            "--id", type=str, default="",
            help="An id identifying this run/job."
        )
        parser.add_argument(
            "--log_dir", type=str, default=ROOT_DIR,
            help="str: Logging / Saving directory."
        )
        parser.add_argument(
            "--start_from", type=str, default="",
            help="str: Load model parameters from this directory."
        )
        parser.add_argument(
            "--resume_training", action="store_true",
            help="bool: If True, resume training."
        )
        parser.add_argument(
            "--save_checkpoint_every", type=int, default=6000,
            help="int: How often to save a model checkpoint (in iterations)"
        )
        parser.add_argument(
            "--losses_log_every", type=int, default=25,
            help="int: How often to perform Tensorboard dump."
        )
        # Training
        parser.add_argument(
            "--batch_size", type=int, default=15,
            help="int: Batch size."
        )
        parser.add_argument(
            "--batch_size_eval", type=int, default=50,
            help="int: Batch size for evaluation."
        )
        parser.add_argument(
            "--max_epochs", type=int, default=15,
            help="int: Maximum training epoch."
        )
        parser.add_argument(
            "--weight_decay", type=float, default=0,
            help="weight_decay"
        )
        parser.add_argument(
            "--grad_clip", type=float, default=0.1,  # 5.,
            help="clip gradients at this value"
        )
        parser.add_argument(
            "--label_smoothing", type=float, default=0,
            help=""
        )
        parser.add_argument(
            "--optim", type=str, default="adam",
            choices=ALL_OPTIMIZERS,
            help="str: Optimizer name."
        )
        parser.add_argument(
            "--optim_alpha", type=float, default=0.9,
            help="alpha for adam"
        )
        parser.add_argument(
            "--optim_beta", type=float, default=0.999,
            help="beta used for adam"
        )
        parser.add_argument(
            "--optim_epsilon", type=float, default=1e-8,
            help="epsilon that goes into denominator for smoothing"
        )
        parser.add_argument(
            "--lr_scheduler", type=str, default="noam",
            choices=ALL_SCHEDULERS,
            help="str: Scheduler name."
        )
        parser.add_argument(
            "--noamopt_warmup", type=int, default=10000,
            help=""
        )
        parser.add_argument(
            "--noamopt_factor", type=float, default=1,
            help=""
        )
        # parser.add_argument(
        #     "--reduce_on_plateau", action="store_true",
        #     help=""
        # )

        parser.add_argument(
            "--learning_rate", type=float, default=5e-4,
            help="float: Learning rate"
        )
        parser.add_argument(
            "--learning_rate_min", type=float, default=1e-5,
            help="float: Minimum learning rate, used by Cosine Annealing."
        )
        parser.add_argument(
            "--learning_rate_decay_start", type=int, default=0,
            help="int: St which epoch to start decaying learning rate? (-1 = disabled)"
        )
        parser.add_argument(
            "--learning_rate_decay_every", type=int, default=3,
            help="int: Every how many epoch thereafter to drop LR?"
        )
        parser.add_argument(
            "--learning_rate_decay_rate", type=float, default=0.8,
            help="float: every how many epoch thereafter to drop LR?"
        )

        # Optimization: SCST
        parser.add_argument(
            "--scst_start_epoch", type=int, default=-1,
            help="int: Epoch to start SCST, -1 to disable."
        )
        parser.add_argument(
            "--scst_num_samples", type=int, default=10,
            help="int: Number of samples per example for SCST, must be > 0."
        )
        parser.add_argument(
            "--scst_sample", type=str, default="random",
            choices=TrainingModule.SCST_SAMPLE,
            help="str: SCST sampling method."
        )
        parser.add_argument(
            "--scst_baseline", type=str, default="sample",
            choices=TrainingModule.SCST_BASELINE,
            help="str: SCST baseline method."
        )
        parser.add_argument(
            "--scst_cider_weight", type=float, default=1.,
            help="float: The reward weight from CIDEr-D."
        )
        parser.add_argument(
            "--scst_bleu_weight", type=csv_to_float_list, default=(0., 0., 0., 0.),
            help="str: Comma-separated reward weights from BLEU-1 to BLEU-4."
        )
        # Eval
        parser.add_argument(
            "--beam_size_test", type=int, default=2,
            help="int: Beam size used for test set."
        )
        parser.add_argument(
            "--beam_size_val", type=int, default=1,
            help="int: Beam size used for validation set."
        )
        # fmt: on
        # return parser
