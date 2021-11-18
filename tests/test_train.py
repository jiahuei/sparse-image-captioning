# -*- coding: utf-8 -*-
"""
Created on 08 Jan 2021 17:39:15
@author: jiahuei
"""
import unittest
import os
from sparse_caption.opts import parse_opt
from sparse_caption.utils.config import Config
from .paths import TEST_DIRPATH, TEST_DATA_DIRPATH


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.common_args = (
            "--dataset mscoco_testing "
            f"--dataset_dir {TEST_DATA_DIRPATH} "
            f"--log_dir {os.path.join(TEST_DIRPATH, 'experiments')} "
            "--learning_rate 0.01 "
            "--optim_epsilon 0.01 "
            "--batch_size 2 "
            "--batch_size_eval 2 "
            "--save_checkpoint_every 2 "
            "--cache_min_free_ram 1.0 "
            "--max_epochs 1 "
            "--vocab_size 10 "
        )
        self.model_args = dict(
            up_down_lstm=(
                "--caption_model up_down_lstm "
                "--id TESTING_UpDownLSTM "
                "--lr_scheduler cosine "
                "--rnn_size 8 "
                "--input_encoding_size 8 "
                "--att_hid_size 8 "
            ),
            transformer=(
                "--caption_model transformer "
                "--id TESTING_Trans "
                "--lr_scheduler noam "
                "--d_model 8 "
                "--dim_feedforward 8 "
                "--num_layers 2 "
            ),
            relation_transformer=(
                "--caption_model relation_transformer "
                "--id TESTING_RTrans "
                "--lr_scheduler noam "
                "--d_model 8 "
                "--dim_feedforward 8 "
                "--num_layers 2 "
            ),
        )
        self.prune_args = dict(
            supermask=(
                "--prune_type supermask " "--prune_sparsity_target 0.9 " "--prune_supermask_sparsity_weight 120 "
            ),
            # mag_grad_uniform=(
            #     "--prune_type mag_grad_uniform "
            #     "--prune_sparsity_target 0.9 "
            # ),
            snip=("--prune_type snip " "--prune_sparsity_target 0.9 "),
            mag_blind=("--prune_type mag_blind " "--prune_sparsity_target 0.9 "),
            mag_uniform=("--prune_type mag_uniform " "--prune_sparsity_target 0.9 "),
            mag_dist=("--prune_type mag_dist " "--prune_sparsity_target 0.9 "),
        )

    def _test_model(self, config, main_fn):
        name = f"{config.caption_model} with prune type: {config.get('prune_type', None)}"
        with self.subTest(f"Training model: {name}"):
            try:
                main_fn(config)
            except FileNotFoundError as e:
                if not ("model_best.pth" in str(e) or "model_best_pruned_sparse.pth" in str(e)):
                    self.fail(f"Training failed: {name}")
            except Exception:
                self.fail(f"Training failed: {name}")

    # noinspection PyTypeChecker
    def test_train_regular(self):
        from scripts.train_transformer import main

        for model_args in self.model_args.values():
            args = self.common_args + model_args
            print(args)
            args = parse_opt(args.split())
            self._test_model(Config(**vars(args)), main)

    # noinspection PyTypeChecker
    def test_train_prune(self):
        from scripts.train_n_prune_transformer import main

        for model, model_args in self.model_args.items():
            if model == "transformer":
                continue
            model_args = model_args.replace(model, f"{model}_prune")
            for prune, prune_args in self.prune_args.items():
                args = self.common_args + model_args + prune_args
                print(args)
                args = parse_opt(args.split())
                args.log_dir += f"_{prune}"
                self._test_model(Config(**vars(args)), main)


if __name__ == "__main__":
    unittest.main()
