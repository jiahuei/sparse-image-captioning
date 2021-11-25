# -*- coding: utf-8 -*-
"""
Created on 28 Aug 2020 17:34:10
@author: jiahuei
"""
import os
import torch
from time import time
from sparse_caption.opts import parse_opt
from sparse_caption.utils import losses, optim
from sparse_caption.utils.config import Config
from sparse_caption.utils.misc import configure_logging, replace_from_right
from sparse_caption.utils.model_utils import set_seed, map_to_cuda, densify_state_dict
from sparse_caption.utils.file import dump_json
from sparse_caption.utils.training import TrainingModule
from sparse_caption.pruning import prune


# noinspection PyAttributeOutsideInit
class CaptioningModel(TrainingModule):
    def __init__(self, config: Config):
        # assert config.prune_type in prune.VALID_MASKS, \
        #     f"`config.prune_type` must be one of {prune.VALID_MASKS}, saw `{config.prune_type}`"
        # assert config.caption_model.endswith("_prune")
        super().__init__(config)

    def train(self):
        self.prepare()
        config = self.config
        model = self.model
        tb_writer = self.tb_summary_writer
        batch_size = self.train_loader.batch_size

        # Assure in training mode
        map_to_cuda(model)
        model.train()
        # Save init weights for Lottery Ticket
        torch.save(
            model.state_dict_dense(discard_pruning_mask=True, prune_weights=False), self.checkpoint_path.format("init")
        )

        if config.label_smoothing > 0:
            loss_fn = losses.LabelSmoothing(smoothing=config.label_smoothing)
        else:
            loss_fn = losses.LanguageModelCriterion()
        scst_loss_fn = losses.RewardCriterion()

        # noinspection PyDictCreation
        model_params = {"breakdown": {n: p.nelement() for n, p in model.all_weights(named=True)}}
        model_params["total"] = sum(model_params["breakdown"].values())
        model_params["trainable params"] = model.total_weight_params
        dump_json(
            os.path.join(config.log_dir, "model_params.json"),
            model_params,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        # noinspection PyDictCreation
        mask_params = {"breakdown": {n: p.nelement() for n, p in model.all_pruning_masks(named=True)}}
        mask_params["total"] = sum(mask_params["breakdown"].values())
        mask_params["trainable params"] = model.total_mask_params
        dump_json(
            os.path.join(config.log_dir, "mask_params.json"), mask_params, indent=2, sort_keys=True, ensure_ascii=False
        )

        optim_params = [{"params": list(model.all_weights(named=False))}]
        print(f"Model trainable params (excluding pruning masks): {model.total_weight_params:,d}")
        if model.mask_type in prune.SUPER_MASKS:
            optim_params += [
                {
                    "params": list(model.active_pruning_masks(named=False)),
                    "lr": config.prune_supermask_lr,
                    "weight_decay": 0,
                    "eps": 1e-2,
                    "pruning_mask": True,
                }
            ]
            print(f"Model trainable params (pruning masks): {model.total_mask_params:,d}")
        else:
            print(f"Model params (pruning masks): {model.total_mask_params:,d}")
        optimizer = self.optimizer = optim.get_optim(optim_params, config)

        # Maybe load model
        self.maybe_load_checkpoint(strict=False)

        # Some pruning methods prune before training / fine-tuning begins
        if model.mask_type == prune.SNIP:
            assert config.prune_snip_grad_accum > 0, "`prune_snip_grad_accum` should be greater than 0"
            optimizer.zero_grad()
            for batch_idx, data in enumerate(self.train_loader):
                if batch_idx == config.prune_snip_grad_accum:
                    logger.debug(f"{self.__class__.__name__}: SNIP: Accumulated gradients across {batch_idx} batches.")
                    break
                data = map_to_cuda(data)
                loss = loss_fn(model(**data), data["seqs"][:, 1:], data["masks"][:, 1:])
                loss.backward()
        if model.mask_type in prune.MAG_HARD + prune.LOTTERY + [prune.SNIP]:
            if not prune.SNIP and not config.model_restored:
                logger.warning(
                    f"{self.__class__.__name__}: Pruning ({model.mask_type}): "
                    f"Pruning a randomly initialized model without restoring from checkpoint."
                )
            if model.mask_type != prune.LOTTERY_MASK_FREEZE:
                model.update_masks_once(sparsity_target=config.prune_sparsity_target)
            # Clear gradient after SNIP pruning
            optimizer.zero_grad()
        if model.mask_type in prune.LOTTERY:
            _mask_names = set(tuple(zip(*model.all_pruning_masks()))[0])
            init_ckpt = os.path.join(config.start_from, "model_init.pth")
            init_state_dict = {k: v for k, v in torch.load(init_ckpt).items() if k not in _mask_names}
            missing_keys, unexpected_keys = model.load_state_dict(init_state_dict, strict=False)
            assert len(unexpected_keys) == 0, f"Checkpoint `{init_ckpt}` contains extra parameters."
            # assert len(missing_keys) > 0 and set(missing_keys) == set(_mask_names), \
            #     f"Checkpoint `{init_ckpt}` contains pruning masks."
            logger.info(f"{self.__class__.__name__}: Model weights loaded from `{init_ckpt}`")

        if not self._reached_sparsity_target() and model.mask_type == prune.MASK_FREEZE:
            logger.warning(
                f"{self.__class__.__name__}: "
                f"Mask type is {model.mask_type} but provided sparsity target is too high or too low."
            )

        t_start = t_start_epoch = time()
        for epoch in range(config.max_epochs):
            # If start self critical training
            if 0 <= config.scst_start_epoch <= epoch:
                sc_flag = True
            else:
                sc_flag = False

            for batch_idx, data in enumerate(self.train_loader):
                data = map_to_cuda(data)
                optimizer.zero_grad()
                if not sc_flag:
                    loss = loss_fn(model(**data), data["seqs"][:, 1:], data["masks"][:, 1:])
                    reward = sc_sample = sc_greedy = None
                else:
                    loss, reward, sc_sample, sc_greedy = self.compute_scst_loss(
                        data, gts=data["gts"], loss_fn=scst_loss_fn
                    )
                caption_loss = loss.item()
                if model.mask_type == prune.REGULAR:
                    loss += model.compute_sparsity_loss(
                        config.prune_sparsity_target,
                        weight=config.prune_supermask_sparsity_weight,
                        current_step=self.global_step,
                        max_step=self.max_train_step,
                    )

                loss.backward()
                optim.clip_gradient(optimizer, config.grad_clip)
                optimizer.step(epoch=epoch)
                config.current_lr = optimizer.rate()
                train_loss = loss.item()
                self.global_step += 1

                if model.mask_type in prune.MAG_ANNEAL:
                    prune_start = int((1 / config.max_epochs) * self.max_train_step)  # start of 2nd epoch
                    prune_freq = 1000
                    prune_steps = int((0.50 * self.max_train_step - prune_start) / prune_freq)
                    model.update_masks_gradual(
                        sparsity_target=config.prune_sparsity_target,
                        current_step=self.global_step,
                        start_step=prune_start,
                        prune_steps=prune_steps,
                        prune_frequency=prune_freq,
                    )

                # Console log
                if self.global_step % 5 == 0:
                    num_ex = batch_size * 5 * (1 if sc_flag else config.seq_per_img)
                    t_taken, t_start = time() - t_start, time()
                    eta = (len(self.train_loader) * config.max_epochs - self.global_step) * (t_taken / 5) / 3600
                    log_str = (
                        f"Epoch {epoch:3d} iter {self.global_step:9,d} "
                        f"({(batch_idx + 1) / len(self.train_loader) * 100:5.1f} %), "
                        f"Speed = {num_ex / t_taken:4.0f} ex/sec, ETA = {eta:5.1f} hr, "
                        f"LR = {optimizer.rate():.2e}"
                    )
                    if not sc_flag:
                        print(f"{log_str}, Loss = {train_loss:6.3f}")
                    else:
                        print(f"{log_str}, Avg reward = {reward.mean():6.3f}, Avg baseline = {sc_greedy.mean():.2f}")

                # Write the training loss summary
                if self.global_step % config.losses_log_every == 0:
                    tb_writer.add_scalar("train/loss", train_loss, self.global_step)
                    tb_writer.add_scalar("train/caption_loss", caption_loss, self.global_step)
                    if model.mask_type == prune.REGULAR:
                        for k, v in model.sparsity_loss.items():
                            tb_writer.add_scalar(f"sparsity_loss/{k}", v.item(), self.global_step)
                    tb_writer.add_scalar("train/learning_rate", config.current_lr, self.global_step)
                    sparsity, _, tensor_sps, tensor_names = self.model.all_mask_sparsities
                    tb_writer.add_scalar("prune/sparsity/all", sparsity.item(), self.global_step)
                    for sps, n in zip(tensor_sps, tensor_names):
                        tb_writer.add_scalar(f"prune/sparsity/{n}", sps.item(), self.global_step)
                    tb_writer.add_scalar("prune/all_mask_avg", self.model.all_mask_avg.item(), self.global_step)
                    if sc_flag:
                        tb_writer.add_scalar("train/avg_reward", reward.mean(), self.global_step)
                        tb_writer.add_scalar("train/avg_baseline", sc_greedy.mean(), self.global_step)

                # Evaluate on validation set, and save model
                if self.global_step % config.save_checkpoint_every == 0 or self.global_step == self.max_train_step:
                    predictions, scores, _ = self.eval_on_split(self.val_loader, split="val")

                    # Write validation result into summary
                    for k, v in scores.items():
                        tb_writer.add_scalar(f"val/{k}", v, self.global_step)

                    checkpoint_path = self.checkpoint_path.format("last")
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    torch.save(optimizer.state_dict(), self.optimizer_path.format("last"))

                    # Save model if :
                    # - is improving on validation result
                    # - reaches targeted sparsity
                    val_score = scores["CIDEr"]
                    if self._reached_sparsity_target() and val_score > self.best_val_score:
                        self.best_val_score = val_score
                        torch.save(model.state_dict(), self.checkpoint_path.format("best"))
                        torch.save(optimizer.state_dict(), self.optimizer_path.format("best"))
                        config.best_global_step = self.global_step
                    config.save_config()
            print(f"\nEpoch {epoch} took {int((time() - t_start_epoch) / 60)} minutes\n")
            t_start_epoch = time()
            epoch += 1
        # Prune after training is completed
        self.maybe_prune_best_model()

    def _reached_sparsity_target(self, tolerance=0.05):
        """
        Default: Check if model NNZ is within 5% of target NNZ.
        Also logs current sparsity level in config.
        """
        config = self.config
        with torch.no_grad():
            config.current_sparsity = self.model.all_mask_sparsities[0].item()
        model_nnz = 1.0 - config.current_sparsity
        target_nnz = 1.0 - config.prune_sparsity_target
        nnz_gap = abs(target_nnz - model_nnz) / target_nnz
        target_reached = nnz_gap <= tolerance
        print(
            f"\nCurrent sparsity = {config.current_sparsity * 100:.3f}    "
            f"Target = {config.prune_sparsity_target * 100:.3f}    "
            f"Target reached = {target_reached}\n"
        )
        return target_reached

    def maybe_prune_best_model(self):
        ckpt_path = self.checkpoint_path.format("best")
        if not os.path.isfile(ckpt_path):
            return False
        model = self.model
        model.load_state_dict(torch.load(ckpt_path))
        logger.info(f"{self.__class__.__name__}: Model weights loaded from `{ckpt_path}`")
        map_to_cuda(model)

        # Prune weights and calculate sparsities
        with torch.no_grad():
            model.prune_weights()
            overall_sparsity, overall_nnz, tensor_sps, tensor_names = model.all_mask_sparsities
            overall_sparsity = float(overall_sparsity)
            overall_nnz = int(overall_nnz)
        logger.info(f"{self.__class__.__name__}: Model weights pruned: Overall sparsity = {overall_sparsity * 100:.2f}")
        # Save model weights
        torch.save(
            model.state_dict_sparse(discard_pruning_mask=True, prune_weights=False),
            self.checkpoint_path.format("best_pruned_sparse"),
        )
        torch.save(
            model.state_dict_dense(discard_pruning_mask=True, prune_weights=False),
            self.checkpoint_path.format("best_pruned"),
        )
        if model.mask_type == prune.REGULAR:
            torch.save(
                model.state_dict_dense(discard_pruning_mask=False, prune_weights=False, binarize_supermasks=True),
                self.checkpoint_path.format("best_bin_mask"),
            )
        logger.info(
            f"{self.__class__.__name__}: Model weights saved to: "
            f"\n{self.checkpoint_path.format('best_pruned_sparse')}"
            f"\n{self.checkpoint_path.format('best_pruned')}"
        )
        # Dump sparsity stats
        with open(os.path.join(self.config.log_dir, "sparsities.csv"), "w") as f:
            out_str = f"sparsity,nnz,{','.join(tensor_names)}\n"
            out_str += f"{overall_sparsity:.5f},{overall_nnz},{','.join(f'{_:.5f}' for _ in tensor_sps)}"
            f.write(out_str)
        return True

    @classmethod
    def eval_test(cls, log_dir):
        config = Config.load_config_json(os.path.join(log_dir, "config.json"))
        assert config.caption_model.endswith("_prune")
        config.caption_model = replace_from_right(config.caption_model, "_prune", "", 1)
        ckpt_path = os.path.join(log_dir, "model_best_pruned_sparse.pth")
        state_dict = densify_state_dict(torch.load(ckpt_path))
        logger.info(f"{cls.__name__}: Model weights loaded from `{ckpt_path}`")
        return super().eval_model(state_dict=state_dict, config=config, split="test")


def main(config: Config):
    set_seed(config.seed)
    if config.prune_type in prune.SUPER_MASKS:
        if config.prune_supermask_sparsity_weight < 0:
            if config.caption_model == "up_down_lstm_prune":
                config.prune_supermask_sparsity_weight = max(5.0, 0.5 / (1 - config.prune_sparsity_target))
            else:
                config.prune_supermask_sparsity_weight = max(5.0, 1.5 / (1 - config.prune_sparsity_target))
        config.log_dir += f"__wg_{config.prune_supermask_sparsity_weight:.1f}"
    model = CaptioningModel(config)
    model.train()
    CaptioningModel.eval_test(config.log_dir)


if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parse_opt()
    logger = configure_logging(args.logging_level)
    main(Config(**vars(args)))
