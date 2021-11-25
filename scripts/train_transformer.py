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
from sparse_caption.utils.misc import configure_logging
from sparse_caption.utils.model_utils import set_seed, map_to_cuda
from sparse_caption.utils.file import dump_json
from sparse_caption.utils.training import TrainingModule


# noinspection PyAttributeOutsideInit
class CaptioningModel(TrainingModule):
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
        torch.save(model.state_dict(), self.checkpoint_path.format("init"))

        if config.label_smoothing > 0:
            loss_fn = losses.LabelSmoothing(smoothing=config.label_smoothing)
        else:
            loss_fn = losses.LanguageModelCriterion()
        scst_loss_fn = losses.RewardCriterion()

        trainable_params = sum(_.nelement() for _ in model.parameters() if _.requires_grad)
        # noinspection PyDictCreation
        model_params = {"breakdown": {n: p.nelement() for n, p in model.named_parameters()}}
        model_params["total"] = sum(model_params["breakdown"].values())
        model_params["trainable params"] = trainable_params
        dump_json(
            os.path.join(config.log_dir, "model_params.json"),
            model_params,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        print(f"Model trainable params: {trainable_params:,d}")
        optimizer = self.optimizer = optim.get_optim(model.parameters(), config)

        # Maybe load model
        self.maybe_load_checkpoint()

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

                loss.backward()
                optim.clip_gradient(optimizer, config.grad_clip)
                optimizer.step(epoch=epoch)
                train_loss = loss.item()
                self.global_step += 1

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
                    config.current_lr = optimizer.rate()
                    tb_writer.add_scalar("train/learning_rate", config.current_lr, self.global_step)
                    # tb_writer.add_scalar("train/mu", model.model.mu.mean().item(), self.global_step)
                    # tb_writer.add_scalar("train/logvar", model.model.logvar.mean().item(), self.global_step)
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

                    # Save model if is improving on validation result
                    val_score = scores["CIDEr"]
                    if val_score > self.best_val_score:
                        self.best_val_score = val_score
                        torch.save(model.state_dict(), self.checkpoint_path.format("best"))
                        torch.save(optimizer.state_dict(), self.optimizer_path.format("best"))
                        config.best_global_step = self.global_step
                    config.save_config()
            print(f"\nEpoch {epoch} took {int((time() - t_start_epoch) / 60)} minutes\n")
            t_start_epoch = time()
            epoch += 1

    @classmethod
    def eval_test(cls, log_dir):
        config = Config.load_config_json(os.path.join(log_dir, "config.json"))
        ckpt_path = os.path.join(log_dir, "model_best.pth")
        state_dict = torch.load(ckpt_path)
        logger.info(f"{cls.__name__}: Model weights loaded from `{ckpt_path}`")
        return super().eval_model(state_dict=state_dict, config=config, split="test")


def main(config: Config):
    set_seed(config.seed)
    model = CaptioningModel(config)
    model.train()
    CaptioningModel.eval_test(config.log_dir)


if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parse_opt()
    logger = configure_logging(args.logging_level)
    main(Config(**vars(args)))
