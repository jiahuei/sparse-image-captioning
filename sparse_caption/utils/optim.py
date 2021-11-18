# -*- coding: utf-8 -*-
"""
Created on 16 Sep 2020 15:01:35
@author: jiahuei
"""
import logging
import math
import torch
from torch import optim

logger = logging.getLogger(__name__)


# noinspection PyAttributeOutsideInit
class RateOpt:
    """Optim wrapper that implements rate."""

    def step(self, step=None, epoch=None):
        """Update parameters and rate"""
        self._step += 1
        self._epoch = epoch
        rate = self.rate()
        for p in self.optimizer.param_groups:
            if "pruning_mask" in p:
                logger.debug("Pruning masks encountered. Skip LR setting.")
                continue
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


class NoamOpt(RateOpt):
    """Optim wrapper that implements rate."""

    def __init__(self, optimizer, model_size, factor, warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def rate(self):
        """Implement `lrate` above"""
        step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class StepLROpt(RateOpt):
    """Optim wrapper that implements rate."""

    def __init__(
        self,
        optimizer,
        learning_rate_init,
        learning_rate_decay_start,
        learning_rate_decay_every,
        learning_rate_decay_rate,
    ):
        if learning_rate_decay_start >= 0:
            assert (
                learning_rate_decay_every > 0
            ), f"`learning_rate_decay_every` must be > 0, saw {learning_rate_decay_every}"
            assert (
                0 < learning_rate_decay_rate < 1
            ), f"`learning_rate_decay_rate` must be > 0 and < 1, saw {learning_rate_decay_rate}"
        self.optimizer = optimizer
        self.learning_rate_init = learning_rate_init
        self.learning_rate_decay_start = learning_rate_decay_start
        self.learning_rate_decay_every = learning_rate_decay_every
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self._rate = 0
        self._step = 0
        self._epoch = 0

    def rate(self):
        """Implement `lrate` above"""
        # Assign the learning rate
        if self._epoch > self.learning_rate_decay_start >= 0:
            frac = (self._epoch - self.learning_rate_decay_start) // self.learning_rate_decay_every
            decay_factor = self.learning_rate_decay_rate ** frac
            current_lr = self.learning_rate_init * decay_factor
        else:
            current_lr = self.learning_rate_init
        return current_lr


class CosineOpt(RateOpt):
    """Optim wrapper that implements rate."""

    def __init__(self, optimizer, max_train_step, learning_rate_init, learning_rate_min):
        self.optimizer = optimizer
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=max_train_step, eta_min=learning_rate_min, last_epoch=-1
        # )
        self._step = 0
        self._rate = 0
        self.max_train_step = max_train_step
        self.learning_rate_min = learning_rate_min
        self.learning_rate_init = learning_rate_init

    def rate(self):
        """Implement `lrate` above"""
        step = self._step / self.max_train_step
        step = 1.0 + math.cos(min(1.0, step) * math.pi)
        lr = (self.learning_rate_init - self.learning_rate_min) * (step / 2) + self.learning_rate_min
        return lr


ALL_SCHEDULERS = ("noam", "step", "cosine")


def get_optim(parameters, config):
    scheduler_name = config.lr_scheduler.lower()
    if scheduler_name == "noam":
        if config.optim.lower() != "adam":
            logger.warning(f"Noam scheduler should be used with ADAM. Ignoring optim choice: {config.optim}")
        return NoamOpt(
            torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9),
            model_size=config.d_model,
            factor=config.noamopt_factor,
            warmup=config.noamopt_warmup,
        )
    elif scheduler_name == "step":
        return StepLROpt(
            build_optimizer(parameters, config),
            config.learning_rate,
            config.learning_rate_decay_start,
            config.learning_rate_decay_every,
            config.learning_rate_decay_rate,
        )
    elif scheduler_name == "cosine":
        return CosineOpt(
            build_optimizer(parameters, config),
            config.max_train_step,
            config.learning_rate,
            config.learning_rate_min,
        )
    else:
        raise Exception(f"Bad option `config.lr_scheduler`: {config.lr_scheduler}")


ALL_OPTIMIZERS = ("rmsprop", "adagrad", "sgd", "sgdm", "sgdmom", "adam")


def build_optimizer(params, config):
    optimizer_name = config.optim.lower()
    if optimizer_name == "rmsprop":
        return optim.RMSprop(
            params, config.learning_rate, config.optim_alpha, config.optim_epsilon, weight_decay=config.weight_decay
        )
    elif optimizer_name == "adagrad":
        return optim.Adagrad(params, config.learning_rate, weight_decay=config.weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(params, config.learning_rate, weight_decay=config.weight_decay)
    elif optimizer_name == "sgdm":
        return optim.SGD(params, config.learning_rate, config.optim_alpha, weight_decay=config.weight_decay)
    elif optimizer_name == "sgdmom":
        return optim.SGD(
            params, config.learning_rate, config.optim_alpha, weight_decay=config.weight_decay, nesterov=True
        )
    elif optimizer_name == "adam":
        return optim.Adam(
            params,
            config.learning_rate,
            (config.optim_alpha, config.optim_beta),
            config.optim_epsilon,
            weight_decay=config.weight_decay,
        )
    else:
        raise Exception(f"Bad option `config.optim`: {config.optim}")


# def set_lr(optimizer, lr):
#     for group in optimizer.param_groups:
#         group["lr"] = lr
#
#
# def get_lr(optimizer):
#     for group in optimizer.param_groups:
#         return group["lr"]


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        # for param in group["params"]:
        #     param.grad.data.clamp_(-grad_clip, grad_clip)
        torch.nn.utils.clip_grad_value_(group["params"], grad_clip)
