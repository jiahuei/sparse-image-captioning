# -*- coding: utf-8 -*-
"""
Created on 25 Sep 2020 19:25:43
@author: jiahuei
"""
import os
import logging
import torch
import math
from argparse import ArgumentParser, _ArgumentGroup
from typing import Callable, Union, Dict
from .sampler import rounding_sigmoid
from ..utils.model_utils import count_nonzero, densify_state_dict

logger = logging.getLogger(__name__)

MASK_FREEZE = "mask_freeze"
REGULAR = "supermask"

MAG_BLIND = "mag_blind"
MAG_UNIFORM = "mag_uniform"
MAG_DIST = "mag_dist"

MAG_GRAD_BLIND = "mag_grad_blind"
MAG_GRAD_UNIFORM = "mag_grad_uniform"
MAG_GRAD_DIST = "mag_grad_dist"

LOTTERY_MAG_BLIND = "lottery_mag_blind"
LOTTERY_MAG_UNIFORM = "lottery_mag_uniform"
LOTTERY_MAG_DIST = "lottery_mag_dist"
LOTTERY_MASK_FREEZE = "lottery_mask_freeze"

SNIP = "snip"

# SUPER_MASKS = [REGULAR, SRINIVAS]
SUPER_MASKS = [REGULAR]
MAG_ANNEAL = [MAG_GRAD_BLIND, MAG_GRAD_UNIFORM]
MAG_HARD = [MAG_BLIND, MAG_UNIFORM, MAG_DIST]
LOTTERY = [LOTTERY_MAG_BLIND, LOTTERY_MAG_UNIFORM, LOTTERY_MAG_DIST, LOTTERY_MASK_FREEZE]
MAG_PRUNE_MASKS = MAG_HARD + MAG_ANNEAL + LOTTERY + [SNIP]
# VALID_MASKS = [TRAIN_MASK] + SUPER_MASKS + MAG_PRUNE_MASKS
VALID_MASKS = SUPER_MASKS + MAG_PRUNE_MASKS + [MASK_FREEZE]


# noinspection PyAttributeOutsideInit
class PruningMixin:
    """
    Mixin class to be used together with torch.nn.Module
    """

    named_parameters: Callable
    state_dict: Callable
    load_state_dict: Callable

    def __init__(self, *, mask_type, mask_freeze_scope="", **kwargs):
        assert mask_type in VALID_MASKS, f"`mask_type` must be one of {VALID_MASKS}, saw `{mask_type}`"
        assert isinstance(mask_freeze_scope, str), f"`mask_freeze_scope` must be a str, saw `{type(mask_freeze_scope)}`"
        self.mask_type = mask_type
        if mask_freeze_scope == "":
            self.mask_freeze_scope = None
        else:
            self.mask_freeze_scope = [_ for _ in mask_freeze_scope.split(",") if _ != ""]
        # self.supermask_requires_grad_set = False
        self.sparsity_target = 0.0
        super().__init__(**kwargs)

    def all_pruning_masks(self, named=True):
        return list(
            (name, param) if named else param
            for name, param in self.named_parameters()
            if name.endswith("_pruning_mask")
        )

    def all_pruned_weights(self, named=True):
        weight_names = set(name.replace("_pruning_mask", "") for name, param in self.all_pruning_masks())
        return list(
            (name, param) if named else param for name, param in self.named_parameters() if name in weight_names
        )

    def all_weights(self, named=True):
        return list(
            (name, param) if named else param
            for name, param in self.named_parameters()
            if not name.endswith("_pruning_mask")
        )

    def active_pruning_masks(self, named=True):
        if self.mask_freeze_scope is None:
            # yield from self.all_pruning_masks(named)
            return self.all_pruning_masks(named)
        else:
            # ret = []
            # for name, param in self.all_pruning_masks():
            #     if any(name.startswith(_) for _ in self.mask_freeze_scope):
            #         continue
            #     # yield (name, param) if named else param
            #     ret.append((name, param) if named else param)
            # return ret
            return [
                (name, param) if named else param
                for name, param in self.all_pruning_masks()
                if not any(name.startswith(_) for _ in self.mask_freeze_scope)
            ]

    def active_pruned_weights(self, named=True):
        pruned_weight_names = set(name.replace("_pruning_mask", "") for name, param in self.active_pruning_masks())
        return list(
            (name, param) if named else param for name, param in self.named_parameters() if name in pruned_weight_names
        )

    def trainable_pruning_masks(self, named=True):
        return list(
            (name, param) if named else param for name, param in self.all_pruning_masks() if param.requires_grad
        )

    @property
    def total_mask_params(self):
        return sum(_.nelement() for _ in self.all_pruning_masks(named=False))

    @property
    def total_weight_params(self):
        return sum(_.nelement() for _ in self.all_weights(named=False))

    @staticmethod
    def calculate_sparsities(tensor_list, count_nnz_fn):
        tensor_nelem = [_.nelement() for _ in tensor_list]
        tensor_nnz = [count_nnz_fn(_) for _ in tensor_list]
        tensor_sps = [1.0 - (nnz / nelem) for nnz, nelem in zip(tensor_nnz, tensor_nelem)]
        total_nnz = sum(tensor_nnz)
        total_size = sum(tensor_nelem)
        total_sparsity = 1.0 - (total_nnz / total_size)
        return total_sparsity, total_nnz, tensor_sps

    @property
    def all_weight_sparsities(self):
        names, weights = zip(*self.all_pruned_weights(named=True))
        return self.calculate_sparsities(weights, count_nonzero) + (names,)

    @property
    def all_mask_sparsities(self):
        names, masks = zip(*self.all_pruning_masks(named=True))
        if self.mask_type in SUPER_MASKS:
            masks = [rounding_sigmoid(_) for _ in masks]
        return self.calculate_sparsities(masks, torch.sum) + (names,)

    @property
    def active_mask_sparsities(self):
        names, masks = zip(*self.active_pruning_masks(named=True))
        if self.mask_type in SUPER_MASKS:
            masks = [rounding_sigmoid(_) for _ in masks]
        return self.calculate_sparsities(masks, torch.sum) + (names,)

    @property
    def all_mask_avg(self):
        masks = self.all_pruning_masks(named=False)
        mask_vec = torch.cat([m.view(-1) for m in masks], dim=0)
        return mask_vec.mean()

    @property
    def active_mask_avg(self):
        masks = self.active_pruning_masks(named=False)
        mask_vec = torch.cat([m.view(-1) for m in masks], dim=0)
        return mask_vec.mean()

    @torch.no_grad()
    def prune_weights(self):
        masks = self.all_pruning_masks(named=False)
        weights = self.all_pruned_weights(named=False)
        if self.mask_type in SUPER_MASKS:
            masks = [rounding_sigmoid(_) for _ in masks]
        for w, m in zip(weights, masks):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.__class__.__name__}: Prune weights: Weight = `{w}`    Mask = `{m}`")
            w[:] = w * m

    def state_dict_dense(
        self,
        destination=None,
        prefix="",
        keep_vars=False,
        discard_pruning_mask=False,
        prune_weights=True,
        binarize_supermasks=False,
    ):
        if prune_weights:
            self.prune_weights()
        state_dict = self.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if discard_pruning_mask and binarize_supermasks:
            raise ValueError("`discard_pruning_mask` and `binarize_supermasks` cannot be True at the same time.")
        if discard_pruning_mask:
            for mask_name, _ in self.all_pruning_masks():
                del state_dict[mask_name]
        if binarize_supermasks:
            if self.mask_type not in SUPER_MASKS:
                raise ValueError(f"`binarize_supermasks` can only be True for mask_type in {SUPER_MASKS}.")
            for mask_name, _ in self.all_pruning_masks():
                state_dict[mask_name] = rounding_sigmoid(state_dict[mask_name])
        return state_dict

    def state_dict_sparse(
        self,
        destination=None,
        prefix="",
        keep_vars=False,
        discard_pruning_mask=True,
        prune_weights=True,
        binarize_supermasks=False,
    ):
        state_dict = self.state_dict_dense(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
            discard_pruning_mask=discard_pruning_mask,
            prune_weights=prune_weights,
            binarize_supermasks=binarize_supermasks,
        )
        all_pruned, _ = zip(*self.all_pruned_weights(named=True))
        all_pruned = set(all_pruned)
        return {
            k: v.to_sparse() if (isinstance(v, torch.Tensor) and k in all_pruned) else v for k, v in state_dict.items()
        }

    def load_sparse_state_dict(
        self, sparse_state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], strict: bool = True
    ):
        self.load_state_dict(state_dict=densify_state_dict(sparse_state_dict), strict=strict)

    def compute_sparsity_loss(self, sparsity_target: float, weight: float, current_step: int, max_step: int):
        """
        Loss for controlling sparsity of Supermasks.
        Args:
            sparsity_target: Desired sparsity rate.
            weight:
            current_step:
            max_step:
        Returns:
            Scalar loss value.
        """
        assert self.mask_type in SUPER_MASKS, f"Invalid mask type. Must be one of {SUPER_MASKS}"
        _, masks = zip(*self.active_pruning_masks(named=True))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.__class__.__name__}: Sparsity loss: Mask list = `{list(_)}`")
        # if self.supermask_requires_grad_set is False:
        #     _, all_masks = zip(*self.all_pruning_masks())
        #     for m in list(set(all_masks) - set(masks)):
        #         m.requires_grad = False
        if len(masks) == 0:
            return 0.0
        sampled_masks = [rounding_sigmoid(_) for _ in masks]
        total_sparsity, _, _ = self.calculate_sparsities(sampled_masks, torch.sum)
        loss = torch.abs(sparsity_target - total_sparsity)
        self.sparsity_loss = {"loss": loss}

        # Anneal
        step = current_step / max_step
        step = 1.0 + torch.cos(torch.tensor(min(1.0, step) * math.pi))
        anneal_rate = step / 2
        loss = loss * weight * (1.0 - anneal_rate)
        self.sparsity_loss["anneal_rate"] = anneal_rate
        self.sparsity_loss["loss_scaled"] = loss

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{self.__class__.__name__}: Sparsity loss: "
                f"Loss = `{loss}`    "
                f"Total sparsity = `{total_sparsity}`    "
                f"Mask avg = `{float(self.active_mask_avg)}`"
            )
        return loss

    @staticmethod
    def compute_mask(criterion, sparsity_target):
        assert (
            isinstance(sparsity_target, float) and 0 <= sparsity_target < 1.0
        ), f"`sparsity_target` must be a float >= 0 and < 1, saw {sparsity_target}"
        mask = torch.ones_like(criterion)
        tensor_size = criterion.nelement()
        prune_amount = int(sparsity_target * tensor_size)
        assert 0 <= prune_amount < tensor_size
        if prune_amount > 0:
            topk = torch.topk(criterion.view(-1), k=prune_amount, largest=False)
            mask.view(-1)[topk.indices] = 0
        return mask

    @torch.no_grad()
    def sparsity_check(self, warning_threshold: float = 0.999):
        _, _, _masks_sps, _masks_names = self.all_mask_sparsities
        _high_sps_masks = [(n, s) for n, s in zip(_masks_names, _masks_sps) if s > warning_threshold]
        if len(_high_sps_masks) > 0:
            logger.warning(
                f"{self.__class__.__name__}: Pruning ({self.mask_type}): "
                f"One or more mask has sparsity > {warning_threshold}:\n"
                f"{'   '.join(f'{n} = {s:.5f}' for n, s in _high_sps_masks)}"
            )

    @torch.no_grad()
    def update_masks_once(self, sparsity_target: float):
        """
        Args:
            sparsity_target:
        Returns:
            True if pruning masks are successfully updated.
        """
        assert (
            self.mask_type in MAG_PRUNE_MASKS
        ), f"Invalid mask_type: {self.mask_type}. Must be one of {MAG_PRUNE_MASKS}"
        _, masks = zip(*self.active_pruning_masks())
        _, weights = zip(*self.active_pruned_weights())
        assert len(weights) == len(masks)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.__class__.__name__}: Pruning ({self.mask_type}): Mask list = `{list(_)}`")

        if self.mask_type == SNIP:
            # mask_ori_shape = [m.shape for m in masks]
            # mask_num_elems = [m.nelement() for m in masks]
            saliency = [_.grad for _ in masks]
            assert all(_ is not None for _ in saliency)
            saliency_vec = torch.cat([s.view(-1) for s in saliency], dim=0)
            criterion = [saliency_vec / saliency_vec.sum()]
            # num_params = saliency_vec.size(0)
            # kappa = int(round(num_params * (1. - sparsity_target)))
            # topk = torch.topk(saliency_vec, k=kappa, largest=True)
            # mask_sparse_vec = torch.zeros_like(saliency_vec).scatter_(dim=0, index=topk.indices, value=1)
            # new_masks = torch.split(mask_sparse_vec, mask_num_elems)
            # # new_masks = [m.view(ms) for m, ms in zip(new_masks, mask_ori_shape)]
            # for m, new_m in zip(masks, new_masks):
            #     m.view(-1)[:] = new_m
            # return True

        elif self.mask_type in (MAG_DIST, MAG_GRAD_DIST, LOTTERY_MAG_DIST):
            # Magnitude pruning, class-distribution
            # Calculate standard dev of each class
            # Transform weights as positive factor of standard dev, ie w' = | (w - mean) / std_dev |
            # Reshape and concat all factorised weights, and calculate threshold
            # The rest of the operations are same as class-blind
            criterion = []
            for w in weights:
                std_dev = torch.std(w.view(-1), dim=0, unbiased=False)
                c = torch.abs((w - w.mean()) / std_dev)
                criterion.append(c)
            criterion = [torch.cat([c.view(-1) for c in criterion], dim=0)]

        else:
            criterion = [torch.abs(w) for w in weights]
            # if self.mask_type in MAG_BLIND + MASK_BLIND:
            if self.mask_type in (MAG_UNIFORM, MAG_GRAD_UNIFORM, LOTTERY_MAG_UNIFORM):
                # Magnitude pruning, class-uniform
                pass
            elif self.mask_type in (MAG_BLIND, MAG_GRAD_BLIND, LOTTERY_MAG_BLIND):
                # Magnitude pruning, class-blind
                # We reshape all the weights into a vector, and concat them
                criterion = [torch.cat([c.view(-1) for c in criterion], dim=0)]
            else:
                raise ValueError(f"Unknown `self.mask_type`: {self.mask_type}")

        # len == 1 for class-blind, and len == len(weights) for others
        new_masks = [self.compute_mask(c, sparsity_target) for c in criterion]
        if len(new_masks) == 1:
            new_masks = torch.split(new_masks[0], [m.nelement() for m in masks])

        assert len(new_masks) == len(
            masks
        ), "Threshold list should be either of length 1 or equal length as masks list."
        for m, new_m in zip(masks, new_masks):
            m.view(-1)[:] = new_m.view(-1)
        logger.info(
            f"{self.__class__.__name__}: Pruning ({self.mask_type}): Pruned to sparsity = `{sparsity_target:.5f}`"
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.__class__.__name__}: Pruning ({self.mask_type}): Updated masks = `{list(masks)}`")
        self.sparsity_target = sparsity_target
        self.sparsity_check()
        return True

    @torch.no_grad()
    def update_masks_gradual(
        # self, si: float, sf: float, t: int, t0: int, tn: int, dt: int = 1000
        self,
        sparsity_target: float,
        current_step: int,
        start_step: int,
        prune_steps: int,
        initial_sparsity: float = 0.0,
        prune_frequency: int = 1000,
    ):
        """
        Get current sparsity level for gradual pruning.
        https://arxiv.org/abs/1710.01878
        https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/model_pruning

        Args:
            sparsity_target: Final sparsity
            current_step: Current global step
            start_step: When to start pruning
            prune_steps: Number of pruning steps to take
            initial_sparsity: Starting sparsity
            prune_frequency: Number of training steps per pruning step

        Returns:
            True if pruning masks are updated at this step, False otherwise.
        """
        # prune_start = int((1 / c.max_epoch) * c.max_step)  # start of 2nd epoch
        # n = int((0.50 * c.max_step - prune_start) / prune_freq)
        # pruning_end = prune_start + prune_freq * n
        t = current_step
        si = initial_sparsity
        sf = sparsity_target
        t0 = start_step
        tn = start_step + prune_frequency * prune_steps
        dt = prune_frequency
        assert self.mask_type in MAG_ANNEAL
        assert dt > 0, f"Pruning frequency must be greater than zero, saw `{dt}`"
        assert prune_steps > 0, f"Pruning steps must be greater than zero, saw `{prune_steps}`"
        assert (tn - t0) % dt == 0, "Pruning end step must be equal to start step added by multiples of frequency."

        is_step_within_pruning_range = (
            (t >= t0)
            and
            # If end_pruning_step is negative, keep pruning forever!
            ((t <= tn) or (tn < 0))
        )
        is_pruning_step = ((t - t0) % dt) == 0
        is_pruning_step = is_step_within_pruning_range and is_pruning_step

        if is_pruning_step:
            # Current sparsity target
            p = (t - t0) / (tn - t0)
            p = min(1.0, max(0.0, p))
            st = sf + ((si - sf) * ((1.0 - p) ** 3))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.__class__.__name__}: Gradual pruning: Sparsity target = `{st}`")
            self.update_masks_once(sparsity_target=st)
        return False

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        parser = parser.add_argument_group(
            "Pruning",
            "Arguments for weight pruning."
        )
        parser.add_argument(
            "--prune_type", type=str, default="",
            choices=VALID_MASKS,
            help="str: Type of pruning scheme.",
        )
        parser.add_argument(
            "--prune_sparsity_target", type=float, default=0.8,
            help="float: Desired sparsity."
        )
        parser.add_argument(
            "--prune_mask_freeze_scope", type=str, default="",
            help="str: Scopes to freeze pruning masks."
        )
        parser.add_argument(
            "--prune_snip_grad_accum", type=int, default=1,
            help="int: Number of batches of gradient accumulation for SNIP saliency computation."
        )
        parser.add_argument(
            "--prune_supermask_init", type=float, default=5.,
            help="float: Init value of Supermask pruning masks."
        )
        parser.add_argument(
            "--prune_supermask_sparsity_weight", type=float, default=-1.,
            help="float: Weightage of Supermask sparsity loss."
        )
        parser.add_argument(
            "--prune_supermask_lr", type=float, default=1e2,
            help="float: Learning rate for Supermask."
        )
        parser.add_argument(
            "--prune_supermask_bypass_sigmoid_grad", action="store_true",
            help="bool: If True, bypass sigmoid during gradient backprop (straight-through estimator)."
        )
        # fmt: on
        # return parser
