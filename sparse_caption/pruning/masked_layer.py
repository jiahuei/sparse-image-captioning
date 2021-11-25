# -*- coding: utf-8 -*-
"""
Created on 23 Sep 2020 17:36:39
@author: jiahuei
"""
import logging
import torch
from torch import Tensor
from torch import nn
from torch.nn import init, functional as F
from torch.nn.parameter import Parameter
from copy import deepcopy
from typing import Tuple, List, Union, Optional
from . import prune, sampler

logger = logging.getLogger(__name__)


# noinspection PyAttributeOutsideInit
class MaskMixin:
    mask_type: str
    mask_init_value: float
    mask_trainable: bool
    training: bool

    def setup_masks(
        self,
        parameters: Union[str, List[str], Tuple[str, ...]],
        mask_type: str,
        mask_init_value: float = 1.0,
        bypass_sigmoid_grad: bool = False,
    ) -> None:
        if not isinstance(parameters, (list, tuple)):
            parameters = (parameters,)
        assert all(isinstance(_, str) for _ in parameters)
        self.mask_parameters = []
        for name in parameters:
            weight = getattr(self, name, None)
            assert weight is not None, f"Invalid weight attribute name: {name}"
            if not isinstance(weight, Parameter):
                logger.warning(
                    f"{self.__class__.__name__}: "
                    f"Retrieved weight tensor of type {type(weight)}, converting it into a Parameter."
                )
                weight = Parameter(weight)
            mask_name = f"{name}_pruning_mask"
            setattr(self, mask_name, deepcopy(weight))
            self.mask_parameters.append(getattr(self, mask_name, None))
        assert all(_ is not None for _ in self.mask_parameters)
        assert mask_type in prune.VALID_MASKS, f"`mask_type` must be one of {prune.VALID_MASKS}, saw `{mask_type}`"
        self.mask_type = mask_type

        if self.mask_type in prune.SUPER_MASKS:
            assert isinstance(mask_init_value, (float, int)), "`mask_init_value` must be provided as a float or int."
            self.mask_init_value = float(mask_init_value)
            self.mask_train_sample_fn = lambda x: sampler.bernoulli_sample_sigmoid(x, bypass_sigmoid_grad)
            self.mask_eval_sample_fn = lambda x: sampler.rounding_sigmoid(x, bypass_sigmoid_grad)
            self.mask_trainable = True
        else:
            if mask_init_value is not None:
                logger.info(
                    f"{self.__class__.__name__}: `mask_init_value` is always 1.0 for mask_type = `{self.mask_type}`"
                )
            # Regular pruning
            self.mask_init_value = 1.0
            self.mask_train_sample_fn = self.mask_eval_sample_fn = None
            self.mask_trainable = self.mask_type == prune.SNIP

        for mask in self.mask_parameters:
            mask.requires_grad = self.mask_trainable
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{self.__class__.__name__}: Init: "
                f"mask_type = {mask_type}    "
                f"mask_init_value = {mask_init_value}    "
                f"mask_trainable = {self.mask_trainable}"
            )
        self.reset_masks()

    def reset_masks(self) -> None:
        for mask in self.mask_parameters:
            init.constant_(mask, self.mask_init_value)

    def get_masked_weight(self, weight_name: str):
        # Get weight and its corresponding mask
        weight = getattr(self, weight_name, None)
        assert weight is not None, f"Invalid weight attribute name: {weight_name}"
        mask_name = f"{weight_name}_pruning_mask"
        mask = getattr(self, mask_name, None)
        assert mask is not None, f"Invalid weight attribute name: {mask_name}"
        # TODO: consider caching sampled mask for reuse, and clear cache when sparsity_loss is called
        if self.mask_type in prune.SUPER_MASKS:
            if self.training:
                sample_fn = self.mask_train_sample_fn
            else:
                sample_fn = self.mask_eval_sample_fn
            sampled_mask = sample_fn(mask)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.__class__.__name__}: Mask type = {self.mask_type}    Sample fn = {sample_fn}")
        else:
            sampled_mask = mask
        masked_weight = sampled_mask * weight
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{self.__class__.__name__}: "
                f"Mask type = {self.mask_type}    "
                f"Sampled mask = {sampled_mask}    "
                f"Masked weight = {masked_weight}"
            )
        return masked_weight

    @staticmethod
    def assert_in_kwargs(key, kwargs):
        assert key in kwargs, f"{key} not found in provided keyword arguments: {kwargs}"


# noinspection PyAbstractClass
class MaskedLinear(MaskMixin, nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`"""
    __constants__ = nn.Linear.__constants__ + ["mask_type", "mask_init_value", "bypass_sigmoid_grad"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_type: str,
        mask_init_value: float,
        bypass_sigmoid_grad: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, **kwargs)
        self.setup_masks("weight", mask_type, mask_init_value, bypass_sigmoid_grad)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.get_masked_weight("weight"), self.bias)


# noinspection PyAbstractClass
class MaskedEmbedding(MaskMixin, nn.Embedding):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.
    """
    __constants__ = nn.Embedding.__constants__ + ["mask_type", "mask_init_value", "bypass_sigmoid_grad"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mask_type: str,
        mask_init_value: float,
        bypass_sigmoid_grad: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.setup_masks("weight", mask_type, mask_init_value, bypass_sigmoid_grad)

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(
            input,
            self.get_masked_weight("weight"),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor."""
        raise NotImplementedError


# noinspection PyAbstractClass
class MaskedLSTMCell(MaskMixin, nn.LSTMCell):
    r"""
    A masked long short-term memory (LSTM) cell.
        self.weight_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mask_type: str,
        mask_init_value: float,
        bypass_sigmoid_grad: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(input_size, hidden_size, **kwargs)
        self.setup_masks(("weight_ih", "weight_hh"), mask_type, mask_init_value, bypass_sigmoid_grad)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], "[0]")
        self.check_forward_hidden(input, hx[1], "[1]")
        return torch._VF.lstm_cell(
            input,
            hx,
            self.get_masked_weight("weight_ih"),
            self.get_masked_weight("weight_hh"),
            self.bias_ih,
            self.bias_hh,
        )


# noinspection PyAbstractClass
class MaskedLSTMCellCheckpoint(MaskMixin, nn.LSTMCell):
    r"""
    A masked long short-term memory (LSTM) cell.
        self.weight_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mask_type: str,
        mask_init_value: float,
        bypass_sigmoid_grad: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(input_size, hidden_size, **kwargs)
        self.setup_masks(("weight_ih", "weight_hh"), mask_type, mask_init_value, bypass_sigmoid_grad)

    def _lstm(self, input, hx0, hx1):
        return torch._VF.lstm_cell(
            input,
            (hx0, hx1),
            self.get_masked_weight("weight_ih"),
            self.get_masked_weight("weight_hh"),
            self.bias_ih,
            self.bias_hh,
        )

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], "[0]")
        self.check_forward_hidden(input, hx[1], "[1]")
        return torch.utils.checkpoint.checkpoint(
            self._lstm,
            input,
            *hx,
        )
