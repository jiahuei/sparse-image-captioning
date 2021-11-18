# -*- coding: utf-8 -*-
"""
@author: jiahuei, ruotianluo
"""
import logging
import random
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from copy import deepcopy
from typing import Any, Callable

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


def set_seed(seed: int):
    assert isinstance(seed, int)
    # set Random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(seed)
    logger.debug(f"RNG seed set to {seed}.")


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def repeat_tensors(n, x, dim=0):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        # x = x.unsqueeze(1)  # Bx1x...
        # x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        # x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
        x = torch.repeat_interleave(x, repeats=n, dim=dim)
    # elif type(x) is list or type(x) is tuple:
    elif isinstance(x, (list, tuple)):
        x = [repeat_tensors(n=n, x=_, dim=dim) for _ in x]
    return x


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def sequence_from_numpy(sequence):
    if isinstance(sequence[0], torch.Tensor):
        return sequence
    elif isinstance(sequence[0], np.ndarray):
        return [torch.from_numpy(_) for _ in sequence]
    else:
        raise TypeError(f"Expected `torch.Tensor` or `numpy.ndarray`, saw `{type(sequence[0])}`.")


def reorder_beam(tensor: torch.Tensor, beam_idx: torch.Tensor, beam_dim: int = 0):
    return tensor.index_select(beam_dim, beam_idx)


def map_recursive(x: Any, func: Callable):
    """
    Applies `func` to elements of x recursively.
    Args:
        x: An item or a potentially nested structure of tuple, list or dict.
        func: A single argument function.
    Returns:
        The same x but with `func` applied.
    """
    if isinstance(x, tuple):
        return tuple(map_recursive(item, func) for item in x)
    elif isinstance(x, list):
        return list(map_recursive(item, func) for item in x)
    elif isinstance(x, dict):
        return {key: map_recursive(value, func) for key, value in x.items()}
    else:
        return func(x)


def to_cuda(x: Any):
    if USE_CUDA:
        if isinstance(x, torch.Tensor):
            x = x.cuda(non_blocking=True)
        elif isinstance(x, nn.Module):
            x.cuda()
    return x


def map_to_cuda(x: Any):
    return map_recursive(x, to_cuda)


def count_nonzero(tensor):
    return tensor.ne(0).float().sum()


def densify_state_dict(state_dict):
    # noinspection PyUnresolvedReferences
    state_dict = {
        k: v.to_dense()
        if isinstance(v, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor, torch.cuda.sparse.HalfTensor))
        else v
        for k, v in state_dict.items()
    }
    return state_dict


def penalty_builder(penalty_config):
    if penalty_config == "":
        return lambda x, y: y
    pen_type, alpha = penalty_config.split("_")
    alpha = float(alpha)
    if pen_type == "wu":
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == "avg":
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.0):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = ((5 + length) ** alpha) / ((5 + 1) ** alpha)
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.0):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def sort_pack_padded_sequence(inputs, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(inputs[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(inputs, inv_ix):
    tmp, _ = pad_packed_sequence(inputs, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
