# -*- coding: utf-8 -*-
"""
@author: ruotianluo
"""
import logging
import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from copy import deepcopy
from typing import Dict, Type, Callable, Any, Union, List, Tuple

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    assert isinstance(seed, int)
    # set Random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(seed)
    logger.debug(f"RNG seed set to {seed}.")


def filter_model_inputs(
        input_dict: Dict,
        mode: str,
        required_keys: Union[List, Tuple],
        forward_keys: Union[List, Tuple]
):
    required_keys = tuple(required_keys)
    if mode == "forward":
        required_keys += tuple(forward_keys)
    try:
        inputs = {k: input_dict[k] for k in required_keys}
    except (KeyError, TypeError):
        raise ValueError(
            f"When calling the model in `{mode}` mode, "
            f"first argument `input_dict` must be a dict with keys `{required_keys}`. "
            f"Received input of type: {type(input_dict)}"
        )
    return inputs


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


def find_beam_parent_index(previous: torch.Tensor, current: torch.Tensor):
    """
    Tensors must have shape (batch, beam, time, ...)
    """
    assert previous.dim() > 2, f"`previous` must have at least 3 dims (batch, beam, N). Saw {previous.shape}"
    assert current.dim() > 2, f"`current` must have at least 3 dims (batch, beam, N). Saw {current.shape}"

    batch, beam, ct = current.shape[:3]
    pt = previous.size(2)
    if ct != pt:
        assert ct - pt == 1, (
            f"Shape mismatch between `current` and `previous`. "
            f"current = {current.shape}"
            f"previous = {previous.shape}"
        )
        # current = current.index_select(time_dim, index=torch.arange(previous.size(time_dim)))
        current = current[:, :, :-1, ...]
    match = (previous.unsqueeze(1) == current.unsqueeze(2)).all(dim=-1)
    loc = match.nonzero(as_tuple=False)[:, -1]
    loc = loc.view(batch, beam)
    offset = torch.arange(batch).unsqueeze(1) * beam
    loc = (loc + offset).view(-1)
    return loc


# def compute_incremental_output(
#         cache: Union[torch.Tensor, None], output: torch.Tensor,
#         batch_beam_dim: int, time_dim: int
# ):
#     if isinstance(cache, torch.Tensor):
#         # 1st step of beam search will have batch_beam_dim of (batch)
#         # 2nd step onwards will have batch_beam_dim of (batch * beam)
#         if cache.size(batch_beam_dim) != output.size(batch_beam_dim):
#             assert cache.size(batch_beam_dim) < output.size(batch_beam_dim), (
#                 f"cat_output_with_cache: "
#                 f"Expected dim {batch_beam_dim} of cached tensor to be smaller than that of output. "
#                 f"Saw cache = {cache.size()}, output = {output.size()}"
#             )
#             assert output.size(batch_beam_dim) % cache.size(batch_beam_dim) == 0, (
#                 f"cat_output_with_cache: "
#                 f"Expected dim {batch_beam_dim} of output tensor to be divisible by that of cached tensor. "
#                 f"Saw cache = {cache.size()}, output = {output.size()}"
#             )
#             cache = repeat_tensors(output.size(batch_beam_dim) // cache.size(batch_beam_dim), cache)
#         output = torch.cat((cache, output), dim=time_dim)
#     cache = output.clone()
#     return output, cache


def map_structure_recursive(
        structure: Any,
        func: Callable,
        end_type: Type[torch.Tensor] = torch.Tensor
):
    """
    Applies `func` to elements of structure recursively.
    Args:
        structure: A `Tensor`, or a potentially nested structure containing `Tensor`.
        func: A single argument function.
        end_type: The type of element that `func` expects. Defaults to `torch.Tensor`.
    Returns:
        The same structure but with `func` applied.
    """
    error_mssg = (
        f"Expected `structure` to be either a `tuple`, `list` or `dict`, "
        f"received {type(structure)} instead."
    )
    if isinstance(structure, end_type):
        return func(structure)
    elif structure is None:
        return None
    elif isinstance(structure, tuple):
        return tuple(map_structure_recursive(item, func, end_type) for item in structure)
    elif isinstance(structure, list):
        return list(map_structure_recursive(item, func, end_type) for item in structure)
    elif isinstance(structure, dict):
        return {key: map_structure_recursive(value, func, end_type) for key, value in structure.items()}
    else:
        raise TypeError(error_mssg)


def count_nonzero(tensor, dim=None):
    return (tensor != 0).sum(dim=dim)


def densify_state_dict(state_dict):
    # noinspection PyUnresolvedReferences
    state_dict = {
        k: v.to_dense()
        if isinstance(v, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)) else v
        for k, v in state_dict.items()
    }
    return state_dict


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return (logprobs / modifier)


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


# bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is',
#                'are', 'am', 'the']


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)
