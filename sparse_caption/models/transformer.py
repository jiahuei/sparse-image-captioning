# -*- coding: utf-8 -*-
"""
Created on 28 Dec 2020 18:00:01
@author: jiahuei

Based on `The Annotated Transformer`
https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from argparse import ArgumentParser, _ArgumentGroup
from typing import Union, Callable
from copy import deepcopy
from itertools import chain
from . import register_model
from .caption_model import CaptionModel
from ..data.collate import UpDownCollate
from ..utils.model_utils import repeat_tensors, clones
from ..utils.misc import str_to_sequence, str_to_none

logger = logging.getLogger(__name__)


# noinspection PyAbstractClass
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """

    def __init__(
        self,
        encoder: Callable,
        decoder: Callable,
        src_embed: Callable,
        tgt_embed: Callable,
        generator: Callable,
        autoregressive: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.autoregressive = autoregressive
        self.pad_idx = pad_idx

    def forward(self, src: Tensor, src_mask: Tensor, tgt: Tensor):
        """
        Args:
            src: (N, S, E)
            src_mask: (N, S)
            tgt: (N, T)
        Returns:
        """
        memory, memory_mask = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, memory, memory_mask)
        outputs = self.generator(decoder_output)
        return outputs

    def encode(self, src: Tensor, src_mask: Tensor):
        """
        Args:
            src: (N, S, E)
            src_mask: (N, S)
        Returns:
        """
        assert (
            src_mask.ndimension() == 2
        ), f"{self.__class__.__name__}: Expected `src_mask` has shape (N, S), saw `{src_mask.shape}`"
        src_mask = src_mask.unsqueeze(-2)

        src = self.src_embed(src)
        encoder_output = self.encoder(x=src, mask=src_mask)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{self.__class__.__name__}: "
                f"src.shape = `{src.shape}`    "
                f"src_mask.shape = `{src_mask.shape}`    "
                f"encoder_output.shape = `{encoder_output.shape}`    "
            )
        return encoder_output, src_mask

    def decode(self, tgt: Tensor, memory: Tensor, memory_mask: Tensor):
        """
        Args:
            tgt: (N, T)
            memory: (N, S, E)
            memory_mask: (N, S)
        Returns:
        """
        assert tgt.ndimension() == 2, f"{self.__class__.__name__}: Expected `tgt` has shape (N, T), saw `{tgt.shape}`"
        if memory.size(0) != tgt.size(0):
            assert tgt.size(0) % memory.size(0) == 0
            seq_per_img = int(tgt.size(0) / memory.size(0))
            memory, memory_mask = repeat_tensors(seq_per_img, (memory, memory_mask))

        tgt_mask = tgt.ne(self.pad_idx).unsqueeze(-2)
        if self.autoregressive:
            subsequent_mask = memory.new_ones((1, tgt.size(-1), tgt.size(-1)))
            subsequent_mask = torch.triu(subsequent_mask, diagonal=1).eq(0)
            tgt_mask = tgt_mask & subsequent_mask
        tgt_embed = self.tgt_embed(tgt)
        decoder_output = self.decoder(x=tgt_embed, memory=memory, src_mask=memory_mask, tgt_mask=tgt_mask)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{self.__class__.__name__}: "
                f"tgt.shape = `{tgt.shape}`    "
                f"tgt_mask.shape = `{tgt_mask.shape}`    "
                f"tgt_embed.shape = `{tgt_embed.shape}`    "
                f"memory.shape = `{memory.shape}`    "
                f"memory_mask.shape = `{memory_mask.shape}`    "
                f"decoder_output.shape = `{decoder_output.shape}`    "
            )
        return decoder_output

    def generate(self, x):
        return self.generator(x)


# noinspection PyAbstractClass
class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, N, share_layer=None):
        super().__init__()
        if share_layer:
            if not isinstance(share_layer, (tuple, list)):
                raise TypeError(f"`share_layer` must be a tuple or list, saw {type(share_layer)}")
            layers = [deepcopy(layer) for _ in range(len(set(share_layer)))]
            layers = [layers[i] for i in share_layer]
        else:
            layers = [deepcopy(layer) for _ in range(N)]
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# noinspection PyAbstractClass
class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# noinspection PyAbstractClass
class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N, share_layer=None):
        super().__init__()
        if share_layer:
            if not isinstance(share_layer, (tuple, list)):
                raise TypeError(f"`share_layer` must be a tuple or list, saw {type(share_layer)}")
            layers = [deepcopy(layer) for _ in range(len(set(share_layer)))]
            layers = [layers[i] for i in share_layer]
        else:
            layers = [deepcopy(layer) for _ in range(N)]
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# noinspection PyAbstractClass
class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# noinspection PyAbstractClass
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, self_attention=False, share_att=None):
        """Take in model size and number of heads."""
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.self_attention = self_attention
        assert share_att in (None, "kv", "qk"), f"Invalid `share_att`: {share_att}"
        self.share_att = share_att
        self.linears = clones(nn.Linear(d_model, d_model), 3 if share_att else 4)
        self.dropout = nn.Dropout(p=dropout)
        self.cache = [None, None]
        self.cache_size = 2

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self._project_qkv(self.linears[0], query)
        # Maybe need to repeat cache along batch dim
        if isinstance(self.cache[0], Tensor) and self.cache[0].size(0) != key.size(0):
            cache_batch = self.cache[0].size(0)
            assert cache_batch < key.size(0), (
                f"cat_output_with_cache: "
                f"Expected dim {0} of cached tensor to be smaller than that of key. "
                f"Saw self.cache[0] = {self.cache[0].size()}, key = {key.size()}"
            )
            assert key.size(0) % cache_batch == 0, (
                f"cat_output_with_cache: "
                f"Expected dim {0} of key tensor to be divisible by that of cached tensor. "
                f"Saw self.cache[0] = {self.cache[0].size()}, key = {key.size()}"
            )
            self.cache = repeat_tensors(key.size(0) // cache_batch, self.cache)

        # Only encoder-attention may skip projection and directly reuse from cache
        if not self.self_attention and isinstance(self.cache[0], Tensor):
            key, value = self.cache
        else:
            if self.share_att == "qk":
                key = self._project_qkv(self.linears[0], key)
                value = self._project_qkv(self.linears[1], value)
            else:
                key = self._project_qkv(self.linears[1], key)
                value = key if self.share_att else self._project_qkv(self.linears[2], value)

        if self.self_attention and isinstance(self.cache[0], Tensor):
            # Concat with previous keys and values
            key = torch.cat((self.cache[0], key), dim=2)
            value = torch.cat((self.cache[1], value), dim=2)
            mask = None

        # Cache key and value tensors
        if getattr(self, "incremental_decoding", False):
            self.cache = [key, value]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def _project_qkv(self, layer, x):
        return layer(x).view(x.size(0), -1, self.h, self.d_k).transpose(1, 2)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """Compute 'Scaled Dot Product Attention'"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


# noinspection PyAbstractClass
class CachedMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, *args, **kwargs):
        """Take in model size and number of heads."""
        super().__init__(*args, **kwargs)
        self.incremental_decoding = False

    def reset_cache(self):
        self.cache = [None, None]


# Aliases
MHA = MultiHeadedAttention
CMHA = CachedMultiHeadedAttention


# noinspection PyAbstractClass
class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# noinspection PyAbstractClass
class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# noinspection PyAbstractClass
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


# noinspection PyAbstractClass
class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.incremental_decoding = False
        self.current_time_step = 0

    def reset_cache(self):
        self.current_time_step = 0

    def forward(self, x):
        if self.incremental_decoding:
            assert x.size(1) == 1, f"{self.__class__.__name__}: Expected input to have shape (M, 1, N), saw {x.shape}"
            x = x + self.pe[:, self.current_time_step : self.current_time_step + 1]
            self.current_time_step += 1
        else:
            x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# noinspection PyAbstractClass
class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# noinspection PyAbstractClass
class OutputEmbedding(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# noinspection PyAbstractClass,PyAttributeOutsideInit
class CachedTransformerBase(CaptionModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.share_att_encoder = config.share_att_encoder
        self.share_att_decoder = config.share_att_decoder
        self.share_layer_encoder = config.share_layer_encoder
        self.share_layer_decoder = config.share_layer_decoder
        self.d_model = config.d_model  # default: 512
        self.dim_feedforward = config.dim_feedforward  # default: 2048
        self.num_layers = config.num_layers  # default: 6
        self.num_heads = config.num_heads  # default: 8
        self.drop_prob_src = config.drop_prob_src
        self.seq_length = config.max_seq_length
        self.att_feat_size = config.att_feat_size
        self.vocab_size = config.vocab_size
        self.eos_idx = config.eos_token_id
        self.bos_idx = config.bos_token_id
        self.unk_idx = config.unk_token_id
        self.pad_idx = config.pad_token_id
        assert self.num_layers > 0, "num_layers should be greater than 0"

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def enable_incremental_decoding(module):
        if hasattr(module, "incremental_decoding"):
            module.incremental_decoding = True
            module.reset_cache()

    @staticmethod
    def disable_incremental_decoding(module):
        if hasattr(module, "incremental_decoding"):
            module.incremental_decoding = False
            module.reset_cache()

    def _modules_with_cache(self):
        return filter(lambda x: getattr(x, "incremental_decoding", False) and hasattr(x, "cache"), self.modules())

    def _retrieve_caches(self):
        caches = [m.cache for m in self._modules_with_cache()]
        caches = [_.transpose(0, 1) for _ in chain.from_iterable(caches)]
        return caches

    def _update_caches(self, caches):
        idx = 0
        for i, m in enumerate(self._modules_with_cache()):
            m.cache = [_.transpose(0, 1) for _ in caches[idx : idx + m.cache_size]]
            idx += m.cache_size

    def _generate_captions(self, att_feats, att_masks, memory, state, opt):
        num_random_sample = opt.get("num_random_sample", 0)
        beam_size = opt.get("beam_size", 1)
        temperature = opt.get("temperature", 1.0)
        decoding_constraint = opt.get("decoding_constraint", 0)
        batch_size = att_feats.shape[0]

        # Enable incremental decoding for faster decoding
        self.apply(self.enable_incremental_decoding)

        if num_random_sample <= 0 and beam_size > 1:
            assert beam_size <= self.vocab_size
            it = att_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
            seq_logprobs = att_feats.new_zeros(batch_size, beam_size, self.seq_length)
            seq = att_feats.new_full((batch_size, beam_size, self.seq_length), self.pad_idx, dtype=torch.long)

            # first step, feed bos
            logprobs, state = self.get_logprobs_state(it, memory, att_masks, state)
            memory, att_masks = repeat_tensors(beam_size, [memory, att_masks])
            self.done_beams = self.batch_beam_search(state, logprobs, memory, att_masks, opt=opt)

            for k in range(batch_size):
                for b in range(beam_size):
                    res = self.done_beams[k][b]
                    seq_len = res["seq"].shape[0]
                    seq[k, b, :seq_len] = res["seq"]
                    seq_logprobs[k, b, :seq_len] = res["logps"].gather(1, res["seq"].unsqueeze(1)).squeeze(1)
                # top_seq = self.done_beams[k][0]["seq"]
                # seq_len = top_seq.shape[0]
                # seq[k, :seq_len] = top_seq  # the first beam has highest cumulative score
                # seq_logprobs[k, :seq_len] = self.done_beams[k][0]["logps"].gather(1, top_seq.unsqueeze(1)).squeeze(1)
            # Disable incremental decoding so that regular training can continue
            self.apply(self.disable_incremental_decoding)
            # return the samples and their log likelihoods
            return seq, seq_logprobs

        # Greedy search or random sample
        if num_random_sample > 0:
            assert beam_size < 1, f"Beam size must be < 1, saw {beam_size}"
            batch_size *= num_random_sample
            memory = memory.repeat_interleave(num_random_sample, dim=0)
            att_masks = att_masks.repeat_interleave(num_random_sample, dim=0)
        else:
            assert beam_size == 1, f"Beam size must be 1, saw {beam_size}"

        it = att_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        seq_logprobs = att_feats.new_zeros(batch_size, self.seq_length)
        seq = att_feats.new_full((batch_size, self.seq_length), self.pad_idx, dtype=torch.long)

        unfinished = it != self.eos_idx
        for t in range(self.seq_length + 1):
            logprobs, state = self.get_logprobs_state(it, memory, att_masks, state)
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(batch_size, self.vocab_size)
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float("-inf"))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if num_random_sample > 0:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sample_logprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing
            else:
                # greedy search
                sample_logprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()

            # stop when all finished
            seq[:, t] = it * unfinished.type_as(it)
            unfinished = unfinished * (it != self.eos_idx)
            seq_logprobs[:, t] = sample_logprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        if num_random_sample > 0:
            seq = seq.view(-1, num_random_sample, self.seq_length)
            seq_logprobs = seq_logprobs.view(-1, num_random_sample, self.seq_length)
        else:
            seq = seq.view(-1, 1, self.seq_length)
            seq_logprobs = seq_logprobs.view(-1, 1, self.seq_length)
        # Disable incremental decoding so that regular training can continue
        self.apply(self.disable_incremental_decoding)
        return seq, seq_logprobs

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        parser.add_argument(
            "--d_model", type=int, default=512,
            help="int: The token and feature embedding size."
        )
        parser.add_argument(
            "--dim_feedforward", type=int, default=2048,
            help="int: Size of feedforward layers."
        )
        parser.add_argument(
            "--num_layers", type=int, default=6,
            help="int: Number of transformer layers."
        )
        parser.add_argument(
            "--num_heads", type=int, default=8,
            help="int: Number of transformer attention heads."
        )
        parser.add_argument(
            "--drop_prob_src", type=float, default=0.5,
            help="float: Dropout rate applied to source embedding at the Encoder."
        )
        parser.add_argument(
            "--att_feat_size", type=int, default=2048,
            help="int: Number of channels of CNN features (ResNet = 2048, VGG = 512)."
        )
        parser.add_argument(
            "--share_att_encoder", type=str_to_none, default=None,
            help="str: One of `kv`, `qk`. Defaults to None (no weight sharing)."
        )
        parser.add_argument(
            "--share_att_decoder", type=str_to_none, default=None,
            help="str: One of `kv`, `qk`. Defaults to None (no weight sharing)."
        )
        parser.add_argument(
            "--share_layer_encoder", type=str_to_sequence, default=None,
            help=(
                "str: Layer sharing scheme. "
                "For example: (0, 1, 1, 0) specifies 4 layers with weight sharing for L0-L3, L1-L2. "
                "Defaults to None (no weight sharing)."
            )
        )
        parser.add_argument(
            "--share_layer_decoder", type=str_to_sequence, default=None,
            help=(
                "str: Layer sharing scheme. "
                "For example: (0, 1, 1, 0) specifies 4 layers with weight sharing for L0-L3, L1-L2. "
                "Defaults to None (no weight sharing)."
            )
        )
        # fmt: on


# noinspection PyAbstractClass,PyAttributeOutsideInit
@register_model("transformer")
class Transformer(CachedTransformerBase):
    COLLATE_FN = UpDownCollate

    def __init__(self, config):
        super().__init__(config)
        self.make_model()

    def make_model(self):
        dropout = 0.1

        ff = PositionwiseFeedForward(self.d_model, self.dim_feedforward, dropout)
        position = PositionalEncoding(self.d_model, dropout)
        self.core = EncoderDecoder(
            src_embed=nn.Sequential(
                nn.Linear(self.att_feat_size, self.d_model), nn.ReLU(), nn.Dropout(self.drop_prob_src)
            ),
            encoder=Encoder(
                EncoderLayer(
                    self.d_model,
                    MHA(self.num_heads, self.d_model, share_att=self.share_att_encoder),
                    deepcopy(ff),
                    dropout,
                ),
                self.num_layers,
                share_layer=self.share_layer_encoder,
            ),
            tgt_embed=nn.Sequential(InputEmbedding(self.d_model, self.vocab_size), deepcopy(position)),
            decoder=Decoder(
                DecoderLayer(
                    self.d_model,
                    CMHA(self.num_heads, self.d_model, self_attention=True, share_att=self.share_att_decoder),
                    CMHA(self.num_heads, self.d_model, share_att=self.share_att_decoder),
                    deepcopy(ff),
                    dropout,
                ),
                self.num_layers,
                share_layer=self.share_layer_decoder,
            ),
            generator=OutputEmbedding(self.d_model, self.vocab_size),
            autoregressive=True,
            pad_idx=self.pad_idx,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _forward(self, att_feats: Tensor, att_masks: Tensor, seqs: Tensor, **kwargs):
        """
        Args:
            att_feats: (N, S, E)
            att_masks: (N, S)
            seqs: (N, T, E)
        Returns:
        """
        if seqs is not None:
            # crop the last one
            seqs = seqs[:, :-1]
        return self.core(src=att_feats, src_mask=att_masks, tgt=seqs)

    def get_logprobs_state(self, it, memory, mask, state):
        """
        state = [ys.unsqueeze(0)]
        """
        ys = it.unsqueeze(1)
        if state is None:
            pass
        else:
            # Retrieve reordered cache from state, and update them
            self._update_caches(state[1:])
        # noinspection PyUnresolvedReferences
        decoder_output = self.core.decode(tgt=ys, memory=memory, memory_mask=mask)
        logprobs = self.core.generate(decoder_output[:, -1])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{self.__class__.__name__}: "
                f"it.shape = `{it.shape}`    "
                f"ys.shape = `{ys.shape}`    "
                f"len(state) = `{len(state) if state is not None else None}`    "
                f"decoder_output.shape = `{decoder_output.shape}`    "
                f"logprobs.shape = `{logprobs.shape}`    "
            )
        # Add layer cache into state list, transposed so that beam_step can reorder them
        return logprobs, [ys.unsqueeze(0)] + self._retrieve_caches()

    def _sample(self, att_feats: Tensor, att_masks: Tensor, opt=None, **kwargs):
        if opt is None:
            opt = {}
        memory, att_masks = self.core.encode(src=att_feats, src_mask=att_masks)
        state = None
        return self._generate_captions(att_feats, att_masks, memory, state, opt)

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        Transformer.COLLATE_FN.add_argparse_args(parser)
        CachedTransformerBase.add_argparse_args(parser)
        # fmt: on
