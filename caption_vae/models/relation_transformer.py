##########################################################
# Copyright 2019 Oath Inc.
# Licensed under the terms of the MIT license.
# Please see LICENSE file in the project root for terms.
##########################################################

import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from argparse import ArgumentParser, _ArgumentGroup
from typing import Optional, Union, Dict
from copy import deepcopy
from itertools import chain
from models import register_model
from models.caption_model import CaptionModel
from data.collate import ObjectRelationCollate
from utils.model_utils import repeat_tensors, pack_wrapper, clones, filter_model_inputs
from utils.misc import str_to_bool

logger = logging.getLogger(__name__)


# noinspection PyAbstractClass
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, boxes, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        enc_out = self.encode(src, boxes, src_mask)
        assert enc_out.size(0) == src_mask.size(0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{self.__class__.__name__}: "
                f"Encoder output shape = `{enc_out.shape}`    "
                f"Target shape = `{tgt.shape}`"
            )
        if enc_out.size(0) != tgt.size(0):
            assert tgt.size(0) % enc_out.size(0) == 0
            seq_per_img = int(tgt.size(0) / enc_out.size(0))
            enc_out, src_mask = repeat_tensors(seq_per_img, (enc_out, src_mask))
        return self.decode(enc_out, src_mask, tgt, tgt_mask)

    def encode(self, src, boxes, src_mask):
        return self.encoder(self.src_embed(src), boxes, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# noinspection PyAbstractClass
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# noinspection PyAbstractClass
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, box, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, box, mask)
        return self.norm(x)


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
class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, box, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, box, mask))
        return self.sublayer[1](x, self.feed_forward)


# noinspection PyAbstractClass
class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
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
        self.self_attn.self_attention = True
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
    def __init__(self, h, d_model, dropout=0.1, self_attention=False):
        """Take in model size and number of heads."""
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # self.attn = None
        self.self_attention = self_attention
        self.dropout = nn.Dropout(p=dropout)
        self.incremental_decoding = False
        self.cache = [None, None]
        self.cache_size = 2

    def reset_cache(self):
        self.cache = [None, None]

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
            key = self._project_qkv(self.linears[1], key)
            value = self._project_qkv(self.linears[2], value)

        if self.self_attention and isinstance(self.cache[0], Tensor):
            # Concat with previous keys and values
            key = torch.cat((self.cache[0], key), dim=2)
            value = torch.cat((self.cache[1], value), dim=2)
            mask = None

        # Cache key and value tensors
        if self.incremental_decoding:
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
class BoxMultiHeadedAttention(nn.Module):
    """
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    """

    def __init__(self, h, d_model, trigonometric_embedding=True, dropout=0.1):
        """Take in model size and number of heads."""
        super().__init__()

        assert d_model % h == 0
        self.trigonometric_embedding = trigonometric_embedding

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        if self.trigonometric_embedding:
            self.dim_g = 64
        else:
            self.dim_g = 4
        geo_feature_dim = self.dim_g

        # matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True), 8)

        # self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_query, input_key, input_value, input_box, mask=None):
        """Implements Figure 2 of Relation Network for Object Detection"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = input_query.size(0)

        # tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        relative_geometry_embeddings = self.BoxRelationalEmbedding(
            input_box,
            trigonometric_embedding=self.trigonometric_embedding
        )
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.dim_g)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (input_query, input_key, input_value))
        ]
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [
            ly(flatten_relative_geometry_embeddings).view(box_size_per_head) for ly in self.WGs
        ]
        relative_geometry_weights = torch.cat(relative_geometry_weights_per_head, 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        x, box_attn = self.box_attention(
            query, key, value, relative_geometry_weights, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # # Legacy
        # x = input_value + x

        return self.linears[-1](x)

    @staticmethod
    def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trigonometric_embedding=True):
        """
        Given a tensor with bbox coordinates for detected objects on each batch image,
        this function computes a matrix for each image

        with entry (i,j) given by a vector representation of the
        displacement between the coordinates of bbox_i, and bbox_j

        input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
        output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
        """
        # returns a relational embedding for each pair of bboxes, with dimension = dim_g
        # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

        batch_size = f_g.size(0)

        x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.

        # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
        delta_x = cx - cx.view(batch_size, 1, -1)
        delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
        delta_x = torch.log(delta_x)

        delta_y = cy - cy.view(batch_size, 1, -1)
        delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
        delta_y = torch.log(delta_y)

        delta_w = torch.log(w / w.view(batch_size, 1, -1))
        delta_h = torch.log(h / h.view(batch_size, 1, -1))

        matrix_size = delta_h.size()
        delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
        delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
        delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
        delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

        position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

        if trigonometric_embedding:
            feat_range = torch.arange(dim_g / 8).cuda()
            dim_mat = feat_range / (dim_g / 8)
            dim_mat = 1. / (torch.pow(wave_len, dim_mat))

            dim_mat = dim_mat.view(1, 1, 1, -1)
            position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
            position_mat = 100. * position_mat

            mul_mat = position_mat * dim_mat
            mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
            sin_mat = torch.sin(mul_mat)
            cos_mat = torch.cos(mul_mat)
            embedding = torch.cat((sin_mat, cos_mat), -1)
        else:
            embedding = position_mat
        return embedding

    @staticmethod
    def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None):
        """
        Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
        Follow the implementation in
        https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055
        """

        N = value.size()[:2]
        dim_k = key.size(-1)
        dim_g = box_relation_embds_matrix.size()[-1]

        w_q = query
        w_k = key.transpose(-2, -1)
        w_v = value

        # attention weights
        scaled_dot = torch.matmul(w_q, w_k)
        scaled_dot = scaled_dot / np.sqrt(dim_k)
        if mask is not None:
            scaled_dot = scaled_dot.masked_fill(mask == 0, -1e9)

        # w_g = box_relation_embds_matrix.view(N,N)
        w_g = box_relation_embds_matrix
        w_a = scaled_dot
        # w_a = scaled_dot.view(N,N)

        # multiplying log of geometric weights by feature weights
        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        w_mn = torch.nn.Softmax(dim=-1)(w_mn)
        if dropout is not None:
            w_mn = dropout(w_mn)

        output = torch.matmul(w_mn, w_v)

        return output, w_mn


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
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# noinspection PyAbstractClass
class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
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
            assert x.size(1) == 1, \
                f"{self.__class__.__name__}: Expected input to have shape (M, 1, N), saw {x.shape}"
            x = x + self.pe[:, self.current_time_step:self.current_time_step + 1]
            self.current_time_step += 1
        else:
            x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# noinspection PyAbstractClass,PyAttributeOutsideInit
@register_model("relation_transformer")
class RelationTransformerModel(CaptionModel):
    COLLATE_FN = ObjectRelationCollate

    def make_model(self, h=8, dropout=0.1):
        """Helper: Construct a model from hyperparameters."""
        d_model = self.input_encoding_size
        tgt_vocab = self.vocab_size + 1  # TODO: fix vocab_size, remove + 1

        bbox_attn = BoxMultiHeadedAttention(h, d_model, self.box_trigonometric_embedding)
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, self.rnn_size, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(
                d_model, deepcopy(bbox_attn), deepcopy(ff), dropout), self.num_layers
            ),
            Decoder(DecoderLayer(
                d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout), self.num_layers
            ),
            lambda x: x,  # nn.Sequential(Embeddings(d_model, src_vocab), deepcopy(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), deepcopy(position)),
            Generator(d_model, tgt_vocab)
        )
        self.att_embed = nn.Sequential(
            nn.Linear(self.att_feat_size, self.input_encoding_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm)
        )
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_encoding_size = config.input_encoding_size
        self.rnn_size = config.rnn_size
        self.num_layers = config.num_layers
        self.drop_prob_lm = config.drop_prob_lm
        self.box_trigonometric_embedding = True
        self.seq_length = config.max_seq_length
        self.att_feat_size = config.att_feat_size
        self.vocab_size = config.vocab_size
        self.eos_idx = config.eos_token_id
        self.bos_idx = config.bos_token_id
        self.unk_idx = config.unk_token_id
        self.pad_idx = config.pad_token_id
        # self.eos_idx = self.bos_idx = self.unk_idx = self.pad_idx = 0

        assert self.num_layers > 0, "num_layers should be greater than 0"
        self.model = self.make_model()

    def forward(self, input_dict: Dict, **kwargs):
        inputs = filter_model_inputs(
            input_dict=input_dict,
            mode=kwargs.get("mode", "forward"),
            required_keys=("att_feats", "att_masks", "boxes"),
            forward_keys=("seqs",)
        )
        inputs["fc_feats"] = None
        return super().forward(**inputs, **kwargs)

    def _prepare_feature(
            self,
            att_feats: Tensor,
            att_masks: Optional[Tensor] = None,
            boxes: Optional[Tensor] = None,
            seq: Optional[Tensor] = None
    ):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data != self.pad_idx)  # seq_mask: torch.Tensor
            # noinspection PyUnresolvedReferences
            seq_mask = seq_mask.unsqueeze(-2)
            # noinspection PyUnresolvedReferences
            seq_mask = seq_mask & self.subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, boxes, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, boxes, seqs, att_masks=None):
        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes, seqs)
        out = self.model(att_feats, boxes, seq, att_masks, seq_mask)
        outputs = self.model.generator(out)
        return outputs

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
        return filter(
            lambda x: getattr(x, "incremental_decoding", False) and hasattr(x, "cache"),
            self.modules()
        )

    def _retrieve_caches(self):
        caches = [m.cache for m in self._modules_with_cache()]
        caches = [_.transpose(0, 1) for _ in chain.from_iterable(caches)]
        return caches

    def _update_caches(self, caches):
        idx = 0
        for i, m in enumerate(self._modules_with_cache()):
            m.cache = [_.transpose(0, 1) for _ in caches[idx: idx + m.cache_size]]
            idx += m.cache_size

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
        out = self.model.decode(
            memory, mask, ys, self.subsequent_mask(ys.size(1)).to(memory.device)
        )
        logprobs = self.model.generator(out[:, -1])
        # Add layer cache into state list, transposed so that beam_step can reorder them
        return logprobs, [ys.unsqueeze(0)] + self._retrieve_caches()

    def _sample(self, fc_feats, att_feats, boxes, att_masks=None, opt=None):
        if opt is None:
            opt = {}
        num_random_sample = opt.get("num_random_sample", 0)
        beam_size = opt.get("beam_size", 1)
        temperature = opt.get("temperature", 1.0)
        decoding_constraint = opt.get("decoding_constraint", 0)
        batch_size = att_feats.shape[0]

        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes)
        memory = self.model.encode(att_feats, boxes, att_masks)
        state = None
        # Enable incremental decoding for faster decoding
        self.apply(self.enable_incremental_decoding)

        if num_random_sample <= 0 and beam_size > 1:
            assert beam_size <= self.vocab_size + 1
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
                tmp = logprobs.new_zeros(batch_size, self.vocab_size + 1)
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
    def clip_att(att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    @staticmethod
    def subsequent_mask(size):
        """Mask out subsequent positions."""
        attn_shape = (1, size, size)
        mask = (torch.triu(torch.ones(attn_shape), diagonal=1) == 0)
        return mask

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        # CaptionModel.add_argparse_args(parser)
        # CaptionModel args
        parser.add_argument(
            "--max_seq_length", type=int, default=16,
            help="int: Maximum sequence length excluding <bos> and <eos>.",
        )
        parser.add_argument(
            "--rnn_size", type=int, default=2048,
            help="int: Size of feedforward layers."
        )
        parser.add_argument(
            "--num_layers", type=int, default=6,
            help="int: Number of layers in the model"
        )
        parser.add_argument(
            "--input_encoding_size", type=int, default=512,
            help="int: The encoding size of each token in the vocabulary, and the image."
        )
        parser.add_argument(
            "--att_feat_size", type=int, default=2048,
            help="int: 2048 for resnet, 512 for vgg"
        )
        parser.add_argument(
            "--drop_prob_lm", type=float, default=0.5,
            help="float: Strength of dropout in the Language Model RNN"
        )
        # Relation args
        parser.add_argument(
            "--box_trigonometric_embedding", type=str_to_bool,
            default=True
        )
        # fmt: on
        # return parser
