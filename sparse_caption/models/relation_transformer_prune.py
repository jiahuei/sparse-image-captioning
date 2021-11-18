# -*- coding: utf-8 -*-
"""
Created on 09 Oct 2020 17:27:20
@author: jiahuei
"""
import logging
import torch.nn as nn
from copy import deepcopy
from argparse import ArgumentParser, _ArgumentGroup
from typing import Union
from . import register_model
from . import relation_transformer as rtrans
from ..utils.model_utils import clones
from ..pruning.masked_layer import MaskedLinear, MaskedEmbedding
from ..pruning.prune import PruningMixin

logger = logging.getLogger(__name__)


# noinspection PyAbstractClass
class EncoderDecoder(PruningMixin, rtrans.EncoderDecoder):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    pass


# noinspection PyAbstractClass
class Generator(rtrans.Generator):
    """Define standard linear + softmax generation step."""

    def __init__(self, mask_type, mask_init_value, d_model, vocab):
        nn.Module.__init__(self)
        self.proj = MaskedLinear(d_model, vocab, mask_type, mask_init_value)


# noinspection PyAbstractClass
class CachedMultiHeadedAttention(rtrans.CachedMultiHeadedAttention):
    def __init__(self, mask_type, mask_init_value, h, d_model, dropout=0.1 / 3, self_attention=False, share_att=None):
        """Take in model size and number of heads."""
        nn.Module.__init__(self)
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.self_attention = self_attention
        assert share_att in (None, "kv", "qk"), f"Invalid `share_att`: {share_att}"
        self.share_att = share_att
        self.linears = clones(MaskedLinear(d_model, d_model, mask_type, mask_init_value), 3 if share_att else 4)
        self.dropout = nn.Dropout(p=dropout)
        self.cache = [None, None]
        self.cache_size = 2


# noinspection PyAbstractClass
class BoxMultiHeadedAttention(rtrans.BoxMultiHeadedAttention):
    """
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    """

    def __init__(
        self, mask_type, mask_init_value, h, d_model, trigonometric_embedding=True, dropout=0.1 / 3, share_att=None
    ):
        """Take in model size and number of heads."""
        nn.Module.__init__(self)

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
        assert share_att in (None, "kv", "qk"), f"Invalid `share_att`: {share_att}"
        self.share_att = share_att
        self.linears = clones(MaskedLinear(d_model, d_model, mask_type, mask_init_value), 3 if share_att else 4)
        self.WGs = clones(MaskedLinear(geo_feature_dim, 1, mask_type, mask_init_value, bias=True), self.h)

        # self.attn = None
        self.dropout = nn.Dropout(p=dropout)


# noinspection PyAbstractClass
class PositionwiseFeedForward(rtrans.PositionwiseFeedForward):
    """Implements FFN equation."""

    def __init__(self, mask_type, mask_init_value, d_model, d_ff, dropout=0.1 / 3):
        nn.Module.__init__(self)
        self.w_1 = MaskedLinear(d_model, d_ff, mask_type, mask_init_value)
        self.w_2 = MaskedLinear(d_ff, d_model, mask_type, mask_init_value)
        self.dropout = nn.Dropout(dropout)
        self.cache_output = False
        self.cache = None


# noinspection PyAbstractClass
class Embeddings(rtrans.Embeddings):
    def __init__(self, mask_type, mask_init_value, d_model, vocab):
        nn.Module.__init__(self)
        self.lut = MaskedEmbedding(vocab, d_model, mask_type, mask_init_value)
        self.d_model = d_model
        self.cache_output = False
        self.cache = None


# noinspection PyAbstractClass,PyAttributeOutsideInit
@register_model("relation_transformer_prune")
class RelationTransformerModel(PruningMixin, rtrans.RelationTransformerModel):
    def __init__(self, config):
        super().__init__(mask_type=config.prune_type, mask_freeze_scope=config.prune_mask_freeze_scope, config=config)

    def make_model(self, h=8, dropout=0.1 / 3):
        """Helper: Construct a model from hyperparameters."""
        mask_type = self.config.prune_type
        mask_init_value = self.config.prune_supermask_init

        bbox_attn = BoxMultiHeadedAttention(
            mask_type, mask_init_value, h, self.d_model, self.box_trigonometric_embedding
        )
        attn = CachedMultiHeadedAttention(mask_type, mask_init_value, h, self.d_model)
        self_attn = deepcopy(attn)
        self_attn.self_attention = True
        ff = PositionwiseFeedForward(mask_type, mask_init_value, self.d_model, self.dim_feedforward, dropout)
        position = rtrans.PositionalEncoding(self.d_model, dropout)
        model = EncoderDecoder(
            mask_type=mask_type,
            mask_freeze_scope="",
            encoder=rtrans.Encoder(
                rtrans.EncoderLayer(self.d_model, deepcopy(bbox_attn), deepcopy(ff), dropout), self.num_layers
            ),
            decoder=rtrans.Decoder(
                rtrans.DecoderLayer(self.d_model, self_attn, attn, deepcopy(ff), dropout), self.num_layers
            ),
            src_embed=lambda x: x,
            tgt_embed=nn.Sequential(
                Embeddings(mask_type, mask_init_value, self.d_model, self.vocab_size), deepcopy(position)
            ),
            generator=Generator(mask_type, mask_init_value, self.d_model, self.vocab_size),
        )
        self.att_embed = nn.Sequential(
            MaskedLinear(
                self.att_feat_size,
                self.d_model,
                mask_type,
                mask_init_value,
            ),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_src),
        )
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.all_weights(named=False):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.model = model

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        rtrans.RelationTransformerModel.add_argparse_args(parser)
        PruningMixin.add_argparse_args(parser)
        # fmt: on
        # return parser
