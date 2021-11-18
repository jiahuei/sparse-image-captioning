"""
https://github.com/yahoo/object_relation_transformer

##########################################################
# Copyright 2019 Oath Inc.
# Licensed under the terms of the MIT license.
# Please see LICENSE file in the project root for terms.
##########################################################
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from argparse import ArgumentParser, _ArgumentGroup
from typing import Optional, Union
from copy import deepcopy
from . import register_model
from .transformer import (
    CachedMultiHeadedAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    InputEmbedding as Embeddings,
    OutputEmbedding as Generator,
    LayerNorm,
    SublayerConnection,
    Decoder,
    DecoderLayer,
    CachedTransformerBase,
)
from ..data.collate import ObjectRelationCollate
from ..utils.model_utils import repeat_tensors, pack_wrapper, clones

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
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

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

    def forward(self, x, box, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, box, mask)
        return self.norm(x)


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
class BoxMultiHeadedAttention(nn.Module):
    """
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    """

    def __init__(self, h, d_model, trigonometric_embedding=True, dropout=0.1, share_att=None):
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
        assert share_att in (None, "kv", "qk"), f"Invalid `share_att`: {share_att}"
        self.share_att = share_att
        self.linears = clones(nn.Linear(d_model, d_model), 3 if share_att else 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True), self.h)

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
            input_box, trigonometric_embedding=self.trigonometric_embedding
        )
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.dim_g)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        if self.share_att == "kv":
            att_inputs = (input_query, input_key)
        elif self.share_att == "qk":
            att_inputs = (input_query, input_value)
        else:
            att_inputs = (input_query, input_key, input_value)
        att_outputs = (self._project_qkv(l, x) for l, x in zip(self.linears, att_inputs))
        if self.share_att == "kv":
            query, key = att_outputs
            value = key
        elif self.share_att == "qk":
            query, value = att_outputs
            key = self._project_qkv(self.linears[0], input_key)
        else:
            query, key, value = att_outputs
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [
            ly(flatten_relative_geometry_embeddings).view(box_size_per_head) for ly in self.WGs
        ]
        relative_geometry_weights = torch.cat(relative_geometry_weights_per_head, 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        x, box_attn = self.box_attention(query, key, value, relative_geometry_weights, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

    def _project_qkv(self, layer, x):
        return layer(x).view(x.size(0), -1, self.h, self.d_k).transpose(1, 2)

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
        w = (x_max - x_min) + 1.0
        h = (y_max - y_min) + 1.0

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
            feat_range = torch.arange(dim_g / 8, device=f_g.device)
            dim_mat = feat_range / (dim_g / 8)
            dim_mat = 1.0 / (torch.pow(wave_len, dim_mat))

            dim_mat = dim_mat.view(1, 1, 1, -1)
            position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
            position_mat = 100.0 * position_mat

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


# noinspection PyAbstractClass,PyAttributeOutsideInit
@register_model("relation_transformer")
class RelationTransformerModel(CachedTransformerBase):
    COLLATE_FN = ObjectRelationCollate

    def __init__(self, config):
        super().__init__(config)
        self.box_trigonometric_embedding = not config.no_box_trigonometric_embedding
        self.make_model(h=self.num_heads)

    def make_model(self, h=8, dropout=0.1):
        """Helper: Construct a model from hyperparameters."""
        bbox_attn = BoxMultiHeadedAttention(
            h, self.d_model, self.box_trigonometric_embedding, share_att=self.share_att_encoder
        )
        attn = CachedMultiHeadedAttention(h, self.d_model, share_att=self.share_att_decoder)
        self_attn = deepcopy(attn)
        self_attn.self_attention = True
        ff = PositionwiseFeedForward(self.d_model, self.dim_feedforward, dropout)
        position = PositionalEncoding(self.d_model, dropout)
        model = EncoderDecoder(
            Encoder(
                EncoderLayer(self.d_model, deepcopy(bbox_attn), deepcopy(ff), dropout),
                self.num_layers,
                share_layer=self.share_layer_encoder,
            ),
            Decoder(
                DecoderLayer(self.d_model, self_attn, attn, deepcopy(ff), dropout),
                self.num_layers,
                share_layer=self.share_layer_decoder,
            ),
            lambda x: x,  # nn.Sequential(Embeddings(self.d_model, src_vocab), deepcopy(position)),
            nn.Sequential(Embeddings(self.d_model, self.vocab_size), deepcopy(position)),
            Generator(self.d_model, self.vocab_size),
        )
        self.att_embed = nn.Sequential(
            nn.Linear(self.att_feat_size, self.d_model), nn.ReLU(), nn.Dropout(self.drop_prob_src)
        )
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.model = model

    def _prepare_feature(
        self,
        att_feats: Tensor,
        att_masks: Optional[Tensor] = None,
        boxes: Optional[Tensor] = None,
        seq: Optional[Tensor] = None,
    ):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = seq.data.ne(self.pad_idx)  # seq_mask: torch.Tensor
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & self.subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, boxes, seq, att_masks, seq_mask

    # noinspection PyMethodOverriding
    def _forward(self, att_feats, boxes, seqs, att_masks=None, **kwargs):
        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes, seqs)
        out = self.model(att_feats, boxes, seq, att_masks, seq_mask)
        outputs = self.model.generator(out)
        return outputs

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
        out = self.model.decode(memory, mask, ys, self.subsequent_mask(ys.size(1)).to(memory.device))
        logprobs = self.model.generator(out[:, -1])
        # Add layer cache into state list, transposed so that beam_step can reorder them
        return logprobs, [ys.unsqueeze(0)] + self._retrieve_caches()

    # noinspection PyMethodOverriding
    def _sample(self, att_feats, boxes, att_masks=None, opt=None, **kwargs):
        if opt is None:
            opt = {}
        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes)
        memory = self.model.encode(att_feats, boxes, att_masks)
        state = None
        return self._generate_captions(att_feats, att_masks, memory, state, opt)

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
        mask = torch.triu(torch.ones(attn_shape), diagonal=1).eq(0)
        return mask

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        RelationTransformerModel.COLLATE_FN.add_argparse_args(parser)
        CachedTransformerBase.add_argparse_args(parser)
        # Relation args
        parser.add_argument(
            "--no_box_trigonometric_embedding",
            action="store_true",
            help="bool: If `True`, do not use trigonometric embedding.",
        )
        # fmt: on
        # return parser
