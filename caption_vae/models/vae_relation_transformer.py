# -*- coding: utf-8 -*-
"""
Source:
https://github.com/pytorch/examples/blob/49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de/word_language_model/model.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from copy import deepcopy
from utils.config import Config
from models import relation_transformer as relation
from models.caption_model import CaptionModel


class POSVAE(nn.Module):
    """POS encoder based on transformer."""

    def __init__(self, n_token, n_inp=512, n_head=8, d_ff=2048, n_layers=6, dropout=0.1):
        super().__init__()
        self.model_type = "TransformerEncoder"
        self.pos_encoder = relation.PositionalEncoding(n_inp, dropout)
        encoder_layers = TransformerEncoderLayer(n_inp, n_head, d_ff, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_token, n_inp)
        self.z_layer = nn.Linear(n_inp, n_inp * 2)
        self.n_inp = n_inp

        # self.init_weights()

    # def init_weights(self):
    #     init_range = 0.1
    #     nn.init.uniform_(self.encoder.weight, -init_range, init_range)
    #     nn.init.zeros_(self.z_layer.bias)
    #     nn.init.uniform_(self.z_layer.weight, -init_range, init_range)

    def reparameterize(self, mu, logvar):
        # https://www.kaggle.com/ethanwharris/fashion-mnist-vae-with-pytorch-and-torchbearer
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, src, src_key_padding_mask=None, pad_token_id=None):
        """
        Args:
            src: Input tensor of shape (N, S).
            src_key_padding_mask: BoolTensor of shape (N, S), True to mask positions.
            pad_token_id: Token ID for padding.
        Returns:
            Embedding vector.
        """
        if src_key_padding_mask is None:
            assert isinstance(pad_token_id, int), f"`{pad_token_id}` must be an int, saw `{type(pad_token_id)}`"
            src_key_padding_mask = src == pad_token_id
        else:
            assert src_key_padding_mask.dtype == torch.bool
            assert src.shape == src_key_padding_mask.shape

        src_embed = self.encoder(src) * math.sqrt(self.n_inp)
        src_embed = self.pos_encoder(src_embed)
        src_embed = src_embed.transpose(0, 1)  # Transpose to (S, N, E)
        output = self.transformer_encoder(src_embed, mask=None, src_key_padding_mask=src_key_padding_mask)
        # output: (16, 75, 512)
        assert src_embed.shape[:2] == output.shape[:2]
        # output = output.transpose(0, 1)
        # print("VAE src", src)
        div = (src != pad_token_id).sum(dim=1, keepdim=True).float()
        output = output.sum(dim=0) / div
        # print("VAE outputs", output)
        # output: (75, 512)
        mu, logvar = torch.split(self.z_layer(output), (self.n_inp, self.n_inp), dim=-1)
        # mu: (75, 512)
        sample = self.reparameterize(mu, logvar)
        return sample, mu, logvar


# noinspection PyAttributeOutsideInit
class EncoderDecoder(nn.Module):
    """
    A Encoder-Decoder architecture with an additional POS encoder.
    """

    def __init__(self, encoder, pos_encoder, decoder, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.pos_encoder = pos_encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, pos, boxes, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        memory, z = self.encode(src, boxes, src_mask, pos)
        return self.decode(memory, z, src_mask, tgt, tgt_mask)

    def encode(self, src, boxes, src_mask, pos=None):
        memory = self.encoder(src, boxes, src_mask)
        if pos is None:
            sample = self.mu = self.logvar = memory.new_zeros(size=memory.shape[::2])
        else:
            sample, self.mu, self.logvar = self.pos_encoder(pos, pad_token_id=0)
        return memory, sample

    def decode(self, memory, pos, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(1)
        tgt = torch.cat((pos, tgt[:, 1:, :]), dim=1)
        return self.decoder(tgt, memory, src_mask, tgt_mask)


class VAERelationTransformer(relation.RelationTransformerModel):

    def make_model(self, n_head=8, dropout=0.1):
        """Helper: Construct a model from hyperparameters."""
        d_model = self.input_encoding_size
        tgt_vocab = self.vocab_size + 1

        bbox_attn = relation.BoxMultiHeadedAttention(n_head, d_model, self.box_trigonometric_embedding)
        attn = relation.MultiHeadedAttention(n_head, d_model)
        ff = relation.PositionwiseFeedForward(d_model, self.rnn_size, dropout)
        position = relation.PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            encoder=relation.Encoder(relation.EncoderLayer(
                d_model, deepcopy(bbox_attn), deepcopy(ff), dropout), self.num_layers
            ),
            pos_encoder=POSVAE(
                n_token=self.config.num_special_symbols,
                n_inp=d_model,
                n_head=n_head,
                d_ff=self.rnn_size,
                n_layers=self.num_layers,
            ),
            # pos_encoder=None,
            decoder=relation.Decoder(relation.DecoderLayer(
                d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout), self.num_layers
            ),
            tgt_embed=nn.Sequential(relation.Embeddings(d_model, tgt_vocab), deepcopy(position)),
            generator=relation.Generator(d_model, tgt_vocab)
        )

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, config: Config):
        super().__init__(config)

    def _forward(self, att_feats, pos, boxes, seq, att_masks=None):
        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes, seq)
        out = self.model(att_feats, pos, boxes, seq, att_masks, seq_mask)
        # print("out", out)
        outputs = self.model.generator(out)
        # print("outputs", outputs)
        return outputs
