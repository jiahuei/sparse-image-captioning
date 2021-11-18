# -*- coding: utf-8 -*-
"""
Created on 14 Oct 2020 14:34:47
@author: jiahuei
"""
import torch.nn as nn
from argparse import ArgumentParser, _ArgumentGroup
from typing import Union
from functools import reduce
from . import register_model, att_model
from ..tokenizer import Tokenizer
from ..pruning.prune import PruningMixin
from ..pruning.masked_layer import MaskedLinear, MaskedEmbedding, MaskedLSTMCell


# noinspection PyAbstractClass,PyAttributeOutsideInit
class AttModel(att_model.AttModel):
    # def __init__(self, config, tokenizer: Tokenizer = None):
    #     super().__init__(config, tokenizer)

    def make_model(self):
        mask_params = {"mask_type": self.config.prune_type, "mask_init_value": self.config.prune_supermask_init}
        self.embed = nn.Sequential(
            MaskedEmbedding(self.vocab_size, self.input_encoding_size, **mask_params),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm),
        )
        self.fc_embed = nn.Sequential(
            MaskedLinear(self.fc_feat_size, self.rnn_size, **mask_params), nn.ReLU(), nn.Dropout(self.drop_prob_lm)
        )
        self.att_embed = nn.Sequential(
            *(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())
                + (
                    MaskedLinear(self.att_feat_size, self.rnn_size, **mask_params),
                    nn.ReLU(),
                    nn.Dropout(self.drop_prob_lm),
                )
                + ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())
            )
        )

        self.logit_layers = self.config.get("logit_layers", 1)
        if self.logit_layers == 1:
            self.logit = MaskedLinear(self.rnn_size, self.vocab_size, **mask_params)
        else:
            self.logit = [
                [MaskedLinear(self.rnn_size, self.rnn_size, **mask_params), nn.ReLU(), nn.Dropout(self.drop_prob_lm)]
                for _ in range(self.config.logit_layers - 1)
            ]
            self.logit = nn.Sequential(
                *(
                    reduce(lambda x, y: x + y, self.logit)
                    + [MaskedLinear(self.rnn_size, self.vocab_size, **mask_params)]
                )
            )
        self.ctx2att = MaskedLinear(self.rnn_size, self.att_hid_size, **mask_params)


# noinspection PyAbstractClass
class Attention(att_model.Attention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.rnn_size = config.rnn_size
        self.att_hid_size = config.att_hid_size
        mask_params = {"mask_type": self.config.prune_type, "mask_init_value": self.config.prune_supermask_init}

        self.h2att = MaskedLinear(self.rnn_size, self.att_hid_size, **mask_params)
        self.alpha_net = MaskedLinear(self.att_hid_size, 1, **mask_params)


# noinspection PyAbstractClass
class UpDownCore(att_model.UpDownCore):
    def __init__(self, config, use_maxout=False):
        nn.Module.__init__(self)
        self.config = config
        self.drop_prob_lm = config.drop_prob_lm
        mask_params = {"mask_type": self.config.prune_type, "mask_init_value": self.config.prune_supermask_init}

        self.att_lstm = MaskedLSTMCell(
            config.input_encoding_size + config.rnn_size * 2, config.rnn_size, **mask_params
        )  # we, fc, h^2_t-1
        self.lang_lstm = MaskedLSTMCell(config.rnn_size * 2, config.rnn_size, **mask_params)  # h^1_t, \hat v
        self.attention = Attention(config)


# noinspection PyAbstractClass,PyAttributeOutsideInit
@register_model("up_down_lstm_prune")
class UpDownModel(PruningMixin, AttModel):
    COLLATE_FN = att_model.UpDownModel.COLLATE_FN

    def __init__(self, config, tokenizer: Tokenizer = None):
        self.num_layers = 2
        super().__init__(
            mask_type=config.prune_type,
            mask_freeze_scope=config.prune_mask_freeze_scope,
            config=config,
            tokenizer=tokenizer,
        )
        self.core = UpDownCore(self.config)

    @staticmethod
    def add_argparse_args(parser: Union[_ArgumentGroup, ArgumentParser]):
        # fmt: off
        att_model.UpDownModel.add_argparse_args(parser)
        PruningMixin.add_argparse_args(parser)
        # fmt: on
        # return parser
