# -*- coding: utf-8 -*-
"""
Created on 16 Sep 2020 15:00:29
@author: jiahuei
"""
import torch
from torch import nn


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, mask, reward):
        """
        This function computes
            log(y_t) * reward * mask_t  (where mask_t zeroes out non-words in the sequence)
        given
            input = predicted probability
            sequence = predicted word index
            reward = ...
        """
        input = input.contiguous().view(-1)
        # reward = reward.repeat(mask.shape[1], 1).transpose(0, 1)
        reward = reward.contiguous().view(-1).unsqueeze(1)
        mask = mask.float()
        output = -input * (mask * reward).contiguous().view(-1)
        output = torch.sum(output) / torch.sum(mask)
        return output


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, : input.size(1)]
        mask = mask[:, : input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, : input.size(1)]
        mask = mask[:, : input.size(1)]

        input = input.contiguous().view(-1, input.size(-1))
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()
