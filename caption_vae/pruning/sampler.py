# -*- coding: utf-8 -*-
"""
Created on 24 Sep 2020 19:53:25
@author: jiahuei
"""
import torch


# noinspection PyMethodOverriding
class BernoulliSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        return torch.bernoulli(probs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BernoulliSampleSigmoid(BernoulliSample):
    @staticmethod
    def forward(ctx, logits):
        return torch.bernoulli(torch.sigmoid(logits))


# noinspection PyMethodOverriding
class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        return torch.round(probs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class RoundSigmoid(Round):
    @staticmethod
    def forward(ctx, logits):
        return torch.round(torch.sigmoid(logits))


def bernoulli_sample_sigmoid(logits, bypass_sigmoid_grad=False):
    """
    Performs stochastic Bernoulli sampling.
    Accepts raw logits instead of normalised probabilities.
    """
    # if sampling_temperature:
    #     logits = logits / sampling_temperature
    if bypass_sigmoid_grad:
        sample = BernoulliSampleSigmoid.apply(logits)
    else:
        sample = BernoulliSample.apply(torch.sigmoid(logits))
    return sample


def rounding_sigmoid(logits, bypass_sigmoid_grad=False):
    """
    Performs deterministic binarisation with adjustable threshold.
    Accepts raw logits instead of normalised probabilities.
    """
    if bypass_sigmoid_grad:
        sample = RoundSigmoid.apply(logits)
    else:
        sample = Round.apply(torch.sigmoid(logits))
    return sample
