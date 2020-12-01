# -*- coding: utf-8 -*-
"""
Created on 04 Oct 2020 00:14:15
@author: jiahuei
"""
import os
import logging
import numpy as np
import torch
from torch import nn, optim
# from torch.nn import functional as F


# noinspection PyAbstractClass
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 5)
        self.b = nn.Linear(5, 4)
        self.loss_layer = nn.CrossEntropyLoss()

    def forward(self, inputs):
        net = self.a(inputs)
        net = torch.tanh(net)
        return self.b(net)

    def compute_loss(self, predictions, labels):
        loss = self.loss_layer(predictions, labels)
        return loss


def train_model():
    # model.cuda()
    # .grad.data.zero_()
    model = Model()
    inputs = np.random.normal(size=(1, 3)).astype("float32")
    inputs_repeated = np.repeat(inputs, 5, axis=0)
    inputs = torch.from_numpy(inputs)  # .cuda()
    inputs_repeated = torch.from_numpy(inputs_repeated)  # .cuda()
    labels = torch.from_numpy(np.random.randint(low=0, high=3, size=(5,), dtype="int64"))  # .cuda()

    optimizer = optim.Adam(model.parameters())
    preds = model(inputs)
    preds = preds.repeat_interleave(5, dim=0)
    loss = model.compute_loss(preds, labels)
    loss.backward()
    grad_a = [_.grad.numpy() for _ in model.parameters()]
    print(model.a.weight.grad)
    optimizer.zero_grad()

    preds = model(inputs_repeated)
    loss = model.compute_loss(preds, labels)
    loss.backward()
    grad_b = [_.grad.numpy() for _ in model.parameters()]
    print(model.a.weight.grad)

    print(all(np.all(x == y) for x, y in zip(grad_a, grad_b)))


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    logging.basicConfig(level=logging.DEBUG)
    train_model()
