# -*- coding: utf-8 -*-
"""
Created on 25 Sep 2020 22:14:31
@author: jiahuei

cd caption_vae
python -m pruning.prune_test
"""
import os
import logging
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from copy import deepcopy
from pruning import prune
from pruning.masked_layer import MaskedLinear, MaskedLSTMCell, MaskedEmbedding
from utils.model_utils import set_seed


# noinspection PyAbstractClass
class Model(prune.PruningMixin, nn.Module):
    def __init__(self, mask_type):
        super().__init__(mask_type=mask_type, mask_freeze_scope="out.")
        # super().__init__(mask_type=mask_type)
        mask_params = {
            "mask_type": mask_type,
            "mask_init_value": 5.
        }
        self.embed = MaskedEmbedding(3, 5, **mask_params)
        self.mid = nn.Linear(5, 5)
        self.lstm = MaskedLSTMCell(5, 5, **mask_params)
        # self.mid = MaskedLinear(5, 5, **mask_params)
        ffl = MaskedLinear(5, 5, **mask_params)
        self.ff = nn.ModuleList([deepcopy(ffl) for _ in range(2)])
        self.out = MaskedLinear(5, 4, **mask_params)
        self.loss_layer = nn.CrossEntropyLoss()
        if USE_CUDA:
            self.cuda()

    def forward(self, inputs):
        if USE_CUDA:
            inputs = inputs.cuda()
        net = F.relu(self.embed(inputs))
        net = F.relu(self.mid(net))
        net = F.relu(self.lstm(net)[0])
        for ly in self.ff:
            net = F.dropout(F.relu(ly(net)), p=0.35, training=self.training)
        return self.out(net)

    def compute_loss(self, predictions, labels, step, max_step):
        if USE_CUDA:
            labels = labels.cuda()
        loss = self.loss_layer(predictions, labels)
        if self.mask_type == prune.REGULAR:
            loss += self.compute_sparsity_loss(SPARSITY_TARGET, weight=7.5, current_step=step, max_step=max_step)
        return loss


def train_model(model):
    # if USE_CUDA:
    #     model.cuda()
    print("=========================================")
    print(model.mask_type)
    print("=========================================")
    print("\nnamed_parameters\n", list(model.named_parameters()))
    # print([(n, t.grad) for n, t in model.named_parameters()])

    optim_params = [{"params": list(model.all_weights(named=False))}]
    if model.mask_type in prune.SUPER_MASKS:
        optim_params += [
            {"params": list(model.active_pruning_masks(named=False)), "lr": 100, "weight_decay": 0, "eps": 1e-2}
        ]
    optimizer = optim.Adam(optim_params, lr=1e-2, weight_decay=1e-5)

    if model.mask_type in prune.MAG_HARD + [prune.SNIP]:
        if model.mask_type == prune.SNIP:
            inputs = torch.from_numpy(np.random.randint(low=0, high=2, size=(5,), dtype="int64"))  # .cuda()
            labels = torch.from_numpy(np.random.randint(low=0, high=3, size=(5,), dtype="int64"))  # .cuda()
            preds = model(inputs)
            loss = model.compute_loss(preds, labels, 0, 40)
            loss.backward()
        model.update_masks_once(sparsity_target=SPARSITY_TARGET)

    for i in range(40):
        inputs = torch.from_numpy(np.random.randint(low=0, high=2, size=(5,), dtype="int64"))  # .cuda()
        labels = torch.from_numpy(np.random.randint(low=0, high=3, size=(5,), dtype="int64"))  # .cuda()
        optimizer.zero_grad()
        # optimizer2.zero_grad()
        preds = model(inputs)
        loss = model.compute_loss(preds, labels, i, 40)
        loss.backward()
        optimizer.step()
        if model.mask_type in prune.MAG_ANNEAL:
            model.update_masks_gradual(
                sparsity_target=SPARSITY_TARGET, current_step=i, start_step=10, prune_steps=10, prune_frequency=3
            )
        print(model.sparsity_target)
        # optimizer2.step()

    print("\nnamed_parameters\n", list(model.named_parameters()))

    # if model.mask_type in prune.MAG_HARD:
    #     model.update_masks_once(sparsity_target=0.5)
    #     print("\nnamed_parameters\n", list(model.named_parameters()))
    #     # print([(n, t.grad) for n, t in model.named_parameters()])

    model.prune_weights()
    print("\nnamed_parameters\n", list(model.named_parameters()))

    print(model.all_mask_sparsities)
    print(model.active_mask_sparsities)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    logging.basicConfig(level=logging.DEBUG)
    set_seed(0)

    SPARSITY_TARGET = 0.8
    USE_CUDA = True
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    # train_model(Model(prune.REGULAR))
    train_model(Model(prune.MAG_BLIND))
    # train_model(Model(prune.MAG_DIST))
    # train_model(Model(prune.MAG_UNIFORM))
    # train_model(Model(prune.SNIP))
    # train_model(Model(prune.MAG_GRAD_BLIND))
    # train_model(Model(prune.MAG_GRAD_UNIFORM))

    # model = Model(prune.MAG_GRAD_UNIFORM)
    # train_model(model)
    # torch.save(model.state_dict_sparse(), os.path.join(CURR_DIR, "sparse.pth"))
    # torch.save(model.state_dict_dense(discard_pruning_mask=True), os.path.join(CURR_DIR, "dense.pth"))
    # print("\n\n\n==================================================================")
    # print(model.state_dict_sparse())
    # model = Model(prune.MAG_GRAD_UNIFORM)
    # print("\nnamed_parameters\n", list(model.named_parameters()))
    # model.load_sparse_state_dict(torch.load(os.path.join(CURR_DIR, "sparse.pth")), strict=False)
    # print("\nnamed_parameters\n", list(model.named_parameters()))
