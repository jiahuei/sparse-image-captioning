# -*- coding: utf-8 -*-
"""
Created on 25 Sep 2020 22:14:31
@author: jiahuei

cd caption_vae
python -m pruning.quantize_test
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


# noinspection PyAbstractClass,PyAttributeOutsideInit
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.make_model()

    def make_model(self):
        self.embed = nn.Embedding(3, 5)
        self.lstm = nn.LSTMCell(5, 5)
        self.out = nn.Linear(5, 4)

    def forward(self, inputs):
        net = F.relu(self.embed(inputs))
        net = F.relu(self.lstm(net)[0])
        return self.out(net)

    def print_named_parameters(self, header=""):
        print(f"-------{header}---------\nnamed_parameters\n")
        print(list(self.named_parameters()))

    def print_model(self, header=""):
        print(f"-------{header}---------\nmodel\n")
        print(self)


# noinspection PyAbstractClass,PyAttributeOutsideInit
class ModelPrune(prune.PruningMixin, Model):
    def __init__(self, mask_type):
        super().__init__(mask_type=mask_type)

    def make_model(self):
        mask_params = {
            "mask_type": self.mask_type,
            "mask_init_value": 5.
        }
        self.embed = MaskedEmbedding(3, 5, **mask_params)
        self.lstm = MaskedLSTMCell(5, 5, **mask_params)
        self.out = MaskedLinear(5, 4, **mask_params)


def quantize_model(mask_type):
    model = ModelPrune(mask_type)

    model.update_masks_once(sparsity_target=SPARSITY_TARGET)
    state_dict = model.state_dict_dense(discard_pruning_mask=True, prune_weights=True)
    state_dict_sparse = model.state_dict_sparse(discard_pruning_mask=True)
    print(state_dict_sparse)
    model.print_named_parameters(" after pruning ")

    del model
    model = Model()
    model.load_state_dict(state_dict)
    model.print_named_parameters(" new model after reloading weights ")
    model.cpu()
    # model = torch.quantization.quantize_dynamic(
    #     model, dtype=torch.qint8
    # )
    # set quantization config for server (x86)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # insert observers
    torch.quantization.prepare(model, inplace=True)
    # Calibrate the model and collect statistics
    model(torch.from_numpy(np.random.randint(low=0, high=2, size=(5,), dtype="int64")))

    # convert to quantized version
    torch.quantization.convert(model, inplace=True)
    model.print_named_parameters(" after quantization ")
    model.print_model(" after quantization ")
    print(model.state_dict())


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    logging.basicConfig(level=logging.DEBUG)
    set_seed(0)

    SPARSITY_TARGET = 0.8
    USE_CUDA = True
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    quantize_model(prune.MAG_UNIFORM)

