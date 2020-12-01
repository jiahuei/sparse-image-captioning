# -*- coding: utf-8 -*-
"""
Created on 29 Jul 2020 18:52:54
@author: jiahuei
"""
import torch
from torch.utils.data import TensorDataset, DataLoader


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True, shuffle=True)
# loader = iter(loader)
# while True:
#     try:
#         print(next(loader))
#     except StopIteration as e:
#         loader = iter(loader)
for batch_ndx, sample in enumerate(loader):
    print(batch_ndx)
    print(sample.inp)
    print(sample.tgt)
for batch_ndx, sample in enumerate(loader):
    print(batch_ndx)
    print(sample.inp)
    print(sample.tgt)
