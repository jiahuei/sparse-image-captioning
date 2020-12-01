# -*- coding: utf-8 -*-
"""
Created on 29 Jul 2020 18:52:54
@author: jiahuei
"""
from torch.utils.data import DataLoader, Dataset


class ListDataset(Dataset):
    """Basically a `list` but is a subclass ofm `Dataset`."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Collate:
    def __init__(self):
        pass

    def __call__(self, batch):
        print(batch)
        print(tuple(_["x"] for _ in batch))
        # image_paths, captions, image_ids, all_gts, pos = zip(*batch)


data = [
    dict(x=i, y=-i*2)
    for i in range(10)
]
loader = DataLoader(
    dataset=ListDataset(data),
    batch_size=3,
    shuffle=False,
    num_workers=0,
    collate_fn=Collate(),
    pin_memory=False,
    drop_last=False,
)

for batch_ndx, sample in enumerate(loader):
    print(batch_ndx)
    print(sample)
