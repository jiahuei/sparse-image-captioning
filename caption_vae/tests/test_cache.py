# -*- coding: utf-8 -*-
"""
Created on 21 Oct 2020 00:40:44
@author: jiahuei

Script to demonstrate the usage of shared dicts using multiple workers.

In the first epoch the shared dict in the dataset will be filled with
random values. The next epochs will just use the dict without "loading" the
data again.

@author: ptrblck
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from time import sleep, time
from multiprocessing import Manager


class ListDataset(Dataset):
    """Basically a `list` but is a subclass ofm `Dataset`."""

    def __init__(self, data: List):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Collate:
    def __init__(self, cache_dict, cache_max_items=5):
        self.cache_dict = cache_dict
        self.cache_max_items = cache_max_items

    def _cache_data(self, key, key_value_fn):
        try:
            data = self.cache_dict[key]
        except KeyError:
            print(f"Cache miss, loading {key}")
            data = key_value_fn(key)
            if len(self.cache_dict) < self.cache_max_items:
                self.cache_dict[key] = data
        return data

    @staticmethod
    def _get_item(key):
        sleep(0.3)
        return f"{key}_returned"

    def __call__(self, batch):
        x, y = zip(*batch)
        x = [self._cache_data(_, self._get_item) for _ in x]
        y = [self._cache_data(_, self._get_item) for _ in y]
        return x, y


if __name__ == "__main__":
    # Init
    manager = Manager()
    shared_dict = manager.dict()

    loader = DataLoader(
        ListDataset([(str(_), str(_ * -2)) for _ in range(15)]),
        collate_fn=Collate(cache_dict=shared_dict, cache_max_items=28),
        batch_size=5,
        num_workers=2,
        shuffle=True,
        pin_memory=True
    )

    for i in range(10):
        # First loop will add data to the shared_dict
        # Subsequent loop will just get the data
        t0 = time()
        for x in loader:
            # print(x)
            pass
        print(f"---> Time taken for loop #{i}: {time() - t0:.1f}")
