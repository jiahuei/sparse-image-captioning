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
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List
from time import sleep, time
from tqdm import tqdm
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
    def __init__(self, cache_dict, input_att_dir, input_fc_dir, cache_max_items=-1):
        self.cache_dict = cache_dict
        self.cache_max_items = cache_max_items
        self.input_att_dir = input_att_dir
        self.input_fc_dir = input_fc_dir

    def _cache_data(self, key, key_value_fn):
        try:
            data = self.cache_dict[key]
        except KeyError:
            data = key_value_fn(key)
            if len(self.cache_dict) < self.cache_max_items:
                self.cache_dict[key] = data
        return data

    @staticmethod
    def _get_att_feats(path):
        data = np.load(path)['feat']
        data = data.reshape(-1, data.shape[-1]).astype("float32")
        # data = np.random.normal(size=(36, 2048)).astype("float32")
        return data

    @staticmethod
    def _get_fc_feats(path):
        data = np.load(path).astype("float32")
        # data = np.random.normal(size=(2048,)).astype("float32")
        return data

    def __call__(self, image_ids):
        att_feats = [
            self._cache_data(os.path.join(self.input_att_dir, f"{imgid}.npz"), self._get_att_feats)
            for imgid in image_ids
        ]
        fc_feats = [
            self._cache_data(os.path.join(self.input_fc_dir, f"{imgid}.npy"), self._get_fc_feats)
            for imgid in image_ids
        ]
        return att_feats, fc_feats


def get_memory_info():
    """
    Get node total memory and memory usage
    https://stackoverflow.com/a/17718729
    """
    with open("/proc/meminfo", "r") as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == "MemTotal:":
                ret["total"] = int(sline[1])
            elif str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                tmp += int(sline[1])
        ret["free"] = tmp
        ret["used"] = int(ret["total"]) - int(ret["free"])
    return ret


def main(cache_dict):
    bu_dir = os.path.join("/master", "datasets", "mscoco", "bu")
    input_att_dir = os.path.join(bu_dir, "cocobu_att")
    dataset = ListDataset([_.replace(".npz", "") for _ in os.listdir(input_att_dir)])

    collate_fn = Collate(
        cache_dict=cache_dict,
        input_att_dir=input_att_dir,
        input_fc_dir=os.path.join(bu_dir, "cocobu_fc"),
        cache_max_items=len(dataset) * 2,
    )
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=5,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )

    for i in range(1):
        # First loop will add data to the shared_dict
        # Subsequent loop will just get the data
        t0 = time()
        for j, x in enumerate(tqdm(loader)):
            if j % 500 == 0:
                mem_info = get_memory_info()
                print(f"Memory info: {mem_info}")
                if mem_info["free"] / mem_info["total"] < 0.3:
                    raise SystemExit("Memory limit reached.")
            pass
        print(f"Time taken for loop #{i}: {time() - t0:.1f}")
        print(f"Memory info: {get_memory_info()}")
        sleep(60 * 3)


if __name__ == "__main__":
    manager = Manager()
    shared_dict = manager.dict()
    main(shared_dict)
