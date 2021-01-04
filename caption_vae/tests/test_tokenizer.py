# -*- coding: utf-8 -*-
"""
Created on 29 Jul 2020 18:52:54
@author: jiahuei

cd caption_vae
python -m tests.test_tokenizer.py
"""
import os
import json
from tokenizer import get_tokenizer


class Config:
    pass


with open("/master/datasets/mscoco/dataset_coco.json", "r") as f:
    data = json.load(f)

# Process and tokenize
self_data = {}
all_img_id = set()
all_filename = set()
for d in data["images"]:
    # Process image ID
    img_id = d["imgid"]
    all_img_id.add(img_id)
    all_filename.add(d["filename"])
    # The rest
    img_path = os.path.join(d.get("filepath", "images"), d["filename"])
    split = "train" if d["split"] == "restval" else d["split"]
    if split not in self_data:
        self_data[split] = []
    all_gts = [sent["raw"] for sent in d["sentences"]]
    for sent in d["sentences"]:
        tmp_dict = {
            "split": split,
            "img_path": img_path,
            "img_id": img_id,
            "karpathy_tokens": sent["tokens"],
            "raw": sent["raw"],
            "all_gts": all_gts,
        }
        self_data[split].append(tmp_dict)

x = [sent["raw"] for sent in self_data["train"]]
y = [" ".join(sent["karpathy_tokens"]) for sent in self_data["train"]]

with open("train_captions.txt", "w") as f:
    f.write("\n".join([" ".join(sent["karpathy_tokens"]) for sent in self_data["train"]]))
config = Config()
config.log_dir = "/master/src"
config.tokenizer_train_files = "train_captions.txt"
config.vocab_size = 10000
config.logging_level = 2
tokenizer = get_tokenizer("word")(config)

print(tokenizer.tokenize("TABLE TABLE table"))
print(tokenizer.encode("table table", add_bos_eos=False, max_seq_length=0))
print(tokenizer.decode(tokenizer.encode("TABLE TABLE table", add_bos_eos=False, max_seq_length=0)))

i = 0
for sent in self_data["train"]:
    enc = tokenizer.encode(" ".join(sent["karpathy_tokens"]), add_bos_eos=False, max_seq_length=0)
    if len(sent["karpathy_tokens"]) == len(enc):
        pass
    else:
        # print(" ".join(sent["karpathy_tokens"]))
        # print(tokenizer.decode(enc))
        # print(len(sent["karpathy_tokens"]))
        # print(len(enc))
        i += 1

print(len(self_data["train"]))
print(i)
