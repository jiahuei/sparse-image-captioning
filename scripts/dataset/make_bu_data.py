import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import binascii
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument("--downloaded_feats", default="data/bu_data", help="downloaded feature directory")
parser.add_argument("--output_dir", default="data/cocobu", help="output feature files")

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]
infiles = [
    "trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv",
    "trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv",
    "trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0",
    "trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1",
    "test2014/test2014_resnet101_faster_rcnn_genome.tsv.0",
    "test2014/test2014_resnet101_faster_rcnn_genome.tsv.1",
    "test2014/test2014_resnet101_faster_rcnn_genome.tsv.2",
]

os.makedirs(args.output_dir + "_att", exist_ok=True)
os.makedirs(args.output_dir + "_fc", exist_ok=True)
os.makedirs(args.output_dir + "_box", exist_ok=True)

all_image_ids = set()
pbar = tqdm(total=123287 + 40775, desc="Saving image features")
for infile in infiles:
    print("Reading " + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES)
        for item in reader:
            item["image_id"] = int(item["image_id"])
            item["num_boxes"] = int(item["num_boxes"])
            if len(item["boxes"]) % 4 != 0 or len(item["features"]) % 4 != 0:
                assert item["image_id"] in (
                    300104,
                    147295,
                    321486,
                ), f"Expected problematic images (300104, 147295, 321486), saw {item['image_id']} instead."
                # https://github.com/peteanderson80/bottom-up-attention/issues/7#issuecomment-332477816
                # https://github.com/ruotianluo/self-critical.pytorch/issues/63#issuecomment-432939699
                continue
            for field in ["boxes", "features"]:
                feature = np.frombuffer(base64.decodebytes(item[field].encode()), dtype=np.float32)
                item[field] = feature.reshape((item["num_boxes"], -1))
            np.save(os.path.join(args.output_dir + "_att", str(item["image_id"])), item["features"])
            np.save(os.path.join(args.output_dir + "_fc", str(item["image_id"])), item["features"].mean(0))
            np.save(os.path.join(args.output_dir + "_box", str(item["image_id"])), item["boxes"])
            all_image_ids.add(item["image_id"])
            pbar.update()
pbar.close()

for i in (300104, 147295, 321486):
    assert i in all_image_ids
assert len(all_image_ids) == (123287 + 40775)

"""
## Avoiding padding errors with Python 3 base64 encoding

Based on solution for Python 2:
https://gist.github.com/perrygeo/ee7c65bb1541ff6ac770

```
>>> import base64
>>> data = '{"u": "test"}'
>>> code = base64.b64encode(data.encode())
>>> code
b'eyJ1IjogInRlc3QifQ=='
```

Note the trailing `==` to make len a multiple of 4. This decodes properly

```
>>> len(code)
20
>>> base64.b64decode(code)
b'{"u": "test"}'
>>> base64.b64decode(code).decode() == data
True
```

*without* the == padding (this is how many things are encoded for e.g. access tokens)
```
>>> base64.b64decode(code[0:18]).decode() == data
...
binascii.Error: Incorrect padding 
```

However, you can add an arbitrary amount of padding (it will ignore extraneous padding)

```
>>> base64.b64decode(code + b"========").decode() == data
True
```

Thus adding 3 padding `=` will always produce the same result.
```
>>> base64.b64decode(code + b"===").decode() == data
True
```

"""
