# -*- coding: utf-8 -*-
"""
Created on 28 Aug 2020 17:34:10
@author: jiahuei
"""
import os
import sys

up_dir = os.path.dirname
CURR_DIR = up_dir(os.path.realpath(__file__))
BASE_DIR = up_dir(CURR_DIR)
sys.path.insert(1, BASE_DIR)

from opts import parse_opt
from utils.config import Config
from utils.misc import configure_logging
from utils.model_utils import set_seed
from utils.lightning import LightningModule


# noinspection PyAttributeOutsideInit
class CaptioningModel(LightningModule):
    def __init__(self, config: Config):
        super().__init__(config)

    def train(self):
        self.prepare()
        for epoch in range(3):
            for batch_idx, data in enumerate(self.train_loader):
                print(data.keys())


def main(config: Config):
    set_seed(config.seed)
    config.batch_size = 5
    config.rnn_size = 100
    model = CaptioningModel(config)
    model.train()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parse_opt()
    logger = configure_logging("DEBUG")
    main(Config(**vars(args)))
