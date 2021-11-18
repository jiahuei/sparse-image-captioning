# -*- coding: utf-8 -*-
r"""
Created on 02 Apr 2021 22:07:05
@author: jiahuei
"""
import os
import time
import math
import random
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

sns.set_theme(style="darkgrid", rc={"legend.loc": "lower left", "legend.framealpha": 0.7})


def set_seed(seed: int):
    assert isinstance(seed, int)
    # set Random seed
    random.seed(seed)
    np.random.seed(seed)
    print(f"RNG seed set to {seed}.")


def get_pe(height=6, width=6):
    pe = np.zeros((height, width), dtype=np.float32)
    position = np.expand_dims(np.arange(0, height, dtype=np.float32), 1)
    div_term = np.exp(np.arange(0, width, 2, dtype=np.float32) * -(math.log(10) / width))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def get_gauss(height=6, width=6):
    x, y = np.meshgrid(np.linspace(0, 1.75, width), np.linspace(0, 1.75, height))
    dst = np.sqrt(x * x + y * y)
    sigma = 1
    muu = 0.000
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))
    gauss = gauss * 2 - 1
    return gauss


def get_mask(height=6, width=6):
    return np.random.uniform(low=-5.0, high=5.0, size=(height, width))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bernoulli_sample(probs: np.ndarray):
    rand = np.random.uniform(low=0.0, high=1.0, size=probs.shape)
    # return np.ceil(probs - rand)
    return np.where(probs > rand, 1, 0)


def round_sample(probs: np.ndarray):
    return np.where(probs > 0.5, 1, 0)


def test_bernoulli_sample():
    x = np.array(0.21)
    res = []
    for i in range(100000):
        res.append(bernoulli_sample(x))
    print(np.mean(res))


def main(
    output_dir,
    palette=sns.diverging_palette(20, 220, as_cmap=True),
    annot=False,
    output_dpi=600,
    linewidth=2.0,
    context="paper",
    fig_scale=1.5,
):
    sns.set_context(context)
    common_kwargs = dict(
        cmap=palette,
        annot=annot,
        annot_kws={"fontsize": 18},
        fmt=".1f",
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )

    mask = get_mask()
    mask_sigmoid = sigmoid(mask)
    mask_bern = bernoulli_sample(mask_sigmoid)
    mask_round = round_sample(mask_sigmoid)
    weight = get_gauss()
    weight_bern = weight * mask_bern
    weight_round = weight * mask_round
    matrices = dict(
        mask=mask,
        mask_sigmoid=mask_sigmoid,
        mask_bern=mask_bern,
        mask_round=mask_round,
        weight=weight,
        weight_bern=weight_bern,
        weight_round=weight_round,
    )

    for name, mat in tqdm(matrices.items()):
        if name == "mask":
            vmin = -5
            vmax = 5
        else:
            vmin = -1
            vmax = 1
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.0 * fig_scale, 3.0 * fig_scale))
        ax = sns.heatmap(mat, vmin=vmin, vmax=vmax, ax=ax, **common_kwargs)
        # Adjust margins and layout
        plt.tight_layout(pad=0)
        fname = name
        if annot:
            fname += "_annot"
        plt.savefig(f"{os.path.join(output_dir, fname)}.png", dpi=output_dpi)  # , plt.show()
        plt.clf()
        plt.close("all")


if __name__ == "__main__":
    t = int(time.time())
    # Good seeds: 1617446054, 1617446013, 1617445976
    print(t)
    set_seed(1617446054)
    main("./matrices", annot=True)
