# -*- coding: utf-8 -*-
r"""
Created on 26 Apr 2021 14:50:39
@author: jiahuei
"""
import os
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

sns.set_theme(style="darkgrid", rc={"legend.loc": "lower left", "legend.framealpha": 0.7})


def compute_sim(x):
    x = np.expand_dims(x, axis=1)
    y = x.transpose([1, 0, 2])
    z = np.sqrt(np.mean((x - y) ** 2, axis=-1))
    return z


def main(
    output_dir,
    palette=sns.color_palette("light:#5A9", as_cmap=True),
    # palette=sns.diverging_palette(220, 20, as_cmap=True),
    annot=False,
    output_dpi=600,
    linewidth=2.0,
    context="paper",
    fig_scale=1.5,
):
    sns.set_context(context, font_scale=3.0)
    common_kwargs = dict(
        cmap=palette,
        annot=annot,
        annot_kws={"fontsize": 18},
        fmt=".1f",
        cbar=True,
        xticklabels=True,
        yticklabels=True,
        square=True,
    )

    state_dict = torch.load(
        "/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1/RTrans__baseline/model_best.pth"
    )

    encoder_params = [[] for _ in range(6)]
    decoder_params = [[] for _ in range(6)]
    encoder_layer_sizes = [0] * 6
    decoder_layer_sizes = [0] * 6
    for k, v in state_dict.items():
        v = torch.nn.functional.normalize(v, p=2.0, dim=-1, eps=1e-12, out=None)
        v = v.cpu().numpy().reshape((-1))
        if k.startswith("model.encoder.layers"):
            i = int(k.split(".")[3])
            encoder_layer_sizes[i] += v.size
            encoder_params[i].append(v)
        elif k.startswith("model.decoder.layers"):
            i = int(k.split(".")[3])
            decoder_layer_sizes[i] += v.size
            decoder_params[i].append(v)
    print(encoder_layer_sizes)
    print(decoder_layer_sizes)

    encoder_params = np.array([np.concatenate(_) for _ in encoder_params])
    decoder_params = np.array([np.concatenate(_) for _ in decoder_params])
    print(encoder_params.shape)
    print(decoder_params.shape)

    encoder = compute_sim(encoder_params)
    decoder = compute_sim(decoder_params)
    matrices = dict(
        encoder=encoder,
        decoder=decoder,
    )

    for name, mat in tqdm(matrices.items()):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5 * fig_scale, 3 * fig_scale))
        ax = sns.heatmap(mat, vmin=np.unique(mat)[1], ax=ax, **common_kwargs)
        # https://stackoverflow.com/a/51760629
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(length=0)
        # Adjust margins and layout
        plt.tight_layout(pad=0)
        fname = f"layer-sim norm-last {name} (light:#5A9)"
        if annot:
            fname += "_annot"
        plt.savefig(f"{os.path.join(output_dir, fname)}.png", dpi=output_dpi)  # , plt.show()
        plt.clf()
        plt.close("all")


if __name__ == "__main__":
    main(".")
