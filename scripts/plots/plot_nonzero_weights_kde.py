# -*- coding: utf-8 -*-
"""
Created on 09 Nov 2020 22:25:38
@author: jiahuei

python scripts/plot_nonzero_weights_kde.py --log_dir x --id y

/home/jiahuei/Documents/1_TF_files/prune/mscoco_v3
word_w256_LSTM_r512_h1_ind_xu_REG_1.0e+02_init_5.0_L1_wg_60.0_ann_sps_0.975_dec_prune_cnnFT/run_01_sparse

/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1
UpDownLSTM__supermask__0.991__wg_120.0
RTrans__supermask__0.991__wg_120.0
"""
import os
import logging
import torch
import numpy as np
import seaborn as sns
from scipy.stats import mstats
from matplotlib import pyplot as plt
from matplotlib import ticker
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from sparse_caption.pruning import prune
from sparse_caption.utils.model_utils import densify_state_dict
from sparse_caption.utils.misc import replace_from_right, configure_logging
from sparse_caption.utils.file import list_dir
from sparse_caption.utils.config import Config

logger = logging.getLogger(__name__)
gray3 = sns.color_palette("gray_r", n_colors=3)
crest3 = sns.color_palette("crest_r", n_colors=3)
summer3 = sns.color_palette("summer_r", n_colors=4)[1:]
mako3 = sns.color_palette("mako_r", n_colors=3)
flare3 = sns.color_palette("flare", n_colors=3)
blue3 = sns.cubehelix_palette(3, start=0.5, rot=-0.5)
cranberry3 = sns.dark_palette("#b2124d", n_colors=3, reverse=True)[:3]
coffee3 = sns.dark_palette("#a6814c", n_colors=4, reverse=True)[:3]

# sns.set_theme(style="darkgrid", rc={"legend.loc": "lower left", "legend.framealpha": 0.7})
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.edgecolor": ".3",
        "grid.color": "0.9",  # "axes.grid.axis": "y",
        "legend.loc": "lower left",
        "legend.framealpha": "0.6",
    },
)


def is_white_style():
    return plt.rcParams["axes.facecolor"] == "white"


def despine_white(fig):
    # Despine whitegrid
    if is_white_style():
        sns.despine(fig=fig, top=False, right=False, left=False, bottom=False, offset=None, trim=False)


def process_output_path(output_path):
    output_name, output_ext = os.path.splitext(output_path)
    if is_white_style():
        output_name += " (w)"
    else:
        output_name += " (d)"
    output_path = output_name + output_ext
    return output_path


class KDE:
    CONTEXT = "paper"
    FIG_SCALE = 1.5
    FIG_DPI = 600
    PRUNE_TYPE_TITLE = {
        prune.REGULAR: "Proposed",
        prune.MAG_GRAD_BLIND: "Gradual (blind)",
        prune.MAG_GRAD_UNIFORM: "Gradual (uniform)",
        prune.MAG_GRAD_DIST: "Gradual (distribution)",
        prune.LOTTERY_MASK_FREEZE: "Lottery (gradual)",  # For now, we only pair this with MAG_GRAD_UNIFORM
        prune.LOTTERY_MAG_BLIND: "Lottery (hard-blind)",
        prune.LOTTERY_MAG_UNIFORM: "Lottery (hard-uniform)",
        prune.LOTTERY_MAG_DIST: "Lottery (hard-distribution)",
        prune.MAG_BLIND: "Hard-blind",
        prune.MAG_UNIFORM: "Hard-uniform",
        prune.MAG_DIST: "Hard-distribution",
        prune.SNIP: "SNIP",
    }

    # https://stackoverflow.com/a/39566040
    FONT_XTINY = 10.5
    FONT_TINY = 11
    FONT_XSMALL = 12
    FONT_SMALL = 13
    FONT_MEDIUM = 14
    FONT_LARGE = 16
    FONT_XLARGE = 18

    TWO_DECIMAL_FMT = ticker.StrMethodFormatter("{x:.2f}")

    def __init__(self):
        self.config = self.parse_opt()
        self.config.model_file = self.config.model_file.split(",")

    def __call__(self, model_dir, visualise_weights_only=True):
        print(f"Processing `{model_dir}`")
        try:
            model_config = Config.load_config_json(os.path.join(model_dir, "config.json"))
            ckpt_path = [os.path.join(model_dir, _) for _ in self.config.model_file]
            ckpt_path = list(filter(os.path.isfile, ckpt_path))
            if len(ckpt_path) > 0:
                ckpt_path = ckpt_path[0]
            else:
                return None
            state_dict = densify_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
            print(f"Model weights loaded from `{ckpt_path}`")
            if visualise_weights_only:
                state_dict = {k: v for k, v in state_dict.items() if "weight" in k}
            flat_weights_np = np.concatenate([_.view(-1).numpy() for _ in state_dict.values()])

        except FileNotFoundError:
            flat_weights_np = np.load(os.path.join(model_dir, "nonzero_weights_flat.npy"))
            model_config = {
                # Just hard-code this for now
                "caption_model": "Soft-Attention LSTM",
                "prune_type": prune.REGULAR if "REG" in model_dir else prune.MAG_GRAD_BLIND,
                "prune_sparsity_target": 0.975,
            }

        nonzero_weights = flat_weights_np[flat_weights_np != 0]
        np.save(os.path.join(model_dir, "nonzero_weights_flat.npy"), nonzero_weights)

        # Output Naming
        net_name = model_config.get("caption_model", None)
        if net_name.endswith("_prune"):
            net_name = replace_from_right(net_name, "_prune", "", 1)
        # net_name = net_name.replace("net", "Net")
        output_suffix = net_name
        fig_title = ""

        pruning_type = model_config.get("prune_type", "")
        if pruning_type:
            if pruning_type == prune.MASK_FREEZE:
                logger.warning(f"Mask type = {prune.MASK_FREEZE} not supported")
                return None
            try:
                fig_title = f"{self.PRUNE_TYPE_TITLE[pruning_type]}, "
            except KeyError:
                raise ValueError(f"Invalid pruning type: `{pruning_type}`")
            sparsity = model_config.get("prune_sparsity_target", 0) * 100
            fig_title += f"{sparsity:.1f}% sparse, "
            # TexStudio cannot accept filename with dot
            output_suffix += f"_{int(sparsity)}_{pruning_type}"

        fig_title += " ".join(_.title() for _ in net_name.split("_"))
        fig_title = fig_title.replace("Lstm", "LSTM")
        # TexStudio will annoyingly highlight underscores in filenames
        output_suffix = output_suffix.replace("_", "-")

        # Histogram and KDE
        for i, clip_pct in enumerate([0.005, 0.001]):
            # noinspection PyTypeChecker
            self.plot_kde(
                data=mstats.winsorize(nonzero_weights, limits=clip_pct),
                # TexStudio will annoyingly highlight underscores in filenames
                output_fig_path=process_output_path(os.path.join(model_dir, f"KDE-{i}-{output_suffix}.png")),
                fig_title="",
                fig_footnote=f"* {clip_pct * 100:.1f}% winsorization",
            )
            logger.info(f"Saved graph: clip percent = {clip_pct} (as float between 0. and 1.)")
        print("")

    def plot_kde(self, data, output_fig_path, fig_title, fig_footnote=None):
        sns.set_context(self.CONTEXT)
        plt.rc("font", size=self.FONT_LARGE)
        plt.rc("axes", labelsize=self.FONT_LARGE, titlesize=self.FONT_LARGE)
        plt.rc("xtick", labelsize=self.FONT_LARGE)
        plt.rc("ytick", labelsize=self.FONT_LARGE)
        plt.rc("legend", fontsize=self.FONT_MEDIUM)

        # colours = ("goldenrod", "sandybrown", "chocolate", "peru")
        # colours = ("c", "cadetblue", "lightseagreen", "skyblue")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.0 * self.FIG_SCALE, 3.0 * self.FIG_SCALE))
        ax = sns.kdeplot(
            data,
            fill=True,
            common_norm=False,  # palette="crest",
            alpha=0.5,
            linewidth=0,
            color="c",
            ax=ax,
        )
        ax.xaxis.set_major_formatter(self.TWO_DECIMAL_FMT)
        ax.yaxis.set_major_formatter(self.TWO_DECIMAL_FMT)
        if fig_title:
            ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
        if isinstance(fig_footnote, str):
            ft = plt.figtext(
                0.90,
                0.0,
                fig_footnote,
                horizontalalignment="right",
                fontsize="xx-small",
            )
            bbox_extra_artists = [ft]
        else:
            bbox_extra_artists = None
        despine_white(fig)
        # Adjust margins and layout
        # https://stackoverflow.com/a/56727331
        plt.savefig(output_fig_path, dpi=self.FIG_DPI, bbox_extra_artists=bbox_extra_artists, bbox_inches="tight")
        print(f"Saved figure: `{output_fig_path}`")
        plt.clf()
        plt.close("all")

    @staticmethod
    def parse_opt() -> Namespace:
        # noinspection PyTypeChecker
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument("--log_dir", type=str, default="", help="str: Logging / Saving directory.")
        parser.add_argument("--id", type=str, default="", help="An id identifying this run/job.")
        parser.add_argument(
            "--model_file",
            type=str,
            default="model_best_pruned_sparse.pth,model_best.pth",
            help="str: Model checkpoint file.",
        )
        parser.add_argument(
            "--logging_level",
            type=str,
            default="INFO",
            choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
            help="str: Logging level.",
        )
        args = parser.parse_args()
        return args


if __name__ == "__main__":
    configure_logging("WARNING")
    kde = KDE()
    if kde.config.id:
        dirs = [os.path.join(kde.config.log_dir, kde.config.id)]
    else:
        dirs = list(filter(os.path.isdir, list_dir(kde.config.log_dir)))
    for d in dirs:
        kde(d)
