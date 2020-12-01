# -*- coding: utf-8 -*-
"""
Created on 09 Nov 2020 22:25:38
@author: jiahuei


cd caption_vae
python -m scripts.plot_nonzero_weights_kde
"""
import os
import re
import logging
import json
import torch
import numpy as np
import seaborn as sns
from scipy.stats import mstats
from matplotlib import pyplot as plt
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from pruning import prune
from utils.model_utils import densify_state_dict
from utils.misc import replace_from_right, configure_logging
from utils.file import list_dir
from utils.config import Config

logger = logging.getLogger(__name__)


class KDE:
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

    def __init__(self):
        self.config = self.parse_opt()
        self.config.model_file = self.config.model_file.split(",")

    def __call__(self, model_dir, visualise_weights_only=True):
        print(f"Processing `{model_dir}`")
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
        nonzero_weights = flat_weights_np[flat_weights_np != 0]
        np.save(os.path.join(model_dir, "nonzero_weights_flat.npy"), nonzero_weights)

        # Output Naming
        net_name = model_config.caption_model
        if net_name.endswith("_prune"):
            net_name = replace_from_right(net_name, "_prune", "", 1)
        # net_name = net_name.replace("net", "Net")
        output_suffix = net_name
        fig_title = ""

        pruning_type = model_config.get("prune_type", None)
        if pruning_type:
            if pruning_type == prune.MASK_FREEZE:
                return None
            try:
                fig_title = f"{self.PRUNE_TYPE_TITLE[pruning_type]}, "
            except KeyError:
                raise ValueError(f"Invalid pruning type: `{pruning_type}`")
            sparsity = model_config.prune_sparsity_target * 100
            fig_title += f"{sparsity:.1f}% sparse, "
            # TexStudio cannot accept filename with dot
            output_suffix += f"_{int(sparsity)}_{pruning_type}"

        fig_title += " ".join(_.title() for _ in net_name.split("_"))
        # TexStudio will annoyingly highlight underscores in filenames
        output_suffix = output_suffix.replace("_", "-")

        # Histogram and KDE
        for i, clip_pct in enumerate([0.005, 0.001]):
            # noinspection PyTypeChecker
            self.plot_kde(
                data=mstats.winsorize(nonzero_weights, limits=clip_pct),
                # TexStudio will annoyingly highlight underscores in filenames
                output_fig_path=os.path.join(model_dir, f"KDE-{i}-{output_suffix}.png"),
                fig_title=f"Distribution of Non-zero Weights\n{fig_title}",
                fig_footnote=f"* {clip_pct * 100:.1f}% winsorization",
            )
            logger.info(f"Saved graph: clip percent = {clip_pct} (as float between 0. and 1.)")
        print("")

    def plot_kde(self, data, output_fig_path, fig_title, fig_footnote=None):
        sns.set()
        # print(sns.axes_style())
        sns.set_style(
            "whitegrid", {
                "axes.edgecolor": ".5",
                "grid.color": ".87",
                "grid.linestyle": "dotted",
                # "lines.dash_capstyle": "round",
            }
        )
        # colours = ("goldenrod", "sandybrown", "chocolate", "peru")
        # colours = ("c", "cadetblue", "lightseagreen", "skyblue")
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=self.FIG_DPI, figsize=(8.5, 6.25))
        ax = sns.distplot(
            data,
            bins=50,
            kde_kws={"gridsize": 200, "color": "darkcyan"},
            color="c",
            ax=ax,
        )
        sns.despine()
        # plt.legend(loc="upper left", bbox_to_anchor=(0.1, 1.), fontsize="small")
        plt.title(fig_title)
        if isinstance(fig_footnote, str):
            plt.figtext(
                0.90, 0.025,
                fig_footnote,
                horizontalalignment="right",
                fontsize="xx-small",
            )
        plt.savefig(output_fig_path)
        plt.clf()
        plt.close("all")

    @staticmethod
    def parse_opt() -> Namespace:
        # fmt: off
        # noinspection PyTypeChecker
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            "--log_dir", type=str, default="",
            help="str: Logging / Saving directory."
        )
        parser.add_argument(
            "--id", type=str, default="",
            help="An id identifying this run/job."
        )
        parser.add_argument(
            "--model_file", type=str, default="model_best_pruned_sparse.pth,model_best.pth",
            help="str: Model checkpoint file."
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


if __name__ == '__main__':
    configure_logging("WARNING")
    kde = KDE()
    if kde.config.id:
        dirs = [os.path.join(kde.config.log_dir, kde.config.id)]
    else:
        dirs = list(filter(os.path.isdir, list_dir(kde.config.log_dir)))
    for d in dirs:
        kde(d)
