# -*- coding: utf-8 -*-
r"""
Created on 25 Mar 2021 19:34:40
@author: jiahuei

"""
import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

# https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/
gray3 = sns.color_palette("gray_r", n_colors=3)
crest3 = sns.color_palette("crest_r", n_colors=3)
summer3 = sns.color_palette("summer_r", n_colors=4)[1:]
mako3 = sns.color_palette("mako_r", n_colors=3)
flare3 = sns.color_palette("flare", n_colors=3)
blue3 = sns.cubehelix_palette(3, start=.5, rot=-.5)
cranberry3 = sns.dark_palette("#9e003a", n_colors=4, reverse=True)[:3]
coffee3 = sns.dark_palette("#a6814c", n_colors=4, reverse=True)[:3]
# sns.palplot([
#     *sns.color_palette("OrRd", 3),
#     *sns.dark_palette("#a6814c", n_colors=4, reverse=True),
# ])

sns.set_context("paper")
sns.set_theme(style="darkgrid", rc={"legend.loc": "lower left", "legend.framealpha": 0.7})


# print(plt.rcParams)


def get_ylim(df, score_key, margin=0.05, min_threshold=0.8):
    max_score = df.loc[:, score_key].max()
    min_score = df.loc[:, score_key]
    min_score = min_score[min_score > max_score * min_threshold].min()
    # snip_min_score = df.loc[df["Prune method"] == "SNIP", score_key].min()
    # other_min_score = df.loc[df["Prune method"] != "SNIP", score_key]
    # other_min_score = other_min_score[other_min_score > 0].min()
    ylim = (min_score, max_score)
    # if min_score / max_score >= 0.75:
    #     ylim = (min_score, max_score)
    # else:
    #     ylim = (max_score * 0.75, max_score)
    margin = (max_score - min_score) * margin
    ylim = (ylim[0] - margin * 2, ylim[1] + margin)
    return ylim


def set_style(ax, linestyle, marker):
    for line, leg_line, ls, m in zip(ax.lines, ax.legend().get_lines(), linestyle, marker):
        if ls is not None:
            line.set_linestyle(ls)
            leg_line.set_linestyle(ls)
        if m is not None:
            line.set_marker(m)
            leg_line.set_marker(m)
    return ax


def plot(df, score_name, fig_title, output_path):
    methods = [_ for _ in df.columns.tolist() if _.lower() != "nnz"]
    if len(methods) == 8:
        palette = [gray3[1], cranberry3[0], flare3[0], mako3[2], *mako3, "#9b59b6"]
    elif len(methods) == 10:
        palette = [gray3[1], cranberry3[0], *flare3, mako3[2], *mako3, "#9b59b6"]
    else:
        raise ValueError(f"Too many methods (len={len(methods)}): {methods}")
    line_styles = []
    for m in methods:
        m = m.lower()
        if m == "baseline":
            line_styles.append(":")
        elif m.startswith("hard"):
            line_styles.append("--")
        else:
            line_styles.append(None)

    marker_styles = []
    for m in methods:
        m = m.lower()
        if m == "baseline":
            marker_styles.append(None)
        elif m == "proposed":
            marker_styles.append("o")
        elif m.startswith("lottery"):
            marker_styles.append("^")
        elif m.startswith("snip"):
            marker_styles.append("v")
        else:
            marker_styles.append("X")

    # Main chart
    df2 = df[methods].stack().reset_index(level=1).rename(columns={"level_1": "Prune method", 0: score_name})
    df2["Prune method"] = df2["Prune method"].map(lambda x: "Dense baseline" if x.lower() == "baseline" else x)
    df2.index = df2.index.map(lambda x: f"{x * 100:.1f} %")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.25))
    ax.set(ylim=get_ylim(df2, score_name))
    ax = sns.lineplot(
        data=df2, x="Sparsity", y=score_name, hue="Prune method", ax=ax, palette=palette,
    )
    ax = set_style(ax, line_styles, marker_styles)
    # NNZ axis
    df2 = df.set_index("NNZ")[methods]
    df2 = df2.stack().reset_index(level=1).rename(columns={"level_1": "Prune method", 0: score_name})
    with sns.axes_style("darkgrid", rc={"axes.grid": False}):
        # print(sns.axes_style())
        ax2 = ax.twiny()
        sns.lineplot(
            data=df2, x="NNZ", y=score_name, hue="Prune method", ax=ax2, legend=None, visible=False
        )
    # Title
    ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    # Adjust margins and layout
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=300)  # , plt.show()
    plt.clf()
    plt.close("all")


def main():
    tsv_files = os.listdir("plot_data")
    for f in tqdm(tsv_files):
        df = pd.read_csv(os.path.join("plot_data", f), sep="\t", header=0, index_col=0)
        fname = os.path.splitext(f)[0]
        title, metric = fname.split(" --- ")
        plot(df, metric, title, f"{fname}.png")


if __name__ == "__main__":
    main()
