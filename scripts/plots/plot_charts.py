# -*- coding: utf-8 -*-
r"""
Created on 25 Mar 2021 19:34:40
@author: jiahuei

Font sizes: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
"""
import os
import math
import seaborn as sns
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt
from matplotlib import ticker
from tqdm import tqdm

# https://stackoverflow.com/a/39566040
FONT_XTINY = 10.5
FONT_TINY = 11
FONT_XSMALL = 12
FONT_SMALL = 13
FONT_MEDIUM = 14
FONT_LARGE = 16
FONT_XLARGE = 18

SINGLE_DECIMAL_FMT = ticker.StrMethodFormatter("{x:.1f}")

# https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/
gray3 = sns.color_palette("gray_r", n_colors=3)
crest3 = sns.color_palette("crest_r", n_colors=3)
summer3 = sns.color_palette("summer_r", n_colors=4)[1:]
mako3 = sns.color_palette("mako_r", n_colors=3)
flare3 = sns.color_palette("flare", n_colors=3)
blue3 = sns.cubehelix_palette(3, start=0.5, rot=-0.5)
cranberry3 = sns.dark_palette("#b2124d", n_colors=3, reverse=True)[:3]
coffee3 = sns.dark_palette("#a6814c", n_colors=4, reverse=True)[:3]
# sns.palplot([
#     *sns.color_palette("OrRd", 3),
#     *sns.dark_palette("#a6814c", n_colors=4, reverse=True),
# ])

# sns.set_theme(style="darkgrid", rc={"legend.loc": "lower left", "legend.framealpha": "0.6"})
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.edgecolor": ".3",
        "grid.color": "0.9",  # "axes.grid.axis": "y",
        "legend.loc": "lower left",
        "legend.framealpha": "0.6",
    },
)


# print(plt.rcParams)


def get_lim(series, margin=(0.10, 0.05), min_threshold=None) -> Tuple[float, float]:
    max_score = series.max()
    if isinstance(min_threshold, (float, int)):
        series = series[series > max_score * min_threshold]
    min_score = series.min()
    score_range = max_score - min_score
    lim = (min_score - score_range * margin[0], max_score + score_range * margin[1])
    return lim


def get_midpoint(series):
    mid = (series.max() - series.min()) / 2 + series.min()
    return mid


def set_style(ax, linestyle=None, marker=None):
    if linestyle is None and marker is None:
        return ax
    if linestyle is None:
        linestyle = [None] * len(marker)
    if marker is None:
        marker = [None] * len(linestyle)

    legend_hdl, legend_lbl = ax.get_legend_handles_labels()
    for line, leg_line, ls, m in zip(ax.lines, legend_hdl, linestyle, marker):
        if ls is not None:
            line.set_linestyle(ls)
            leg_line.set_linestyle(ls)
        if m is not None:
            line.set_marker(m)
            leg_line.set_marker(m)
    return ax


def is_white_style():
    return plt.rcParams["axes.facecolor"] == "white"


def despine_white(fig):
    # Despine whitegrid
    if is_white_style():
        sns.despine(fig=fig, top=False, right=False, left=False, bottom=False, offset=None, trim=False)


def process_output_path(output_path: str) -> str:
    output_name, output_ext = os.path.splitext(output_path)
    if is_white_style():
        output_name += " (w)"
    else:
        output_name += " (d)"
    output_path = output_name + output_ext
    return output_path


def plot_smp_performance(
    df,
    palette,
    score_name,
    output_path,
    fig_title="",
    output_dpi=600,
    min_threshold=0.8,
    context="paper",
    fig_scale=1.5,
):
    sns.set_context(context)
    plt.rc("font", size=FONT_LARGE)
    plt.rc("axes", labelsize=FONT_LARGE, titlesize=FONT_LARGE)
    plt.rc("xtick", labelsize=FONT_LARGE)
    plt.rc("ytick", labelsize=FONT_LARGE)
    plt.rc("legend", fontsize=FONT_XTINY)

    methods = [_ for _ in df.columns.tolist() if _.lower() != "nnz"]
    line_styles = []
    for m in methods:
        m = m.lower()
        if "baseline" in m:
            line_styles.append(":")
        elif m.startswith("hard"):
            line_styles.append("--")
        else:
            line_styles.append(None)

    marker_styles = []
    for m in methods:
        m = m.lower()
        if "baseline" in m:
            marker_styles.append(None)
        elif "proposed" in m:
            marker_styles.append("o")
        elif m.startswith("lottery"):
            marker_styles.append("^")
        elif m.startswith("snip"):
            marker_styles.append("v")
        else:
            marker_styles.append("X")

    # Main chart
    series_name = "Prune method"
    xaxis_name = "Sparsity"
    yaxis_name = score_name
    df2 = df[methods].stack().reset_index(level=1).rename(columns={"level_1": series_name, 0: yaxis_name})
    df2[series_name] = df2[series_name].map(lambda x: "Dense baseline" if x.lower() == "baseline" else x)
    df2.index = df2.index.map(lambda x: f"{x * 100:.1f} %")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.0 * fig_scale, 3.0 * fig_scale))
    # y-axis limit
    if "up-down" in output_path.lower():
        ylim = get_lim(df2.loc[:, yaxis_name], margin=(0.31, 0.05), min_threshold=min_threshold)
    else:
        ylim = get_lim(df2.loc[:, yaxis_name], min_threshold=min_threshold)
    ax.set(ylim=ylim)
    ax = sns.lineplot(
        data=df2,
        x=xaxis_name,
        y=yaxis_name,
        hue=series_name,
        ax=ax,
        palette=palette,
    )
    ax.yaxis.set_major_formatter(SINGLE_DECIMAL_FMT)
    # Lines and legends
    ax = set_style(ax, line_styles, marker_styles)
    legend_xoffset = 0.15 if "soft-" in output_path.lower() and "SNIP" in methods else 0
    ax.legend(loc=plt.rcParams["legend.loc"], bbox_to_anchor=(legend_xoffset, 0))
    # NNZ axis
    df2 = df.set_index("NNZ")[methods]
    df2 = df2.stack().reset_index(level=1).rename(columns={"level_1": series_name, 0: yaxis_name})
    with sns.axes_style(None, rc={"axes.grid": False}):
        # print(sns.axes_style())
        ax2 = ax.twiny()
        sns.lineplot(data=df2, x="NNZ", y=yaxis_name, hue=series_name, ax=ax2, legend=None, visible=False)
    # Title
    if fig_title:
        ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    despine_white(fig)
    # Adjust margins and layout
    plt.tight_layout(pad=0.5)
    plt.savefig(process_output_path(output_path), dpi=output_dpi)  # , plt.show()
    plt.clf()
    plt.close("all")


def plot_smp_progression(
    df,
    palette,
    output_path,
    fig_title="",
    output_dpi=600,
    linewidth=2.0,
    context="paper",
    fig_scale=1.5,
):
    sns.set_context(context)
    plt.rc("font", size=FONT_LARGE)
    plt.rc("axes", labelsize=FONT_LARGE, titlesize=FONT_LARGE)
    plt.rc("xtick", labelsize=FONT_LARGE)
    plt.rc("ytick", labelsize=FONT_LARGE)
    plt.rc("legend", fontsize=FONT_MEDIUM)

    layers = df.columns.tolist()
    line_styles = []
    for m in layers:
        m = m.lower()
        if "target" in m:
            line_styles.append("--")
        else:
            line_styles.append(None)

    # Main chart
    series_name = "Layer"
    xaxis_name = "Training step"
    yaxis_name = "Value" if any("loss" in _ for _ in layers) else "Sparsity"
    df2 = df.stack().reset_index(level=1).rename(columns={"level_1": series_name, 0: yaxis_name})
    # df2.index = df2.index.map(str)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.0 * fig_scale, 3.0 * fig_scale))
    # ax.set(ylim=get_ylim(df2, yaxis_name, min_threshold=min_threshold))
    ax = sns.lineplot(
        data=df2,
        x=xaxis_name,
        y=yaxis_name,
        hue=series_name,
        ax=ax,
        palette=palette,
        linewidth=linewidth,
    )
    ax.yaxis.set_major_formatter(SINGLE_DECIMAL_FMT)
    ax = set_style(ax, line_styles)
    if yaxis_name == "Sparsity":
        ax.legend(loc="lower right", title=series_name)
    else:
        ax.legend(loc="lower left", title="")
    # Title
    if fig_title:
        ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    despine_white(fig)
    # Adjust margins and layout
    plt.tight_layout(pad=0.5)
    plt.savefig(process_output_path(output_path), dpi=output_dpi)  # , plt.show()
    plt.clf()
    plt.close("all")


def plot_smp_layerwise(
    df,
    palette,
    output_path,
    fig_title="",
    output_dpi=600,
    linewidth=2.0,
    context="paper",
    fig_scale=1.5,
):
    sns.set_context(context)
    plt.rc("font", size=FONT_SMALL)
    plt.rc("axes", labelsize=FONT_SMALL, titlesize=FONT_MEDIUM)
    plt.rc("xtick", labelsize=FONT_SMALL)
    plt.rc("ytick", labelsize=FONT_SMALL)
    plt.rc("legend", fontsize=FONT_TINY)

    layers = df.columns.tolist()
    line_styles = []
    for m in layers:
        m = m.lower()
        if "hard" in m:
            line_styles.append("--")
        else:
            line_styles.append(None)

    # Main chart
    series_name = "Method"
    xaxis_name = "Layer"
    yaxis_name = "Sparsity"
    df2 = df.stack().reset_index(level=1).rename(columns={"level_1": series_name, 0: yaxis_name})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.0 * fig_scale, 3.0 * fig_scale))
    ax = sns.lineplot(
        data=df2,
        x=xaxis_name,
        y=yaxis_name,
        hue=series_name,
        ax=ax,
        palette=palette,
        linewidth=linewidth,
    )
    ax = set_style(ax, line_styles)
    # Group Inception layers
    if "lstm" in output_path.lower():
        xticklabels = ["Embedding", "Query", "Key", "Value", "QK", "Initial state", "LSTM", "Output"]
        # prevent UserWarning: FixedFormatter should only be used together with FixedLocator
        # https://stackoverflow.com/a/68794383
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, fontsize="small")
    else:
        xticks = []
        xticklabels = []
        layers = set()
        for i, ly in enumerate(df.index.tolist()):
            ly = ly.split("/")[0].split("_")[1]
            if ly not in layers:
                xticks.append(i)
                xticklabels.append(ly)
            layers.add(ly)
        ax.set_xticks(xticks)
        rotation = 90 if "inception" in output_path.lower() else 0
        fontsize = "x-small" if "inception" in output_path.lower() else "small"
        ax.set_xticklabels(xticklabels, rotation=rotation, fontsize=fontsize)
    if "inception" in output_path.lower():
        ax.legend(loc="lower right", title=series_name)
    # Title
    if fig_title:
        ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    despine_white(fig)
    # Adjust margins and layout
    plt.tight_layout(pad=0.5)
    plt.savefig(process_output_path(output_path), dpi=output_dpi)  # , plt.show()
    plt.clf()
    plt.close("all")


def plot_smp_overview(
    df,
    palette,
    output_path,
    fig_title="",
    output_dpi=600,
    context="paper",
    fig_scale=1.5,
):
    # https://datavizpyr.com/how-to-make-bubble-plot-with-seaborn-scatterplot-in-python/
    sns.set_context(context)
    plt.rc("font", size=FONT_SMALL)
    plt.rc("axes", labelsize=FONT_SMALL, titlesize=FONT_MEDIUM)
    plt.rc("xtick", labelsize=FONT_SMALL)
    plt.rc("ytick", labelsize=FONT_SMALL)

    # Main chart
    series_name = "Method"
    size_name = "Decoder Size (MB)"
    xaxis_name = "Decoder NNZ (M)"
    yaxis_name = "CIDEr"
    sizes = (20, 600)
    df2 = df

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.0 * fig_scale, 3.0 * fig_scale))
    ax.set(xlim=get_lim(df2.index, margin=(0.1, 0.1)), ylim=get_lim(df2.loc[:, yaxis_name], margin=(0.2, 0.2)))
    # Bubble plot
    ax = sns.scatterplot(
        data=df2,
        x=xaxis_name,
        y=yaxis_name,
        size=size_name,
        hue=series_name,
        palette=palette,
        linewidth=0,
        sizes=sizes,
        alpha=1,
        ax=ax,
        legend="full",
    )
    # Line
    ax = sns.lineplot(
        data=df2,
        x=xaxis_name,
        y=yaxis_name,
        hue=series_name,
        linewidth=0.8,
        linestyle=":",
        alpha=0.9,
        ax=ax,
        palette=palette,
        legend=None,
    )

    # Annotate
    for i in range(0, len(df2)):
        x_offset = 0
        y_offset = math.sqrt(df2[size_name].iloc[i] / math.pi)
        if "99.1" in df2["Annotation"].iloc[i]:
            x_offset = -1.5
            y_offset = -y_offset - 6.5
        elif "95" in df2["Annotation"].iloc[i]:
            x_offset = 2.5
        # Size in MB
        ax.annotate(
            f"{df2[size_name].iloc[i]} MB",
            (df2.index[i] + x_offset, df2[yaxis_name].iloc[i] + y_offset / 6),
            fontsize="x-small",
            va="bottom",
            ha="center",
        )
    ax.annotate("8.7 MB (fp16)", (1, 117.5), fontsize="x-small", va="bottom", ha="center")
    ax.annotate("14.5 MB (fp16)", (1, 121.5), fontsize="x-small", va="bottom", ha="center")
    ax.annotate(
        "Pruned to 95% and 99.1% sparsities\nusing proposed Supermask Pruning",
        (25.5, 126),  # (10, 116),
        fontsize="small",
        linespacing=1.3,
        va="bottom",
        ha="center",
        color=cranberry3[1],
    )
    ax.annotate(
        "Dense (original)",
        (48, 121.5),
        fontsize="small",
        linespacing=1.5,
        va="bottom",
        ha="center",
        color=cranberry3[2],
    )

    # ax = set_style(ax, line_styles)
    hdl, lbl = ax.get_legend_handles_labels()
    size_idx = lbl.index(size_name)
    # https://stackoverflow.com/a/53438726
    # config A
    method_legend = ax.legend(
        hdl[:size_idx],
        lbl[:size_idx],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
    )
    size_legend = ax.legend(
        hdl[size_idx::2],
        lbl[size_idx::2],
        ncol=5,
        loc="lower center",
        borderpad=1,
        bbox_to_anchor=(0.5, -0.33),
    )
    ax.add_artist(method_legend)

    # Title
    if fig_title:
        ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    despine_white(fig)
    # Adjust margins and layout
    plt.tight_layout(pad=1.0)
    plt.savefig(process_output_path(output_path), dpi=output_dpi)  # , plt.show()
    plt.clf()
    plt.close("all")


def plot_comic_overview(
    df,
    palette,
    output_path,
    fig_title="",
    output_dpi=600,
    context="paper",
    fig_scale=1.5,
):
    # https://datavizpyr.com/how-to-make-bubble-plot-with-seaborn-scatterplot-in-python/
    sns.set_context(context)

    # Main chart
    series_name = "Method"
    xaxis_name = "Vocab size"
    yaxis_name = "CIDEr"

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.0 * fig_scale, 3.0 * fig_scale))
    ax.set(xlim=get_lim(df.index, margin=(0.1, 0.1)), ylim=get_lim(df.loc[:, yaxis_name], margin=(0.2, 0.2)))
    # Bubble plot
    palette = sns.color_palette(palette, n_colors=len(df[series_name]))
    palette[pd.Index(df[series_name]).get_loc("Baseline")] = mako3[0]
    ax = sns.scatterplot(
        data=df,
        x=xaxis_name,
        y=yaxis_name,
        hue=series_name,
        palette=palette,
        linewidth=0,
        s=100,
        alpha=1,
        ax=ax,
        legend="full",
    )
    # Line
    df2 = df[df["COMIC"] == "Yes"]
    ax = sns.lineplot(
        data=df2,
        x=xaxis_name,
        y=yaxis_name,
        hue="COMIC",
        linewidth=1.0,
        linestyle=":",
        alpha=0.9,
        ax=ax,
        palette=cranberry3[:1],
        legend=None,
    )
    for m in filter(lambda x: "proposed" in x, df[series_name]):
        _df = df2[df2[series_name] == m]
        vsize = _df.index.values[0]
        ax.annotate(
            f"Vocab size = {int(vsize):d}",
            (vsize + 50, _df[yaxis_name] + 1.6),
            fontsize="small",
            va="bottom",
            ha="center",
        )
    ax.annotate(
        "Proposed", (1800, 98), fontsize="medium", linespacing=1.5, va="bottom", ha="center", color=cranberry3[1]
    )
    ax.annotate(
        "Baseline & SOTA", (8000, 98), fontsize="medium", linespacing=1.5, va="bottom", ha="center", color=cranberry3[2]
    )

    # Title
    if fig_title:
        ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    despine_white(fig)
    # Adjust margins and layout
    plt.tight_layout(pad=1.5)
    plt.savefig(process_output_path(output_path), dpi=output_dpi)  # , plt.show()
    plt.clf()
    plt.close("all")


def plot_acort_overview(
    df,
    palette,
    output_path,
    fig_title="",
    output_dpi=600,
    context="paper",
    fig_scale=1.3,
):
    # https://datavizpyr.com/how-to-make-bubble-plot-with-seaborn-scatterplot-in-python/
    sns.set_context(context)

    # Main chart
    series_name = "Method"
    xaxis_name = "Params (M)"
    yaxis_name = "CIDEr"

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.0 * fig_scale, 3.0 * fig_scale))
    ax.set(xlim=get_lim(df.index, margin=(0.1, 0.1)), ylim=get_lim(df.loc[:, yaxis_name], margin=(0.2, 0.2)))
    # Bubble plot
    palette = sns.color_palette(palette, n_colors=len(df[series_name]))
    palette[pd.Index(df[series_name]).get_loc("Baseline")] = mako3[0]
    ax = sns.scatterplot(
        data=df,
        x=xaxis_name,
        y=yaxis_name,
        hue=series_name,
        palette=palette,
        linewidth=0,
        s=100,
        alpha=1,
        ax=ax,
        legend="full",
    )
    # Line
    df2 = df[df["ACORT"] == "Yes"]
    ax = sns.lineplot(
        data=df2,
        x=xaxis_name,
        y=yaxis_name,
        hue="ACORT",
        linewidth=1.0,
        linestyle=":",
        alpha=0.9,
        ax=ax,
        palette=cranberry3[:1],
        legend=None,
    )
    for m in df[series_name]:
        _df = df[df[series_name] == m]
        size = _df.index.values[0]
        ax.annotate(f"{size} M", (size + 1, _df[yaxis_name] + 1), fontsize="small", va="bottom", ha="center")
    ax.annotate(
        "Proposed", (15, 123), fontsize="medium", linespacing=1.5, va="bottom", ha="center", color=cranberry3[1]
    )
    ax.annotate(
        "Baseline & SOTA",
        (40, 120.5),
        fontsize="medium",
        linespacing=1.5,
        va="bottom",
        ha="center",
        color=cranberry3[2],
    )

    # Title
    if fig_title:
        ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    despine_white(fig)
    # Adjust margins and layout
    plt.tight_layout(pad=1.5)
    plt.savefig(process_output_path(output_path), dpi=output_dpi)  # , plt.show()
    plt.clf()
    plt.close("all")


def main():
    d = os.path.join("plot_data", "performance")
    for f in tqdm(sorted(os.listdir(d))):
        min_threshold = 0.8
        fname_low = f.lower()
        if "inception" in fname_low:
            # This must be first condition
            palette = ["#9b59b6", *cranberry3, *mako3, mako3[1], mako3[2]]
            min_threshold = 0.5
        elif "soft-attention" in fname_low or "ort" in fname_low:
            palette = ["#9b59b6", cranberry3[0], flare3[0], mako3[2], *mako3, "#9b59b6"]
        elif "up-down" in fname_low:
            palette = ["#9b59b6", cranberry3[0], *flare3, mako3[2], *mako3, "#9b59b6"]
        else:
            raise ValueError(f"Invalid file: {f}")
        df = pd.read_csv(os.path.join(d, f), sep="\t", header=0, index_col=0)
        fname = os.path.splitext(f)[0]
        title, metric = fname.split(" --- ")
        plot_smp_performance(df, palette, metric, f"{fname}.png", min_threshold=min_threshold)

    d = os.path.join("plot_data", "progression")
    for f in tqdm(sorted(os.listdir(d))):
        df = pd.read_csv(os.path.join(d, f), sep="\t", header=0, index_col=0)
        fname = os.path.splitext(f)[0]
        plot_smp_progression(df, "deep", f"{fname}.png", linewidth=0.8)

    d = os.path.join("plot_data", "layerwise")
    for f in tqdm(sorted(os.listdir(d))):
        fname_low = f.lower()
        if "mobilenet" in fname_low:
            palette = [cranberry3[0], cranberry3[1]]
        else:
            palette = [cranberry3[0], mako3[2], mako3[0]]
        df = pd.read_csv(os.path.join(d, f), sep="\t", header=0, index_col=0)
        fname = os.path.splitext(f)[0]
        plot_smp_layerwise(df, palette, f"{fname}.png", linewidth=0.8)

    for f in tqdm(range(1)):
        # Just for the progress bar
        fname = "Pruning Image Captioning Models (MS-COCO)"
        df = pd.read_csv(os.path.join("plot_data", f"{fname}.tsv"), sep="\t", header=0, index_col=0)
        plot_smp_overview(df, "icefire", f"{fname}.png")

    for f in tqdm(range(1)):
        # Just for the progress bar
        fname = "COMIC (MS-COCO) (fine-tune)"
        df = pd.read_csv(os.path.join("plot_data", f"{fname}.tsv"), sep="\t", header=0, index_col=0)
        plot_comic_overview(df, "icefire_r", f"{fname}.png")

    for f in tqdm(range(1)):
        # Just for the progress bar
        fname = "ACORT (MS-COCO)"
        df = pd.read_csv(os.path.join("plot_data", f"{fname}.tsv"), sep="\t", header=0, index_col=0)
        plot_acort_overview(df, "icefire_r", f"{fname}.png")


if __name__ == "__main__":
    main()
