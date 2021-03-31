# -*- coding: utf-8 -*-
r"""
Created on 25 Mar 2021 19:34:40
@author: jiahuei

"""
import os
import math
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
cranberry3 = sns.dark_palette("#b2124d", n_colors=3, reverse=True)[:3]
coffee3 = sns.dark_palette("#a6814c", n_colors=4, reverse=True)[:3]
# sns.palplot([
#     *sns.color_palette("OrRd", 3),
#     *sns.dark_palette("#a6814c", n_colors=4, reverse=True),
# ])

sns.set_theme(style="darkgrid", rc={"legend.loc": "lower left", "legend.framealpha": 0.7})


# print(plt.rcParams)


def get_lim(series, margin=(0.10, 0.05), min_threshold=None):
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
    for line, leg_line, ls, m in zip(ax.lines, ax.legend().get_lines(), linestyle, marker):
        if ls is not None:
            line.set_linestyle(ls)
            leg_line.set_linestyle(ls)
        if m is not None:
            line.set_marker(m)
            leg_line.set_marker(m)
    return ax


def plot_performance(
        df, palette,
        score_name, fig_title, output_path,
        output_dpi=600, min_threshold=0.8,
        context="paper", fig_scale=1.5,
):
    sns.set_context(context)
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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4. * fig_scale, 3. * fig_scale))
    ax.set(ylim=get_lim(df.loc[:, yaxis_name], min_threshold=min_threshold))
    ax = sns.lineplot(
        data=df2, x=xaxis_name, y=yaxis_name, hue=series_name, ax=ax, palette=palette,
    )
    ax = set_style(ax, line_styles, marker_styles)
    # NNZ axis
    df2 = df.set_index("NNZ")[methods]
    df2 = df2.stack().reset_index(level=1).rename(columns={"level_1": series_name, 0: yaxis_name})
    with sns.axes_style("darkgrid", rc={"axes.grid": False}):
        # print(sns.axes_style())
        ax2 = ax.twiny()
        sns.lineplot(
            data=df2, x="NNZ", y=yaxis_name, hue=series_name, ax=ax2, legend=None, visible=False
        )
    # Title
    ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    # Adjust margins and layout
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=output_dpi)  # , plt.show()
    plt.clf()
    plt.close("all")


def plot_progression(
        df, palette,
        fig_title, output_path,
        output_dpi=600, linewidth=2.,
        context="paper", fig_scale=1.5,
):
    sns.set_context(context)
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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4. * fig_scale, 3. * fig_scale))
    # ax.set(ylim=get_ylim(df2, yaxis_name, min_threshold=min_threshold))
    ax = sns.lineplot(
        data=df2, x=xaxis_name, y=yaxis_name, hue=series_name, ax=ax,
        palette=palette, linewidth=linewidth,
    )
    ax = set_style(ax, line_styles)
    # Title
    ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    # Adjust margins and layout
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=output_dpi)  # , plt.show()
    plt.clf()
    plt.close("all")


def plot_layerwise(
        df, palette,
        fig_title, output_path,
        output_dpi=600, linewidth=2.,
        context="paper", fig_scale=1.5,
):
    sns.set_context(context)
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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4. * fig_scale, 3. * fig_scale))
    ax = sns.lineplot(
        data=df2, x=xaxis_name, y=yaxis_name, hue=series_name, ax=ax,
        palette=palette, linewidth=linewidth,
    )
    ax = set_style(ax, line_styles)
    # Group Inception layers
    if "lstm" in fig_title.lower():
        xticklabels = [
            "Embedding", "Query", "Key", "Value", "QK", "Initial state", "LSTM", "Output"
        ]
        ax.set_xticklabels(xticklabels, fontsize="x-small")
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
        rotation = 90 if "inception" in fig_title.lower() else 0
        ax.set_xticklabels(xticklabels, rotation=rotation, fontsize="x-small")
    # Title
    ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    # Adjust margins and layout
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=output_dpi)  # , plt.show()
    plt.clf()
    plt.close("all")


def plot_overview(
        df, palette,
        fig_title, output_path,
        output_dpi=600,
        context="paper", fig_scale=1.5,
):
    # https://datavizpyr.com/how-to-make-bubble-plot-with-seaborn-scatterplot-in-python/
    sns.set_context(context)

    # Main chart
    series_name = "Method"
    size_name = "Size (MB)"
    xaxis_name = "NNZ (M)"
    yaxis_name = "CIDEr"
    sizes = (20, 600)
    df2 = df

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4. * fig_scale, 3. * fig_scale))
    ax.set(
        xlim=get_lim(df2.index, margin=(0.1, 0.1)),
        ylim=get_lim(df2.loc[:, yaxis_name], margin=(0.2, 0.2))
    )
    # Bubble plot
    ax = sns.scatterplot(
        data=df2, x=xaxis_name, y=yaxis_name, size=size_name, hue=series_name,
        palette=palette, linewidth=0, sizes=sizes, alpha=0.65, ax=ax, legend="full"
    )
    # Line
    ax = sns.lineplot(
        data=df2, x=xaxis_name, y=yaxis_name, hue=series_name,
        linewidth=0.8, linestyle=":", alpha=0.3,
        ax=ax, palette=palette, legend=None,
    )

    # Annotate
    for i in range(0, len(df2)):
        y_offset = math.sqrt(df2[size_name].iloc[i] / math.pi)
        if "99.1" in df2["Annotation"].iloc[i]:
            y_offset = -y_offset - 6
        # Size in MB
        ax.annotate(
            f"{df2[size_name].iloc[i]} MB",
            (df2.index[i], df2[yaxis_name].iloc[i] + y_offset / 6),
            fontsize="x-small", va="bottom", ha="center"
        )
    annot = "99.1% sparse"
    ax.annotate(
        annot, (8, get_midpoint(df2.loc[df2["Annotation"] == annot, yaxis_name] - 1)),
        fontsize="small", va="bottom", ha="center", color="#676767"
    )
    annot = "95% sparse"
    ax.annotate(
        annot, (10, get_midpoint(df2.loc[df2["Annotation"] == annot, yaxis_name])),
        fontsize="small", va="bottom", ha="center", color="#676767"
    )
    annot = "Dense"
    ax.annotate(
        annot, (46, get_midpoint(df2.loc[df2["Annotation"] == annot, yaxis_name])),
        fontsize="small", va="bottom", ha="center", color="#676767"
    )

    # ax = set_style(ax, line_styles)
    hdl, lbl = ax.get_legend_handles_labels()
    size_idx = lbl.index(size_name)
    # https://stackoverflow.com/a/53438726
    # config A
    method_legend = ax.legend(
        hdl[:size_idx], lbl[:size_idx], ncol=5, loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
    )
    size_legend = ax.legend(
        hdl[size_idx::2], lbl[size_idx::2], ncol=5, loc="lower center", borderpad=1,
        bbox_to_anchor=(0.5, -0.33),
    )
    # # config B
    # method_legend = ax.legend(
    #     hdl[:size_idx], lbl[:size_idx], ncol=5, loc="upper center",
    #     bbox_to_anchor=(0.5, 0.28)
    # )
    # size_legend = ax.legend(
    #     hdl[size_idx::2], lbl[size_idx::2], ncol=5, loc="lower center", borderpad=1,
    #     bbox_to_anchor=(0.5, 0.28)
    # )
    # # config C
    # method_legend = ax.legend(
    #     hdl[:size_idx], lbl[:size_idx], ncol=1, loc="lower left",
    # )
    # size_legend = ax.legend(
    #     hdl[size_idx::2], lbl[size_idx::2], ncol=5, loc="lower center", borderpad=1,
    #     bbox_to_anchor=(0.5, -0.3)
    # )
    # # config D
    # method_legend = ax.legend(
    #     hdl[:size_idx], lbl[:size_idx], ncol=1, loc="lower left",
    # )
    # size_legend = ax.legend(
    #     hdl[size_idx::2], lbl[size_idx::2], ncol=1, loc="center", borderpad=1, labelspacing=1.5,
    #     bbox_to_anchor=(1.2, 0.5)
    # )
    ax.add_artist(method_legend)

    # Title
    ax.set_title(fig_title, pad=plt.rcParams["font.size"] * 1.5)
    # Adjust margins and layout
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=output_dpi)  # , plt.show()
    plt.clf()
    plt.close("all")


def main():
    d = os.path.join("plot_data", "performance")
    for f in tqdm(sorted(os.listdir(d))):
        min_threshold = 0.8
        fname_low = f.lower()
        if "inception" in fname_low:
            # This must be first condition
            palette = [gray3[1], *cranberry3, *mako3, mako3[1], mako3[2]]
            min_threshold = 0.5
        elif "soft-attention" in fname_low or "ort" in fname_low:
            palette = [gray3[1], cranberry3[0], flare3[0], mako3[2], *mako3, "#9b59b6"]
        elif "up-down" in fname_low:
            palette = [gray3[1], cranberry3[0], *flare3, mako3[2], *mako3, "#9b59b6"]
        else:
            raise ValueError(f"Invalid file: {f}")
        df = pd.read_csv(os.path.join(d, f), sep="\t", header=0, index_col=0)
        fname = os.path.splitext(f)[0]
        title, metric = fname.split(" --- ")
        plot_performance(df, palette, metric, title, f"{fname}.png", min_threshold=min_threshold)

    d = os.path.join("plot_data", "progression")
    for f in tqdm(sorted(os.listdir(d))):
        df = pd.read_csv(os.path.join(d, f), sep="\t", header=0, index_col=0)
        fname = os.path.splitext(f)[0]
        plot_progression(df, "deep", fname, f"{fname}.png", linewidth=0.8)

    d = os.path.join("plot_data", "layerwise")
    for f in tqdm(sorted(os.listdir(d))):
        fname_low = f.lower()
        if "mobilenet" in fname_low:
            palette = [cranberry3[0], cranberry3[1]]
        else:
            palette = [cranberry3[0], mako3[2], mako3[0]]
        df = pd.read_csv(os.path.join(d, f), sep="\t", header=0, index_col=0)
        fname = os.path.splitext(f)[0]
        plot_layerwise(df, palette, fname, f"{fname}.png", linewidth=0.8)

    fname = "Pruning Image Captioning Models (MS-COCO)"
    df = pd.read_csv(os.path.join("plot_data", f"{fname}.tsv"), sep="\t", header=0, index_col=0)
    plot_overview(df, "icefire", fname, f"{fname}.png")


if __name__ == "__main__":
    main()
