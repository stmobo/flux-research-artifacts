#!/usr/bin/env python3

from __future__ import print_function

import re
import os
import json
import math
import argparse
import itertools
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import six.moves
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from modules import util


def add_suffix_to_label(label):
    if label < 1000:
        return label
    elif label < 1000000:
        return "{:d}K".format(int(label // 1024))
    else:
        return "{:d}M".format(int(label // 1024 // 1024))


def setup_fig(fig, ax):
    global YLIM
    if not args.linear:
        ax.set_xscale("log", basex=2)
        ax.set_yscale("log")
        tick_locs = [int(math.pow(2, x)) for x in range(0, 21)]
        # tick_locs = [int(math.pow(2, x)) for x in range(0, 20)]
        plt.xticks(
            tick_locs,
            [add_suffix_to_label(tick) for tick in tick_locs],
            rotation="vertical",
            fontsize=TEXTSIZE,
        )
    elif args.linear:
        if unique_id in ["sleep5", "firestarter"]:
            YLIM = 250
        elif unique_id == "stream":
            YLIM = 55
        plt.xticks(rotation="vertical", fontsize=TEXTSIZE)
    plt.yticks(fontsize=TEXTSIZE)

    ax.set_xlim(0.8, 1048576 * 1.2)
    if args.makespan:
        ax.set_ylim(1, YLIM)
    else:
        ax.set_ylim(0.1, YLIM)
    ax.grid(axis="y", which="major", color="black")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if not args.linear:
        locmin = matplotlib.ticker.LogLocator(
            base=10.0, subs=np.arange(2, 10) * 0.1, numticks=40
        )
        ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Number of jobs (total)", fontweight="bold", fontsize=AXIS_LABELSIZE)
    if not args.no_yaxis:
        if args.makespan:
            ax.set_ylabel("Makespan (sec)", fontweight="bold", fontsize=AXIS_LABELSIZE)
        else:
            ax.set_ylabel(
                "Avg. Throughput (jobs/sec)", fontweight="bold", fontsize=AXIS_LABELSIZE
            )
    ax.tick_params(axis="y", which="minor", left=True, top=True)


columns_to_print = [
    "num_levels",
    "num_jobs",
    "num_cores",
    "num_nodes",
    "makespan",
    "throughput",
    "just_setup",
]
hierarchical_columns_to_print = [
    ("num_levels", ""),
    ("num_jobs", ""),
    ("just_setup", ""),
] + [(x, "median") for x in ["num_cores", "num_nodes", "makespan", "throughput"]]


def plot_variable_num_jobs(df, unique_id, fig, ax):
    df = (
        df.groupby(["num_levels", "num_jobs"])
        .agg([np.median, np.amax, np.amin])
        .reset_index()
    )
    if len(df) < 3:
        return []

    df = df.sort_values(["num_levels", "num_jobs"])

    # TODO: look into https://stackoverflow.com/questions/41752309/single-legend-item-with-two-lines
    legend_handles = []
    # for (num_levels, marker, label) in [(1, '-ro', 'Depth-1'), (2, '-b^', 'Depth-2'), (3, '-gs', 'Depth-3')]:
    for (num_levels, marker, label) in [
        (1, "-ro", "1"),
        (2, "-b^", "1x32"),
        (3, "-gs", "1x32x36"),
    ]:
        data = df[df.num_levels == num_levels]
        if len(data) > 0:
            x_values = data["num_jobs"].values
            if args.makespan:
                ys = data["makespan"]
            else:
                ys = data["throughput"]
            y_values = ys["median"].values
            yerr_values = (
                (ys["median"] - ys["amin"]).values,
                (ys["amax"] - ys["median"].values),
            )
            line = plt.errorbar(
                x_values,
                y_values,
                yerr=yerr_values,
                fmt=marker,
                label=label,
                elinewidth=LINEWIDTH,
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
            )
            legend_handles.append(line)

    return legend_handles


def log_range(range_min, range_max, base=2):
    return itertools.takewhile(
        lambda x: x <= range_max,
        [
            math.pow(base, x)
            for x in six.moves.range(
                int(math.floor(math.log(range_min, base))),
                int(math.ceil(math.log(range_max, base)) + 1),
            )
        ],
    )


def agg_by(df, agg_labels, func=np.median):
    return df.groupby(agg_labels).agg([func]).reset_index()


def plot_model(model_df, ax):
    model_df = model_df.sort_values(by=["num_jobs"])
    if args.makespan:
        label = "makespan"
    else:
        label = "throughput"

    for level, color_char in [(1, "r"), (2, "b"), (3, "g")]:
        level_df = model_df[model_df.num_levels == level]
        Xs = level_df["num_jobs"]
        Ys = level_df[label]
        ax.plot(Xs, Ys, "{}--".format(color_char))

    legend_handle = None
    if args.plot_max and model_df["throughput_upperbound"].unique()[0] is not None:
        Xs = model_df["num_jobs"]
        Ys = model_df["throughput_upperbound"]
        lines_2d = ax.plot(Xs, Ys, "k--", label="Limit")
        assert len(lines_2d) == 1
        legend_handle = lines_2d.pop()

    return legend_handle


def pretty_print_df(df):
    suffix_blacklist = ["_dir", "_str", "_arg"]
    prefix_blacklist = ["slurm_"]
    # name_blacklist = ["job_name", "command"]
    name_blacklist = ["job_name"]

    def blacklisted(string):
        for suffix in suffix_blacklist:
            if string.endswith(suffix):
                return True
        for prefix in prefix_blacklist:
            if string.startswith(prefix):
                return True
        for name in name_blacklist:
            if string == name:
                return True
        return False

    keys = df.keys()
    printable_keys = [k for k in keys if not blacklisted(k)]
    print(df[printable_keys])


def main():
    real_df = pd.read_pickle(args.df_pkl)
    empirical_model_df = pd.read_pickle(args.empirical_model_pkl)
    empirical_model_df = empirical_model_df[
        empirical_model_df.model_type == args.model_type
    ]
    assert len(empirical_model_df) > 0

    real_df = real_df[real_df.jobs_per_leaf <= args.max_jobs_per_leaf]

    global unique_id
    unique_id = util.unique_id(real_df)
    assert util.unique_id(empirical_model_df) == unique_id

    num_nodes = 32
    cores_per_node = 36
    global total_cores
    total_cores = num_nodes * cores_per_node
    subset_df = real_df[
        (real_df.num_nodes == num_nodes) & (real_df.cores_per_node == cores_per_node)
    ]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    setup_fig(fig, ax)

    legend_handles = plot_variable_num_jobs(subset_df, unique_id, fig, ax)
    legend_handle = plot_model(empirical_model_df, ax)
    if legend_handle:
        legend_handles.append(legend_handle)

    if args.linear:
        loc = "lower right"
    else:
        loc = "upper left"
    legend = plt.legend(
        handles=legend_handles,
        loc=loc,
        title="Tree:",
        frameon=True,
        framealpha=1,
        title_fontsize=TEXTSIZE,
        prop={"size": TEXTSIZE},
    )
    legend.get_title().set_weight("bold")

    if args.nofig:
        pass
    elif args.savefig is not None:
        plt.tight_layout()
        plt.savefig(args.savefig, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("df_pkl")
    parser.add_argument("empirical_model_pkl")
    parser.add_argument("--savefig", type=str)
    parser.add_argument("--nofig", action="store_true")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--makespan", action="store_true")
    parser.add_argument("--plot_max", action="store_true")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--no-yaxis", action="store_true")
    parser.add_argument("--noop", action="store_true") # only exists to make the wrapper shell that uses gnu parallel happier
    parser.add_argument("--max-jobs-per-leaf", type=int, default=1024)
    parser.add_argument("--model-type", type=str, default="empirical")
    args = parser.parse_args()

    if args.plot_max and args.makespan:
        print("Incompatible args: plot_max, makespan.  Disabling plot_max")
        args.plot_max = False

    if args.large:
        LINEWIDTH = 3
        MARKERSIZE = 10
        TEXTSIZE = "large"
        AXIS_LABELSIZE = "x-large"
        FIG_HEIGHT = 4.8
        FIG_WIDTH = 7
    elif args.paper:
        LINEWIDTH = 1.5
        MARKERSIZE = 5
        TEXTSIZE = "x-small"
        AXIS_LABELSIZE = "small"
        FIG_HEIGHT = 2.5
        FIG_WIDTH = 3
    else:
        LINEWIDTH = 2
        MARKERSIZE = 7
        TEXTSIZE = "medium"
        AXIS_LABELSIZE = "large"
        FIG_HEIGHT = 4
        FIG_WIDTH = 5.2
    YLIM = 20000
    if not args.makespan:
        if not args.linear:
            YLIM = 5000
        elif args.linear:
            YLIM = 1300

    main()
