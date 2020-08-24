#!/usr/bin/env python3

from __future__ import print_function

import re
import os
import json
import math
import argparse
import itertools
import logging
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import six.moves

from modules import util


def add_suffix_to_label(label, base=2):
    if base == 2:
        divisor = 1024
    elif base == 10:
        divisor = 1000
    else:
        raise NotImplementedError()

    if label < 1000:
        return label
    elif label < 1_000_000:
        return "{:d}K".format(int(label // divisor))
    elif label < 1_000_000_000:
        return "{:d}M".format(int(label // math.pow(divisor, 2)))
    elif label < 1_000_000_000_000:
        return "{:d}B".format(int(label // math.pow(divisor, 3)))
    else:
        raise NotImplementedError()


def setup_fig(fig, ax, no_yaxis=False):
    global YLIM
    if not args.linear:
        ax.set_xscale("log", basex=2)
        ax.set_yscale("log")
        xtick_locs = [int(math.pow(4, x)) for x in range(0, 18)]
        plt.xticks(
            xtick_locs,
            [add_suffix_to_label(tick) for tick in xtick_locs],
            rotation="vertical",
            fontsize=TEXTSIZE,
        )
        ytick_locs = [0] + [int(math.pow(10, x)) for x in range(0, 7)]
        plt.yticks(
            ytick_locs,
            [add_suffix_to_label(tick, 10) for tick in ytick_locs],
            rotation="horizontal",
            fontsize=TEXTSIZE,
        )
    elif args.linear:
        if unique_id in ["sleep5", "firestarter"]:
            YLIM = 250
        elif unique_id == "stream":
            YLIM = 55
        plt.xticks(rotation="vertical", fontsize=TEXTSIZE)
        plt.yticks(fontsize=TEXTSIZE)

    #ax.set_xlim(0.8, 1048576 * 1.2)
    ax.set_xlim(0.8, 4500 * 44 * 32768 * 1.4)
    if args.makespan:
        ax.set_ylim(1, YLIM)
    else:
        ax.set_ylim(0.1, YLIM)
    ax.grid(axis="y", which="major", color="black")
    #ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if not args.linear:
        locmin = matplotlib.ticker.LogLocator(
            base=10.0, subs=np.arange(2, 10) * 0.1, numticks=40
        )
        ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Number of jobs (total)", fontweight="bold", fontsize=AXIS_LABELSIZE)
    if not no_yaxis:
        if args.makespan:
            ax.set_ylabel("Makespan (sec)", fontweight="bold", fontsize=AXIS_LABELSIZE)
        else:
            ax.set_ylabel(
                "Avg. Throughput (jobs/sec)", fontweight="bold", fontsize=AXIS_LABELSIZE
            )
    ax.tick_params(axis="y", which="minor", left=True, top=True)

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


def plot_model(model_df, ax, unique_id):
    legend_handles = []
    model_df = model_df.sort_values(by=["num_jobs"])
    if args.makespan:
        label = "makespan"
    else:
        label = "throughput"

    level_color_map = {
        1: "red",
        2: "blue",
        3: "green",
        4: "orange",
    }
    # for level, color_char, line_label in [
    #     (1, "r", "1-Level"),
    #     (2, "b", "2-Level"),
    #     (3, "g", "3-Level"),
    # ]:
    def get_num_levels(topology):
        return len(topology.split('x'))

    def get_topology_order(topology):
        topo = [int(x) for x in topology.split('x')]
        num_levels = len(topo)
        if num_levels == 1:
            return (1, 1)
        if topo[1] == 32:
            return (1, num_levels)
        return (2, num_levels)

    for topology in sorted(model_df['topology'].unique(), key=get_topology_order):
        level_df = model_df[model_df.topology == topology]
        num_levels = get_num_levels(topology)
        if num_levels == 1 or int(topology.split('x')[1]) == 32:
            linestyle = "--"
        else:
            linestyle = ":"
        color = level_color_map[num_levels]
        Xs = level_df["num_jobs"]
        Ys = level_df[label]
        line_2d = ax.plot(Xs, Ys, color=color, linestyle=linestyle, label=topology)
        assert len(line_2d) == 1
        legend_handles.append(line_2d.pop())

    if args.plot_max and unique_id not in ["stream", "sleep0"]:
        Xs = model_df["num_jobs"]
        Ys = model_df["throughput_upperbound"]
        lines_2d = ax.plot(Xs, Ys, "k--", label="Limit")
        assert len(lines_2d) == 1
        legend_handles.append(lines_2d.pop())

    return legend_handles


def main():
    scale_up_df = pd.read_pickle(args.scale_up_df_pkl)
    assert len(scale_up_df) > 0

    for idx, unique_id in enumerate(scale_up_df['unique_id'].unique()):
        app_model_df = scale_up_df[scale_up_df.unique_id == unique_id]

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        setup_fig(fig, ax, no_yaxis=(idx > 0))

        legend_handles = plot_model(app_model_df, ax, unique_id)

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
            title_fontsize=LEGEND_TEXTSIZE,
            prop={"size": LEGEND_TEXTSIZE},
        )
        legend.get_title().set_weight("bold")

        if args.nofig:
            pass
        elif args.savefig is not None:
            plt.tight_layout()
            filename = "{0}_scale_up.pdf".format(unique_id)
            logger.info("Saving to %s", filename)
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scale_up_df_pkl")
    parser.add_argument("--savefig", action="store_true")
    parser.add_argument("--nofig", action="store_true")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--makespan", action="store_true")
    parser.add_argument("--plot_max", action="store_true")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("plot_model_scale_study")

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
        LEGEND_TEXTSIZE = "xx-small"
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
            YLIM = 100000
        elif args.linear:
            YLIM = 1300

    main()
