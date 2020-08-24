#!/usr/bin/env python3

import re
import os
import json
import math
import argparse
from datetime import timedelta
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker
import numpy as np
import pandas as pd

from modules.testdir import parse_simple_times


def add_suffix_to_label(label):
    if label < 1000:
        return label
    elif label < 1000000:
        return "{:d}K".format(label // 1024)
    else:
        return "{:d}M".format(label // 1024 // 1024)


def plot_variable_num_jobs(df, unique_id, fig, ax):
    df = (
        df.groupby(["num_levels", "num_jobs"])
        .agg([np.mean, np.amax, np.amin])
        .reset_index()
    )
    if len(df) < 3:
        return []

    df = df.sort_values(["num_levels", "num_jobs"])

    ax.set_xscale("log", basex=2)
    ax.set_yscale("log")
    tick_locs = [int(math.pow(2, x)) for x in range(0, 21)]
    plt.xticks(
        tick_locs,
        [add_suffix_to_label(tick) for tick in tick_locs],
        rotation="vertical",
        fontsize=TEXTSIZE,
    )
    plt.yticks(fontsize=TEXTSIZE)

    ax.set_xlim(1, 1048576 * 2)
    ax.set_ylim(1, YLIM)
    ax.grid(axis="y", which="major", color="black")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    locmin = matplotlib.ticker.LogLocator(
        base=10.0, subs=np.arange(2, 10) * 0.1, numticks=40
    )
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Number of jobs (total)", fontweight="bold", fontsize=AXIS_LABELSIZE)
    ax.set_ylabel("Makespan (sec)", fontweight="bold", fontsize=AXIS_LABELSIZE)
    ax.tick_params(axis="y", which="minor", left=True, top=True)

    legend_handles = []
    for (num_levels, marker, label) in [
        (1, "-ro", "Depth-1"),
        (2, "-b^", "Depth-2"),
        (3, "-gs", "Depth-3"),
    ]:
        data = df[df.num_levels == num_levels]
        if len(data) > 0:
            x_values = data["num_jobs"].values
            ys = data["makespan"]
            y_values = ys["mean"].values
            yerr_values = (
                (ys["mean"] - ys["amin"]).values,
                (ys["amax"] - ys["mean"].values),
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
        legend = plt.legend(
            handles=legend_handles,
            loc="upper left",
            title="Scheduler Hierarchy:",
            frameon=True,
            framealpha=1,
            prop={"size": TEXTSIZE},
        )
        legend.get_title().set_weight("bold")
        if num_levels == 1:
            fig.tight_layout()
        filename = "{}_variable_num_jobs_{}.pdf".format(unique_id, num_levels)
        fig.savefig(filename)

    return legend_handles


def pretty_print_df(df):
    suffix_blacklist = ["_dir", "_str", "_arg"]
    prefix_blacklist = ["slurm_"]
    name_blacklist = ["job_name", "command"]

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
    df = pd.read_pickle(args.df_pkl)
    df["num_levels"] = df["topology"].map(lambda x: len(x))
    failed_jobs = (df.failed) | (df.makespan < 0)
    if len(failed_jobs) > 0:
        print("The following jobs failed:")
        print(df[failed_jobs]["job_name"])
    else:
        print("No jobs failed")
    df = df[(~failed_jobs)]
    num_nodes = 32
    df = df[df.num_nodes == num_nodes]

    pretty_print_df(df)

    unique_ids = df["unique_id"].unique()
    assert len(unique_ids) == 1
    unique_id = unique_ids[0]

    if args.export_csv:
        filename = "{}_variable_num_jobs_full.csv".format(unique_id)
        print("Saving data to {}".format(filename))
        df.to_csv(filename)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    legend_handles = plot_variable_num_jobs(df, unique_id, fig, ax)
    # legend = plt.legend(
    #     loc="upper left", title="Scheduler Hierarchy:", frameon=False
    # )
    # legend.get_title().set_weight("bold")
    plt.tight_layout()

    if args.savefig:
        filename = "{}_variable_num_jobs.pdf".format(unique_id)
        print("Saving to {}".format(filename))
        with PdfPages(filename) as pdf:
            for fig in [plt.figure(n) for n in plt.get_fignums()]:
                pdf.savefig(fig)
                plt.close(fig)
    else:
        plt.show()

    if args.export_csv:
        agg_df = (
            df.groupby(["num_levels", "num_jobs"])
            .agg([np.mean, np.amax, np.amin])
            .reset_index()
        )
        csv_df = pd.DataFrame(
            data={
                "num_jobs": agg_df["num_jobs"],
                "label": agg_df["num_levels"],
                "makespan (mean)": agg_df["makespan"]["mean"],
            }
        )
        csv_df["label"] = csv_df["label"].apply(lambda x: "Depth-{}".format(x))
        filename = "{}_variable_num_jobs.csv".format(unique_id)
        print("Saving data to {}".format(filename))
        csv_df.to_csv(filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("df_pkl")
    parser.add_argument("--savefig", action="store_true")
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()

    if args.large:
        LINEWIDTH = 3
        MARKERSIZE = 10
        TEXTSIZE = "large"
        AXIS_LABELSIZE = "x-large"
        YLIM = 10000
        FIG_HEIGHT = 4.8
        FIG_WIDTH = 7
    else:
        LINEWIDTH = 2
        MARKERSIZE = 7
        TEXTSIZE = "medium"
        AXIS_LABELSIZE = "large"
        YLIM = 20000
        FIG_HEIGHT = 3.9
        FIG_WIDTH = 4.7

    main()
