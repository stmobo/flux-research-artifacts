#!/usr/bin/env python3

from __future__ import print_function

import os
import math
import argparse
import logging
from datetime import timedelta

os.environ["NUMEXPR_MAX_THREADS"] = "16"
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics

from modules import util
from modules.util import pretty_log_df
from modules import modelling
from modules.modelling import trim_max_jobs_per_leaf


def main():
    real_dfs = [pd.read_pickle(real_pkl) for real_pkl in args.real_pkls]
    real_df = pd.concat(real_dfs)
    real_df = trim_max_jobs_per_leaf(real_df, args.max_jobs_per_leaf)

    num_nodes = 32
    cores_per_node = 36
    global total_cores
    total_cores = num_nodes * cores_per_node
    real_df = real_df[
        (real_df.num_nodes == num_nodes) & (real_df.cores_per_node == cores_per_node)
    ]
    real_df["throughput_upperbound"] = real_df[["unique_id", "num_jobs"]].apply(
        lambda x: modelling.calc_upperbound(x["unique_id"], x["num_jobs"], total_cores),
        axis=1,
    )
    real_df.loc[real_df["unique_id"] == "stream", "throughput_upperbound"] = float(
        "inf"
    )

    real_df.unique_id.replace(
        inplace=True,
        to_replace={
            "sleep0": "Sleep 0",
            "sleep5": "Sleep 5",
            "firestarter": "Firestarter",
            "stream": "Stream",
        },
    )
    real_df.rename(
        inplace=True, columns={"unique_id": "Application", "num_levels": "Topology"}
    )

    max_df = real_df.groupby(["Application", "Topology"]).agg("max").reset_index()
    max_df.rename(inplace=True, columns={"throughput": "Throughput"})

    def calc_app_speedup(app_df):
        app_df["1level_throughput"] = app_df[app_df["Topology"] == 1][
            "Throughput"
        ].iloc[0]
        return app_df

    max_df = max_df.groupby(["Application"]).apply(calc_app_speedup)
    max_df["Speedup"] = max_df["Throughput"] / max_df["1level_throughput"]
    max_df["% of Peak"] = (
        (100 * max_df["Throughput"] / max_df["throughput_upperbound"])
        .apply(lambda x: "{:.1f}".format(x))
        .replace(to_replace={"0.0": "-"})
    )
    max_df["Throughput"] = max_df["Throughput"].apply(lambda x: "{:.1f}".format(x))

    model_df = max_df.pivot(
        index="Topology",
        columns="Application",
        values=["Throughput", "% of Peak"],
    )
    model_df = model_df.swaplevel(axis=1).sort_index(axis=1).reindex(["Sleep 0", "Sleep 5", "Firestarter", "Stream"], axis=1, level=-2).reindex(["Throughput", "% of Peak"], axis=1, level=-1)
    #model_df = model_df[
    #    ["Sleep 0", "Sleep 5", "Firestarter", "Stream"]
    #]  # re-order to the app order in the paper
    print(model_df.columns)
    print(model_df)
    logger.debug("\n%s", model_df)
    # print(model_df.to_latex(float_format="%.3f", label="tab:{}".format(model_type)))
    print(model_df.to_latex())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("real_pkls", nargs="+")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--max-jobs-per-leaf", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("print_real_stats")

    main()
