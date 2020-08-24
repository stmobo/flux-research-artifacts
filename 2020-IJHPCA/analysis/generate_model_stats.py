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
from modules.modelling import agg_by, trim_max_jobs_per_leaf


def calculate_metric(real_df, pred_df, metric_func):
    real_df = agg_by(real_df, ["num_levels", "num_jobs"]).droplevel(1, axis=1)[
        ["num_levels", "num_jobs", "throughput"]
    ]
    pred_df = pred_df[["num_levels", "num_jobs", "throughput", "model_type"]]
    merged_df = real_df.merge(
        pred_df, on=["num_levels", "num_jobs"], how="left", suffixes=("_real", "_pred")
    )
    num_levels = merged_df.num_levels.unique()
    values = []
    for nl in num_levels:
        level_df = merged_df[merged_df.num_levels == nl]
        values.append(
            metric_func(level_df["throughput_real"], level_df["throughput_pred"])
        )
    num_levels = np.append(num_levels, np.NaN)
    values.append(
        metric_func(merged_df["throughput_real"], merged_df["throughput_pred"])
    )
    return pd.DataFrame(data={"num_levels": num_levels, "metric": values})


def calculate_r2(real_y, pred_y):
    residual_sum_of_squares = np.power((real_y - pred_y), 2).sum()
    real_y_mean = np.mean(real_y)
    total_sum_of_squares = np.power((real_y - real_y_mean), 2).sum()
    return 1 - (residual_sum_of_squares / total_sum_of_squares)


def main():
    real_df = pd.read_pickle(args.df_pkl)
    model_df = pd.read_pickle(args.model_pkl)

    unique_id = util.unique_id(real_df)
    assert unique_id == util.unique_id(model_df)

    logger.info("App: %s", unique_id)

    real_df = trim_max_jobs_per_leaf(real_df, args.max_jobs_per_leaf)
    # model_df = trim_max_jobs_per_leaf(model_df, args.max_jobs_per_leaf)

    num_nodes = 32
    cores_per_node = 36
    global total_cores
    total_cores = num_nodes * cores_per_node
    real_df = real_df[
        (real_df.num_nodes == num_nodes) & (real_df.cores_per_node == cores_per_node)
    ]


    metric_dfs = []
    for metric_func, metric_label in [
        (sklearn.metrics.mean_absolute_error, "MAE"),
        (sklearn.metrics.r2_score, "r2"),
    ]:
        dfs = []
        for model_type in ["analytical", "analyticalWithContention"]:
            df = calculate_metric(
                real_df,
                model_df[model_df.model_type == model_type],
                metric_func=metric_func,
            ).rename(columns={"metric": metric_label})
            df["model_type"] = model_type
            logger.debug("%s model's %s:\n%s", model_type, metric_label, df)
            dfs.append(df)
        metric_dfs.append(pd.concat(dfs))

    metric_df = metric_dfs[0]
    for metric_df2 in metric_dfs[1:]:
        metric_df = metric_df.merge(
            metric_df2,
            on=["num_levels", "model_type"],
            how="outer",
            suffixes=(False, False),
        )
    metric_df["unique_id"] = unique_id
    logger.debug("\n%s", metric_df)
    logger.info("Saving to %s", args.output_pkl)
    metric_df.to_pickle(args.output_pkl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("df_pkl")
    parser.add_argument("model_pkl")
    parser.add_argument("output_pkl")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--max-jobs-per-leaf", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("generate_model_stats")

    main()
