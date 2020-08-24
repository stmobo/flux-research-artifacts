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


def main():
    stats_dfs = [pd.read_pickle(stats_pkl) for stats_pkl in args.stats_pkls]
    stats_df = pd.concat(stats_dfs)

    stats_df["num_levels"] = (
        stats_df["num_levels"]
        .fillna(-1)
        .astype(int)
        .replace({-1: "Overall", 1: "1-Level", 2: "2-Level", 3: "3-Level"})
    )
    stats_df.drop("MAE", 1, inplace=True)
    stats_df.unique_id.replace(
        inplace=True,
        to_replace={
            "sleep0": "Sleep 0",
            "sleep5": "Sleep 5",
            "firestarter": "Firestarter",
            "stream": "Stream",
        },
    )
    stats_df.model_type.replace(
        inplace=True, to_replace={"analyticalWithContention": "hybrid"}
    )
    stats_df.rename(
        inplace=True, columns={"unique_id": "Application", "num_levels": "Topology"}
    )

    for model_type in stats_df.model_type.unique():
        model_df = stats_df[stats_df.model_type == model_type]
        model_df = model_df.drop("model_type", 1)
        model_df = model_df.pivot(index="Topology", columns="Application", values="r2")
        model_df = model_df[
            ["Sleep 0", "Sleep 5", "Firestarter", "Stream"]
        ]  # re-order to the app order in the paper
        logger.debug("%s\n%s", model_type, model_df)
        # print(model_df.to_latex(float_format="%.3f", label="tab:{}".format(model_type)))
        print(model_df.to_latex(float_format="%.3f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_pkls", nargs="+")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("print_model_stats")

    main()
