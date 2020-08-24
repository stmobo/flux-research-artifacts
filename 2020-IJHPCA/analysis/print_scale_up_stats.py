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

def main():
    scale_up_df = pd.read_pickle(args.scale_up_df_pkl)
    assert len(scale_up_df) > 0

    scale_up_df.loc[scale_up_df["unique_id"] == "stream", "throughput_upperbound"] = float(
        "inf"
    )

    scale_up_df.unique_id.replace(
        inplace=True,
        to_replace={
            "sleep0": "Sleep 0",
            "sleep5": "Sleep 5",
            "firestarter": "Firestarter",
            "stream": "Stream",
        },
    )
    scale_up_df.rename(
        inplace=True, columns={"unique_id": "Application", "topology": "Topology"}
    )

    max_df = scale_up_df.groupby(["Application", "Topology"]).agg("max").reset_index()
    max_df.rename(inplace=True, columns={"throughput": "Throughput"})

    def calc_app_speedup(app_df):
        app_df["1level_throughput"] = app_df[app_df["Topology"] == "1"][
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
    print(model_df.columns)
    print(model_df)
    logger.debug("\n%s", model_df)
    print(model_df.to_latex())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scale_up_df_pkl")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("print_scale_up_stats")

    main()
