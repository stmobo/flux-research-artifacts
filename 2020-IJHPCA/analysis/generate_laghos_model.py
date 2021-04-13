#!/usr/bin/env python3

from __future__ import print_function

import os
import argparse
import logging
import functools

os.environ["NUMEXPR_MAX_THREADS"] = "16"
import numpy as np
import pandas as pd

from modules import util
from modules import modelling
from modules.modelling import agg_by, trim_max_jobs_per_leaf
from modules.modelling import PurelyAnalyticalModel, AnalyticalModelContentedRuntime


def main():
    num_nodes = 32
    cores_per_node = 36
    sched_rate = 4.5
    sched_create_cost = 4.3
    tasks_per_job = 6
    topologies = [(1,), (1, 32), (1, 32, 6)]

    # Maximum number of simultaneous jobs that can fit onto a node
    # (assuming each task uses 1 core)
    slots_per_node = cores_per_node // tasks_per_job

    # Maximum number of simultaneous jobs total
    total_slots = num_nodes * slots_per_node

    analytical_model = PurelyAnalyticalModel(
        sched_rate=sched_rate,
        sched_create_cost=sched_create_cost,
        job_runtime=modelling.get_avg_runtime("laghos"),
        resource_cap=total_slots,
        logger=logger
    )

    analyticalWithContention_model = AnalyticalModelContentedRuntime(
        sched_rate=sched_rate,
        sched_create_cost=sched_create_cost,
        resource_cap=total_slots,
        avg_runtime_func=functools.partial(modelling.get_avg_runtime, "laghos"),
        jobs_per_node=slots_per_node,
        logger=logger,
    )

    analytical_model_df = pd.concat(
        [
            analytical_model.get_interpolated_predictions(topology)
            for topology in topologies
        ]
    )
    analytical_model_df["model_type"] = "analytical"

    analyticalWithContention_model_df = pd.concat(
        [
            analyticalWithContention_model.get_interpolated_predictions(topology)
            for topology in topologies
        ]
    )
    analyticalWithContention_model_df["model_type"] = "analyticalWithContention"

    new_model_df = pd.concat(
        [
            analytical_model_df,
            analyticalWithContention_model_df,
        ],
        sort=True,
    )
    new_model_df["unique_id"] = "laghos"
    new_model_df["throughput_upperbound"] = new_model_df["num_jobs"].apply(
        lambda x: modelling.calc_upperbound("laghos", x, total_slots)
    )

    util.pretty_log_df(new_model_df, "Final DataFrame", logger)

    logger.info("Saving data to {}".format(args.output_pkl))
    new_model_df.to_pickle(args.output_pkl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_pkl")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--max-jobs-per-leaf", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("generate_model")

    main()
