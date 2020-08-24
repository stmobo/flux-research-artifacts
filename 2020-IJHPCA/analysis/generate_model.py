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
    real_df = pd.read_pickle(args.df_pkl)
    model_df = pd.read_pickle(args.model_pkl)

    unique_id = util.unique_id(real_df)
    logger.info("App: %s", unique_id)
    sleep0_df = model_df[
        (model_df.unique_id == "sleep0") & (model_df.num_jobs.isin([1, 32, 36]))
    ]
    setup_df = model_df[(model_df.unique_id == "setup")]
    model_df = model_df[model_df.unique_id == unique_id]

    real_df = trim_max_jobs_per_leaf(real_df, args.max_jobs_per_leaf)
    model_df = trim_max_jobs_per_leaf(model_df, args.max_jobs_per_leaf)
    setup_df = trim_max_jobs_per_leaf(setup_df, args.max_jobs_per_leaf)
    sleep0_df = trim_max_jobs_per_leaf(sleep0_df, args.max_jobs_per_leaf)

    num_nodes = 32
    cores_per_node = 36
    total_cores = num_nodes * cores_per_node
    subset_df = real_df[
        (real_df.num_nodes == num_nodes) & (real_df.cores_per_node == cores_per_node)
    ]
    sched_rate = 4.5
    sched_create_cost = 4.3

    agg_real_df = agg_by(subset_df, ["num_levels", "num_jobs", "just_setup"])
    depth_1_real_df = agg_real_df[agg_real_df.num_levels == 1]
    topologies = real_df["topology"].unique()

    header = """
=========================
====ANALYTICAL MODEL=====
========================="""
    logger.debug(header)
    analytical_model = PurelyAnalyticalModel(
        sched_rate=sched_rate,
        sched_create_cost=sched_create_cost,
        job_runtime=modelling.get_avg_runtime(unique_id),
        resource_cap=total_cores,
        logger=logger
    )
    analytical_model_df = pd.concat(
        [
            analytical_model.get_interpolated_predictions(topology)
            for topology in topologies
        ]
    )
    analytical_model_df["model_type"] = "analytical"
    logger.info(
        "Analytical model cost (single rep): {:.2f} wall second, {:.2f} node seconds".format(
            *analytical_model.calc_model_cost()
        )
    )

    header = """
====================================
=ANALYTICAL MODEL w/ CONTENTION=====
===================================="""
    logger.debug(header)
    analyticalWithContention_model = AnalyticalModelContentedRuntime(
        sched_rate=sched_rate,
        sched_create_cost=sched_create_cost,
        resource_cap=total_cores,
        avg_runtime_func=functools.partial(modelling.get_avg_runtime, unique_id),
        cores_per_node=cores_per_node,
        logger=logger,
    )
    analyticalWithContention_model_df = pd.concat(
        [
            analyticalWithContention_model.get_interpolated_predictions(topology)
            for topology in topologies
        ]
    )
    analyticalWithContention_model_df["model_type"] = "analyticalWithContention"
    logger.info(
        "AnalyticalWithContention model cost (single rep): {:.2f} wall second, {:.2f} node seconds".format(
            *analyticalWithContention_model.calc_model_cost()
        )
    )

#     header = """
# =========================
# ======SIMPLE MODEL=======
# ========================="""
#     logger.debug(header)
#     simple_model = SimpleModel(
#         model_df[model_df.num_nodes == 1],
#         setup_df[(setup_df.num_nodes == 1) & (setup_df.num_levels == 1)],
#         sleep0_df[sleep0_df.num_nodes == 1],
#     )
#     simple_model_df = pd.concat(
#         [simple_model.get_interpolated_predictions(topology) for topology in topologies]
#     )
#     simple_model_df["model_type"] = "simple"
#     logger.info(
#         "Simple model cost (single rep): {:.2f} wall second, {:.2f} node seconds".format(
#             *simple_model.calc_model_cost()
#         )
#     )

#     header = """


# =========================
# =====EMPIRICAL MODEL=====
# ========================="""
#     logger.debug(header)
#     empirical_model = EmpiricalModel(model_df, setup_df, depth_1_real_df)
#     empirical_model_df = pd.concat(
#         [
#             empirical_model.get_interpolated_predictions(topology)
#             for topology in topologies
#         ]
#     )
#     empirical_model_df["model_type"] = "empirical"
#     logger.info(
#         "Empirical model cost (single rep): {:.2f} wall second, {:.2f} node seconds".format(
#             *empirical_model.calc_model_cost()
#         )
#     )

    new_model_df = pd.concat(
        [
            #simple_model_df,
            #empirical_model_df,
            analytical_model_df,
            analyticalWithContention_model_df,
        ],
        sort=True,
    )
    new_model_df["unique_id"] = unique_id
    new_model_df["throughput_upperbound"] = new_model_df["num_jobs"].apply(
        lambda x: modelling.calc_upperbound(unique_id, x, total_cores)
    )
    util.pretty_log_df(new_model_df, "Final DataFrame", logger)

    logger.info("Saving data to {}".format(args.output_pkl))
    new_model_df.to_pickle(args.output_pkl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("df_pkl")
    parser.add_argument("model_pkl")
    parser.add_argument("output_pkl")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--max-jobs-per-leaf", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("generate_model")

    main()
