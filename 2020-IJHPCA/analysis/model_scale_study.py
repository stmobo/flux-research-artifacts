#!/usr/bin/env python3

from __future__ import print_function

import os
import argparse
import logging
import functools
import itertools
import multiprocessing as mp

os.environ["NUMEXPR_MAX_THREADS"] = "16"
import numpy as np
import pandas as pd

from modules import util
from modules import modelling
from modules.modelling import agg_by, trim_max_jobs_per_leaf
from modules.modelling import AnalyticalModelContentedRuntime


def create_model(unique_id):
    return AnalyticalModelContentedRuntime(
        sched_rate=sched_rate,
        sched_create_cost=sched_create_cost,
        resource_cap=total_cores,
        avg_runtime_func=functools.partial(modelling.get_avg_runtime, unique_id, system="lassen"),
        cores_per_node=cores_per_node,
        max_jobs_per_leaf=32768,
        logger=logger,
    )

def gen_df(topology, unique_id):
    analyticalWithContention_model = create_model(unique_id)
    df = analyticalWithContention_model.get_interpolated_predictions(topology)
    df["unique_id"] = unique_id
    df["topology"] = "x".join([str(x) for x in topology])
    df["throughput_upperbound"] = df["num_jobs"].apply(
        lambda x: modelling.calc_upperbound(unique_id, x, total_cores)
    )
    return df


def parallel_gen_df(tupl):
    return gen_df(*tupl)


def main():
    num_nodes = 4500
    global cores_per_node
    cores_per_node = 44
    global total_cores
    total_cores = num_nodes * cores_per_node
    global sched_rate
    sched_rate = 3.6
    global sched_create_cost
    sched_create_cost = 3.4

    topologies = [
        [1],
        [1, 32],
        #[1, num_nodes],
        [1, 32, 36],
        #[1, 444, 444],
        [1, num_nodes, cores_per_node],
        [1, 55, 60, 60],
    ]
    unique_ids = ["sleep0", "sleep5", "firestarter", "stream"]
    func_args = itertools.product(topologies, unique_ids)
    with mp.Pool() as pool:
        dfs = pool.map(parallel_gen_df, func_args)

    model = create_model("sleep5")
    for topology in topologies:
        create_time = model.get_empty_hierarchy_init_cost(topology)
        logger.info("Topology %s is predicted to take %f seconds to create", topology, create_time)

    new_model_df = pd.concat(dfs, sort=True)
    new_model_df["model_type"] = "analyticalWithContention"
    util.pretty_log_df(new_model_df, "Final DataFrame", logger)

    logger.info("Saving data to {}".format(args.output_pkl))
    new_model_df.to_pickle(args.output_pkl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_pkl")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("model_scale_study")

    main()
