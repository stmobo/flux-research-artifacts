#!/usr/bin/env python3

import re
import os
import json
import math
import argparse
import operator
import functools
from datetime import timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

from modules.testdir import parse_simple_times
from modules import util


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
    df = pd.DataFrame.from_records(
        [parse_simple_times(test_dir) for test_dir in args.test_dirs]
    )
    failed_jobs = (df.failed) | (df.makespan < 0)
    if len(failed_jobs) > 0:
        print("The following jobs failed:")
        print(df[failed_jobs]["job_name"])
    else:
        print("No jobs failed")
    df = df[(~failed_jobs)]

    # ensure that topology is hash-able
    df["topology"] = df["topology"].map(lambda x: tuple(x))
    df["num_levels"] = df["topology"].map(lambda x: len(x))
    df["first_branch_factor"] = df.topology.apply(lambda x: -1 if len(x) < 2 else x[1])
    df["second_branch_factor"] = df.topology.apply(lambda x: -1 if len(x) < 3 else x[2])
    df["num_leaves"] = df["topology"].apply(
        lambda x: functools.reduce(operator.mul, x, 1)
    )
    df["jobs_per_leaf"] = df["num_jobs"] / df["num_leaves"]
    df["throughput"] = df["num_jobs"] / df["makespan"]
    df["num_cores"] = df["num_nodes"] * df["cores_per_node"]
    if "just_setup" in df.columns:
        df["just_setup"] = df["just_setup"].fillna(False)
    else:
        df["just_setup"] = False

    print("\nThe parsed dataframe looks like:")
    pretty_print_df(df)

    unique_id = util.unique_id(df)
    filename = "{}_{}.pkl".format(unique_id, args.output_label)
    print("Saving data to {}".format(filename))
    df.to_pickle(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_label", type=str)
    parser.add_argument("test_dirs", nargs="+")
    args = parser.parse_args()

    main()
