from __future__ import print_function

import os
import re
import json
import warnings
from datetime import datetime

import six
import dateutil.parser
import pytz
from pytz import timezone
import pandas as pd

if six.PY2:
    from memorize import Memorize
    from time_conversion import walltime_str_to_timedelta
elif six.PY3:
    from .memorize import Memorize
    from .time_conversion import walltime_str_to_timedelta

PACIFIC = timezone("US/Pacific")


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def newFunc(*args, **kwargs):
        warnings.warn(
            "Call to deprecated function %s." % func.__name__,
            category=DeprecationWarning,
        )
        return func(*args, **kwargs)

    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc


def get_test_config(test_dir):
    with open(os.path.join(test_dir, "test_config.json"), "r") as conffile:
        test_config = json.load(conffile)
    return test_config


def get_results_dir(test_dir, test_config=None):
    if test_config is None:
        test_config = get_test_config(test_dir)

    try:
        results_dir = test_config["results_dir"]
    except KeyError:
        results_dir = os.path.abspath(
            os.path.join(test_dir, "../results", test_config["job_name"])
        )
    return results_dir


def get_log_dir(test_dir, test_config=None):
    if test_config is None:
        test_config = get_test_config(test_dir)

    try:
        log_dir = test_config["log_dir"]
    except KeyError:
        log_dir = os.path.abspath(
            os.path.join(test_dir, "../logs", test_config["job_name"])
        )
    return log_dir


def gen_jobfile_paths(test_dir, test_config=None, curr_level=0, id_list=None):
    """
    Generates the job file paths for all instances (root + all descendants)
    """
    if test_config is None:
        test_config = get_test_config(test_dir)
    if id_list is None:
        id_list = []

    results_dir = get_results_dir(test_dir, test_config)

    if curr_level == 0:
        instances_at_curr_level = [0]
    else:
        instances_at_curr_level = range(
            1, test_config["num_children"][curr_level - 1] + 1
        )
    for instance_id in instances_at_curr_level:
        new_id_list = id_list + [instance_id]
        job_filename = "job-{}".format(".".join([str(x) for x in new_id_list]))
        if curr_level < test_config["num_levels"] - 1:
            expected_num_children = test_config["num_children"][curr_level]
            for log_tuple in gen_jobfile_paths(
                test_dir, test_config, curr_level + 1, new_id_list
            ):
                yield log_tuple
        else:
            expected_num_children = -1
        yield (
            os.path.join(results_dir, job_filename),
            expected_num_children,
            new_id_list,
        )


def gen_broker_log_paths(test_dir, test_config=None, curr_level=0, id_list=None):
    """
    Genereates the broker log file paths for all instances (root + all descendants)
    """
    if test_config is None:
        test_config = get_test_config(test_dir)
    if id_list is None:
        id_list = []

    log_dir = get_log_dir(test_dir, test_config)

    if curr_level == 0:
        instances_at_curr_level = [0]
    else:
        instances_at_curr_level = range(1, test_config["topology"][curr_level - 1] + 1)
    for instance_id in instances_at_curr_level:
        new_id_list = id_list + [instance_id]
        job_filename = "{}-broker.out".format(".".join([str(x) for x in new_id_list]))
        if curr_level < test_config["num_levels"] - 1:
            expected_num_children = test_config["num_children"][curr_level]
            for log_tuple in gen_broker_log_paths(
                test_dir, test_config, curr_level + 1, new_id_list
            ):
                yield log_tuple
        else:
            expected_num_children = -1
        yield (os.path.join(log_dir, job_filename), expected_num_children, new_id_list)


def all_jobs_completed(test_config):
    results_dir = test_config["results_dir"]

    total_expected_jobs = test_config["num_jobs"]
    total_completed_jobs = 0
    for jobfile_path, expected_num_children, id_list in gen_jobfile_paths(
        results_dir, test_config=test_config
    ):
        if expected_num_children > 0:
            pass
        else:  # is a leaf instance
            with open(jobfile_path, "r") as jobfile_fd:
                jobfile_lines = jobfile_fd.readlines()
                total_completed_jobs += len(jobfile_lines) - 1

    return total_completed_jobs == total_expected_jobs


job_list = None


def gen_currently_running_jobnames():
    global job_list
    if job_list is None:
        try:
            import pyslurm

            job_list = pyslurm.job().find_user("herbein1").items()
        except ImportError:
            return
            raise StopIteration
    for jobid, jobdict in job_list:
        yield jobdict["name"]


def parse_simple_times(test_dir):
    test_config = get_test_config(test_dir)
    memorized = Memorize(test_dir)(_parse_simple_times)
    return memorized(test_dir)


def _parse_simple_times(test_dir):
    assert os.path.isdir(test_dir)

    test_config = get_test_config(test_dir)
    results_dir = get_results_dir(test_dir, test_config)

    # make command hashable (compatibilty with equality in pandas' queries)
    # e.g., subset_df = df[(df.command == command) & (df.num_nodes == num_nodes)]
    # if test_config["command"] is not None:
    #     test_config["command"] = tuple(test_config["command"])

    test_config["failed"] = True
    test_config["makespan"] = -1

    try:
        perf_filename = os.path.join(results_dir, "perf.out")
        df = pd.read_csv(perf_filename, delim_whitespace=True, header=0)
        top_instance = df[df.TreeID == "tree"]
        if "Begin(Epoch)" in df:
            makespan = top_instance["End(Epoch)"] - top_instance["Begin(Epoch)"]
        else:
            makespan = top_instance["End"] - top_instance["Begin"]
        assert len(makespan) == 1
        test_config["makespan"] = makespan.iloc[0]
        test_config["failed"] = False
    except Exception as e:
        print(
            "Exception occured when processing test at {}: {}".format(test_dir, str(e))
        )
        return test_config

    return test_config

    # timelimit_td = walltime_str_to_timedelta(test_config["timelimit"])
