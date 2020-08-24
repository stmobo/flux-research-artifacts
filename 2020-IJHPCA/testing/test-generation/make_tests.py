#!/usr/bin/env python3

import os
import shutil
import argparse
import json
import math
import operator
import glob
import hashlib
from string import Template
from typing import List

RDL_DIRNAME = "rdls"
HFILE_DIRNAME = "hfiles"
LOG_DIRNAME = "logs"
PERSIST_DIRNAME = "persist"
RESULTS_DIRNAME = "results"

TEMPLATE_EXTENSION = "template"
TEST_CONFIG_FILENAME = "test_config.json"


pretty_print_json_kwargs = {"indent": 4, "separators": (",", ": ")}


def rchop(string, suffix):
    assert string.endswith(suffix)
    return string[: -len(suffix)]


def make_test(
    topology: List[int],
    num_jobs,
    num_nodes,
    cores_per_node,
    repetition,
    template_dir,
    output_dir,
    job_gen_type,
    timelimit,
    unique_id=None,
    dry_run=True,
    force=False,
    verbose=False,
    **kwargs,
):
    """
    kwargs include:
        "tasklist_path": just the filename of the task list (should be in template dir)
    """

    direct = job_gen_type == "direct"
    tasklist = job_gen_type == "tasklist"
    runtime = job_gen_type == "runtime"

    if direct:
        assert "command" in kwargs
    if tasklist:
        assert "tasklist_path" in kwargs

    if not unique_id:
        print("WARN: No name given, using 'unnamed'")
        unique_id = "unnamed"

    if "prologue" not in kwargs:
        kwargs["prologue"] = ""

    topology_str = "x".join([str(x) for x in topology])

    ##
    # Make test directory by copying and filling out template directory
    ##
    num_cores = num_nodes * cores_per_node
    job_name = "{}-{}topo-{:07d}jobs-{:04d}cores-{}-repetition{:02d}".format(
        unique_id, topology_str, num_jobs, num_cores, job_gen_type, repetition
    )
    new_dirname = os.path.join(output_dir, job_name)

    if os.path.isdir(new_dirname) and not force:
        if verbose:
            print("Skipping, it already exists".format(new_dirname))
        return

    print("Processing {}".format(new_dirname))
    if os.path.isdir(new_dirname) and force:
        if dry_run:
            print("\tWould remove existing directory {}".format(new_dirname))
        else:
            print("\tRemoving existing directory {}".format(new_dirname))
            shutil.rmtree(new_dirname)
            # for root, dirs, files in os.walk(new_dirname, topdown=False):
            #     for name in files:
            #         os.remove(os.path.join(root, name))
            #     for name in dirs:
            #         if name != new_dirname:
            #             os.rmdir(os.path.join(root, name))
    if not dry_run:
        print("\tCopying {} to {}".format(template_dir, new_dirname))
        shutil.copytree(template_dir, new_dirname)  # , dirs_exist_ok=True)
    elif dry_run and verbose:
        print("\tWould copy {} to {}".format(template_dir, new_dirname))

    test_dir = os.path.abspath(new_dirname)
    log_dir = os.path.join(test_dir, LOG_DIRNAME)
    persist_dir = os.path.join(test_dir, PERSIST_DIRNAME)
    results_dir = os.path.join(test_dir, RESULTS_DIRNAME)
    for directory in [log_dir, persist_dir, results_dir]:
        try:
            if not dry_run:
                os.mkdir(directory)
        except OSError:
            pass

    if tasklist:
        assert "tasklist_path" in kwargs
        # make sure it actually exists
        assert os.path.isfile(kwargs["tasklist_path"])
        # grab just the filename, then join with test dir
        tasklist_filename = os.path.basename(kwargs["tasklist_path"])
        kwargs["tasklist_path"] = os.path.join(test_dir, tasklist_filename)
        # extra cautious check that it got copied to the test dir
        if not dry_run:
            assert os.path.isfile(kwargs["tasklist_path"])

    if len(topology) > 0:
        topology_arg = "--topology={}".format(topology_str)
        leaf_arg = ""
    elif len(topology) == 0:
        topology_arg = ""
        leaf_arg = "--leaf"

    num_nodes = int(math.ceil(num_cores / float(cores_per_node)))
    # Create the test configuration dictionary and save it to disk
    test_dict = {
        "topology": topology,
        "topology_str": topology_str,
        "topology_arg": topology_arg,
        "leaf_arg": leaf_arg,
        "num_jobs": num_jobs,
        "num_nodes": num_nodes,
        "cores_per_node": cores_per_node,
        "repetition": repetition,
        "job_gen_type": job_gen_type,
        "timelimit": timelimit,
        "job_name": job_name,
        "unique_id": unique_id,
        "runtime": runtime,
        "direct": direct,
        # 'tasklist' is included in kwargs
        "slurm_output_file": os.path.join(test_dir, "slurm.out"),
        "log_dir": log_dir,
        "results_dir": results_dir,
        "persist_dir": persist_dir,
        "test_dir": test_dir,
    }

    if len(kwargs) > 0:
        test_dict.update(kwargs)

    json_config_filename = os.path.join(new_dirname, TEST_CONFIG_FILENAME)
    if not dry_run:
        with open(json_config_filename, "w") as outfile:
            json.dump(test_dict, outfile, **pretty_print_json_kwargs)
    elif dry_run and verbose:
        print(
            "\tWould populate {} with:\n{}".format(
                json_config_filename, json.dumps(test_dict, **pretty_print_json_kwargs)
            )
        )

    templates_to_concretize = set(
        glob.glob(
            os.path.join(new_dirname, "**", "*.{}".format(TEMPLATE_EXTENSION)),
            recursive=True,
        )
    )

    # Use the test dict to convert the info template into a concrete info file
    for template_filepath in templates_to_concretize:
        suffix_to_chop = ".{}".format(TEMPLATE_EXTENSION)
        concrete_filepath = rchop(template_filepath, suffix_to_chop)
        if not dry_run:
            with open(template_filepath, "r") as infile:
                info_template = Template(infile.read())
                with open(concrete_filepath, "w") as outfile:
                    outfile.write(info_template.substitute(test_dict))
            os.remove(template_filepath)
        elif dry_run and verbose:
            print(
                "\tWould concretize {} into {}".format(
                    template_filepath, concrete_filepath
                )
            )
            print("\tWould remove {}".format(template_filepath))
    if dry_run and verbose:
        print("\n\n")
