import os
import argparse
import socket
import operator
from functools import reduce


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--dry_run",
        action="store_true",
        help="take no action, just print expected actions",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="will forceably run, removing existing directories",
    )
    parser.add_argument(
        "-u",
        "--unique_id",
        help="unique identifier of this test; appears in test dir names",
        default="sleep0",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--small-scale", action="store_true")
    parser.add_argument("--medium-scale", action="store_true")
    return parser


def get_command(unique_id):
    # NOTE: assumes cwd is 2020-IJHPCA folder
    exe_dir = os.path.join(os.getcwd(), "test-executables")

    if unique_id == "sleep0":
        return "flux mini submit -n1 -c1 sleep 0"
    elif unique_id == "sleep5":
        return "flux mini submit -n1 -c1 sleep 5"
    elif unique_id == "stream":
        return (
            "flux mini submit -n1 -c1 " +
            os.path.join(exe_dir, "stream", "stream_70") +
            " | sed -n '23,27p'"
        )
    elif unique_id == "firestarter":
        return (
            "flux mini submit -n1 -c1 " +
            os.path.join(exe_dir, "FIRESTARTER") +
            " -t 5 -q"
        )
    raise NotImplementedError()


def get_output_dir(test_name, args):
    hostname = socket.gethostname()
    if hostname.startswith("opal"):
        lustre_prefix = "/p/lquake/"
    else:
        raise NotImplementedError()

    output_dir = os.path.abspath(
        os.path.join(lustre_prefix, "herbein1/{}-{}".format(args.unique_id, test_name))
    )
    if args.small_scale:
        output_dir += "-small-scale"
    elif args.medium_scale:
        output_dir += "-medium-scale"
    return output_dir


def get_template_dir(job_gen_type):
    template_dir_prefix = os.path.expanduser(
        "~/Repositories/flux-framework/hierarchical-sched-research/testing/test-template-dirs"
    )
    template_dir = os.path.join(template_dir_prefix, job_gen_type)
    return template_dir


def get_default_kwargs(args):
    kwargs = {
        "dry_run": args.dry_run,
        "force": args.force,
        "verbose": args.verbose,
        "queue_depth": 1,
        "unique_id": args.unique_id,
        "timelimit": "00:30:00",
    }
    return kwargs


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def estimate_num_minutes(num_jobs, num_leaves, unique_id):
    if unique_id == "sleep0":
        jobs_per_second = 1
    else:
        jobs_per_second = 0.1
    if unique_id == "stream":
        num_leaves = min(num_leaves, 32)  # stream saturates at 1 sched per node
    est_minutes = max(10, (num_jobs / 60 / jobs_per_second / num_leaves))
    est_minutes = min(est_minutes, 36 * 60)  # cap at 36 hours
    return est_minutes
