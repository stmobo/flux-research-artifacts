#!/usr/bin/env python3

import os

from make_tests import make_test
import common


def variable_num_jobs(
    template_dir, output_dir, job_gen_type, topologies, num_nodes, repetition, **kwargs
):
    num_cores = num_cores_per_node * num_nodes

    for topology in topologies:
        num_leaves = common.prod(topology)
        jobs_per_leaf = [1, 4, 16, 64, 256, 1024, 4096]
        if len(topology) > 1:
            # prevent OOM from too many jobs per node
            jobs_per_leaf = jobs_per_leaf[:-1]

        for idx, num_jobs in enumerate([x * num_leaves for x in jobs_per_leaf]):
            est_minutes = common.estimate_num_minutes(
                num_jobs, num_leaves, kwargs["unique_id"]
            )
            hours = int(est_minutes // 60)
            minutes = int(est_minutes % 60)
            kwargs["timelimit"] = "{:02d}:{:02d}:00".format(hours, minutes)
            make_test(
                topology,
                num_jobs,
                num_nodes,
                num_cores_per_node,
                repetition,
                template_dir,
                output_dir,
                job_gen_type,
                **kwargs,
            )


def main():
    kwargs = common.get_default_kwargs(args)
    job_gen_type = "direct"
    template_dir = common.get_template_dir(job_gen_type)
    kwargs["command"] = common.get_command(args.unique_id)
    output_dir = common.get_output_dir("multi-level", args)
    assert os.path.isdir(template_dir), template_dir
    assert os.path.isdir(output_dir), output_dir

    if args.small_scale:
        num_nodes = 2
        topologies = [[1], [4], [2, 4]]
    elif args.medium_scale:
        num_nodes = 8
        topologies = [[1], [4], [2, 4]]
    else:
        num_nodes = 32
        topologies = [[1], [1, num_nodes], [1, num_nodes, num_cores_per_node]]

    repetitions = range(1) if args.small_scale or args.medium_scale else range(3)
    if args.repetitions:
        repetitions = range(args.repetitions)
    for repetition in repetitions:
        variable_num_jobs(
            template_dir,
            output_dir,
            job_gen_type,
            topologies,
            num_nodes,
            repetition,
            **kwargs,
        )


if __name__ == "__main__":
    parser = common.get_parser()
    args = parser.parse_args()

    num_cores_per_node = 36

    main()
