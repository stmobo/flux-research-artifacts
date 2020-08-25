#!/usr/bin/env python3

import os

from make_tests import make_test
import common


def build_model(template_dir, output_dir, job_gen_type, repetition, **kwargs):
    num_nodes = 1
    num_cores = num_cores_per_node * num_nodes
    topologies = [[1], [1, num_cores_per_node]]
    for topology in topologies:
        jobs = [1, 4, 16, 64, 256, 1024, 4096]
        if len(topology) > 1:
            # prevent OOM from too many jobs per node
            jobs = jobs[:-1]

        if args.small_scale:
            jobs = jobs[:3]
        elif args.medium_scale:
            jobs = jobs[:6]

        for idx, jobs_per_leaf in enumerate(jobs):
            num_leaves = common.prod(topology)
            num_jobs = jobs_per_leaf * num_leaves
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


def just_hierarchy_setup(
    template_dir, output_dir, job_gen_type, topologies, num_nodes, repetition, **kwargs
):
    fake_jobs_kwargs = kwargs.copy()
    fake_jobs_kwargs["command"] = "sleep 0"
    fake_jobs_kwargs["unique_id"] = "setup"
    fake_jobs_kwargs["timelimit"] = "00:05:00"
    fake_jobs_kwargs["just_setup"] = True

    for topology in topologies:
        for nnodes in [1, num_nodes]:
            num_jobs = common.prod(topology)
            make_test(
                topology,
                num_jobs,
                nnodes,
                num_cores_per_node,
                repetition,
                template_dir,
                output_dir,
                job_gen_type,
                **fake_jobs_kwargs,
            )

    fake_jobs_kwargs = kwargs.copy()
    fake_jobs_kwargs["command"] = "flux mini submit -n1 -c1 sleep 0"
    fake_jobs_kwargs["unique_id"] = "sleep0"
    fake_jobs_kwargs["timelimit"] = "00:15:00"

    branch_factors = {bf for topo in topologies for bf in topo}
    for bf in branch_factors:
        for nnodes in [1, num_nodes]:
            num_jobs = bf
            make_test(
                [1],
                num_jobs,
                nnodes,
                num_cores_per_node,
                repetition,
                template_dir,
                output_dir,
                job_gen_type,
                **fake_jobs_kwargs,
            )


def main():
    kwargs = common.get_default_kwargs(args)
    job_gen_type = "direct"
    template_dir = common.get_template_dir(job_gen_type)
    kwargs["command"] = common.get_command(args.unique_id)
    output_dir = common.get_output_dir("model", args)
    assert os.path.isdir(template_dir), template_dir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    num_nodes = 32
    topologies = [[1], [1, num_nodes], [1, num_nodes, num_cores_per_node]]

    repetitions = range(1) if args.small_scale or args.medium_scale else range(3)
    if args.repetitions:
        repetitions = range(args.repetitions)
    for repetition in repetitions:
        build_model(template_dir, output_dir, job_gen_type, repetition, **kwargs)
        just_hierarchy_setup(
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
