#!/usr/bin/env python3

import re
import json
import argparse
import itertools
from pathlib import Path
import subprocess
from subprocess import CalledProcessError
from datetime import timedelta
import os.path as osp

from modules.time_conversion import walltime_str_to_timedelta

LOCK_NAME = "sbatch-lock"
RUN_SCRIPT_NAME = "run.sh"


class Test(object):
    def __init__(self, testing_dir):
        self.path = Path(testing_dir).resolve(strict=True)
        assert self.path.is_dir()

        with (self.path / "test_config.json").open(mode="r") as conffile:
            self.test_config = json.load(conffile)

    @property
    def timelimit_str(self):
        return self.test_config["timelimit"]

    @property
    def timelimit(self):
        return walltime_str_to_timedelta(self.timelimit_str)

    @property
    def repetition(self):
        return self.test_config["repetition"]

    @property
    def locked(self):
        return (self.path / LOCK_NAME).is_dir()

    def lock(self):
        (self.path / LOCK_NAME).mkdir()

    @property
    def unlocked(self):
        return not self.locked


cmd_out_re = re.compile(r"Job \<([0-9]+)\> is submitted")


def run_bsub(command, cwd, stdin):
    """
    If check is true, and the process exits with a non-zero exit code,
    a CalledProcessError exception will be raised. Attributes of that
    exception hold the arguments, the exit code, and stdout and stderr
    if they were captured.
    """
    completed_proc = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        encoding="ascii",
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    match = cmd_out_re.match(completed_proc.stdout)
    if match is None:
        raise RuntimeError("Unexpected bsub stdout: {}".format(completed_proc.stdout))
    return int(match.group(1))


def submit_test(test, dependencies=None):
    if dependencies is None:
        dependencies = []
    dependencies = [x for x in dependencies if x >= 0]

    print("{}".format(test.path))
    with open(osp.join(test.path, RUN_SCRIPT_NAME), "rb") as f:
        run_script_bytes = f.read()

    command = ["bsub"]

    if args.partition is not None:
        command.append("-q")
        command.append(args.partition)

    if len(dependencies) == 0:
        dependency_str = "N/A"
    else:
        dependency_str = "&&".join(["ended("+str(x)+")" for x in dependencies])
        command.append("-w")
        command.append(dependency_str)

    if args.debug_job:
        command.append("-env")
        command.append("LOG_LEVEL=7")

    if args.dry_run:
        print(
            "\tWould submit by running '{}' (with timelimit {})".format(
                " ".join(command), test.timelimit_str
            )
        )
        curr_jobid = (
            int(dependencies[0]) + 1
            if len(dependencies) > 0 and isinstance(dependencies[0], int)
            else 1
        )
    else:
        test.lock()
        try:
            curr_jobid = run_bsub(command, test.path, run_script_bytes)
        except CalledProcessError as e:
            print("\tSkipping, encountered an exception while submitting: {}".format(e))
            return -1
        print(
            "\tJust submitted test ({}) with dependencies == {}".format(
                curr_jobid, dependency_str
            )
        )

    return curr_jobid


def main():
    tests = [Test(path) for path in args.testing_dirs]
    locked_tests = [test for test in tests if test.locked]
    # unlocked_tests = sorted(tests - locked_tests, key=lambda x: (x.timelimit, x.repetition))
    unlocked_tests = [test for test in tests if test.unlocked]
    assert len(locked_tests) + len(unlocked_tests) == len(tests)
    # unlocked_tests = sorted(unlocked_tests, key=lambda x: (x.repetition, x.timelimit))
    unlocked_tests = sorted(unlocked_tests, key=lambda x: (x.timelimit, x.repetition))
    if args.verbose:
        for locked_test in locked_tests:
            print("Skipping {} because it is locked".format(locked_test.path))

    if args.no_chain:
        for test in unlocked_tests:
            submit_test(test)
    elif args.max_parallelism > 1:
        last_jobs = []
        chains = [list() for x in range(args.max_parallelism)]
        for test, chain in zip(unlocked_tests, itertools.cycle(chains)):
            chain.append(test)
        for idx, chain in enumerate(chains):
            chain_time = timedelta(seconds=0)
            if args.dry_run:
                print(
                    '===================Job "lane" #{}==================='.format(
                        idx + 1
                    )
                )
            dependencies = DEPENDENCIES
            for test in chain:
                chain_time += test.timelimit
                prev_jobid = submit_test(test, dependencies=dependencies)
                dependencies = [prev_jobid]
            last_jobs.append(str(prev_jobid))
            if args.dry_run:
                print(
                    "===================Lane will maximally take {} hours (~{} days)===================".format(
                        chain_time.total_seconds() / 60 / 60, chain_time.days
                    )
                )
        print("To run after all lanes, depend on: {}".format(",".join(last_jobs)))
    else:
        dependencies = DEPENDENCIES
        for test in unlocked_tests:
            prev_jobid = submit_test(test, dependencies=dependencies)
            dependencies = [prev_jobid]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testing_dirs", nargs="+")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--dry-run", action="store_true")
    parser.add_argument(
        "--debug-job", action="store_true", help="run job in debug mode"
    )
    parser.add_argument("--no-chain", action="store_true")
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=1,
        help="when chaining, allow N jobs at a time",
    )
    parser.add_argument(
        "--dependencies",
        default="",
        help="comma-separated list of jobids that these jobs should only run after",
    )
    parser.add_argument(
        "-p", "--partition"
    )
    args = parser.parse_args()

    if (args.max_parallelism > 1) + (args.no_chain) > 1:
        print("Incompatible arguments: max-parallelism, no-chain")
        exit(1)

    DEPENDENCIES = [int(x) for x in args.dependencies.split(",") if x.isdigit()]

    main()
