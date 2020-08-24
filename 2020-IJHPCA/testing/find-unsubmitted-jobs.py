#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

LOCK_NAME = "sbatch-lock"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testing_dirs", nargs="+")
    args = parser.parse_args()

    for test_dir in args.testing_dirs:
        path = Path(test_dir)
        if not (path / "test_config.json").exists():
            print("{} is missing a test_config.json".format(path), file=sys.stderr)
        elif not (path / LOCK_NAME).exists():
            print("{}".format(path))
