#!/usr/bin/env python3

import os
import argparse
import logging

os.environ["NUMEXPR_MAX_THREADS"] = "16"
import pandas as pd


def main():
    df = pd.read_pickle(args.df)

    logger.info("\n%s", df)

    if args.csv is not None:
        logger.info("Saving data to {}".format(args.csv))
        df.to_csv(args.csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("df")
    parser.add_argument("--csv")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("dump_df")

    main()
