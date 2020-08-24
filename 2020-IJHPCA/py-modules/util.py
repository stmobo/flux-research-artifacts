import operator
import logging
from functools import reduce

import pandas as pd


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def unique_id(df):
    unique_ids = set(df["unique_id"].unique())
    if len(unique_ids) == 0:
        raise ValueError("unique_id key has no entries")
    try:
        unique_ids.remove("setup")
    except KeyError:
        pass
    if len(unique_ids) == 1:
        return unique_ids.pop()
    else:
        unique_ids.remove("sleep0")
    assert len(unique_ids) == 1
    return unique_ids.pop()


columns_to_print = [
    "unique_id",
    "topology",
    "num_levels",
    "num_jobs",
    "jobs_per_leaf",
    "num_nodes",
    "repetition",
    "makespan",
    "throughput",
]


def pretty_log_df(df, label="DF", logger=None):
    if logger is None:
        logger = logging.getLogger()
    if isinstance(df.columns, pd.MultiIndex):
        df_columns = [
            title for column_tuple in df.columns.to_numpy() for title in column_tuple
        ]
        print_columns = [column for column in columns_to_print if column in df_columns]
    else:
        print_columns = [x for x in columns_to_print if x in df.columns]
    with pd.option_context(
        "display.max_rows", 100, "display.max_columns", 160, "display.width", 160
    ):
        logger.debug("%s:\n%s", label, df[print_columns])
