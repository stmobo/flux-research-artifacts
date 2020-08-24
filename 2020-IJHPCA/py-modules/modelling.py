import os
import logging
from abc import ABC, abstractmethod
from typing import SupportsFloat, SupportsInt

os.environ["NUMEXPR_MAX_THREADS"] = "16"
import numpy as np
import pandas as pd

from modules.interpolate import interpolate_real_data
from modules import util


def agg_by(df, agg_labels, func=np.median):
    return df.groupby(agg_labels).agg([func]).reset_index()


def trim_max_jobs_per_leaf(df, max_jobs_per_leaf):
    if max_jobs_per_leaf is not None:
        df = df[df.jobs_per_leaf.le(max_jobs_per_leaf)]
    return df


def _get_avg_runtime(unique_id, num_jobs=None, total_cores=None, system="opal"):
    if system == "opal":
        runtime_bounds = {
            "sleep0": (0, 1),
            "sleep5": (5, 5.23),
            "firestarter": (5, 5.69),
            "stream": (5.2, 22.15),
        }
    elif system == "lassen":
        runtime_bounds = {
            "sleep0": (0, 3.2),
            "sleep5": (5, 5.48),
            "firestarter": (5, 5.69),
            "stream": (5, 20.01),
        }
    fastest, slowest = runtime_bounds[unique_id]
    if num_jobs is None or total_cores is None:
        return fastest
    elif num_jobs >= total_cores:
        return slowest
    else:
        percentage_used = num_jobs / total_cores
        return ((slowest - fastest) * (percentage_used)) + fastest


def get_avg_runtime(unique_id, num_jobs=None, total_cores=None, system="opal"):
    if isinstance(num_jobs, np.ndarray):
        return np.array([_get_avg_runtime(unique_id, x, total_cores, system) for x in num_jobs])
    else:
        return _get_avg_runtime(unique_id, num_jobs, total_cores, system)


def calc_upperbound(unique_id, num_jobs, total_cores):
    total_core_seconds = num_jobs * get_avg_runtime(unique_id)
    if total_core_seconds == 0:
        return float("inf")
    return num_jobs / (total_core_seconds / min(num_jobs, total_cores))


def calc_cost_to_generate_dfs(dfs, df_labels, logger=None):
    if logger is None:
        logger = logging.getLogger()

    total_wall_seconds = 0
    total_node_seconds = 0
    for df, df_label in zip(dfs, df_labels):
        if isinstance(df.columns, pd.MultiIndex):
            subset = df
            makespan = subset.makespan["median"]
            num_nodes = subset.num_nodes["median"]
        else:
            subset = df[df.repetition == 0]
            makespan = subset.makespan
            num_nodes = subset.num_nodes
        df_wall_seconds = makespan.sum()
        df_node_seconds = (makespan * num_nodes).sum()
        total_wall_seconds += df_wall_seconds
        total_node_seconds += df_node_seconds
        logger.debug(
            "Processing DF with the name %s, that took %f wall seconds and %f node seconds to build (single repetition)",
            df_label,
            df_wall_seconds,
            df_node_seconds,
        )
        util.pretty_log_df(subset, "DF subset used for cost calculation", logger)
    return total_wall_seconds, total_node_seconds


def get_num_leaves(topology):
    return util.prod(topology)


class Model(ABC):
    def __init__(self, logger=None):
        self.level_one_interp = None
        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger

    @abstractmethod
    def get_valid_jobs_per_leaf(self):
        pass

    @abstractmethod
    def get_single_level_predictions(self, num_jobs):
        pass

    @abstractmethod
    def get_job_launch_cost(self, topology, jobs_per_leaf):
        pass

    @abstractmethod
    def get_empty_hierarchy_init_cost(self, topology):
        pass

    def get_interpolated_predictions(self, topology):
        num_levels = len(topology)
        if num_levels not in [1, 2, 3]:
            raise NotImplementedError("{} levels not implemented".format(num_levels))
        elif num_levels == 1 and self.level_one_interp is not None:
            return self.level_one_interp
        prediction_df = self.get_predictions(topology)
        self.logger.debug(
            "Num-Levels: %d, Prediction DF:\n%s", num_levels, prediction_df
        )
        out_df = self.interpolate(
            prediction_df.num_jobs.to_numpy(), prediction_df.makespan.to_numpy()
        )
        out_df["num_levels"] = len(topology)
        if num_levels == 1:
            self.level_one_interp = out_df

        return out_df

    def get_predictions(self, topology):
        num_levels = len(topology)

        jobs_per_leaf = self.get_valid_jobs_per_leaf()
        jobs_per_leaf = jobs_per_leaf.unique()
        assert len(jobs_per_leaf) > 0

        if num_levels == 1:
            num_jobs = jobs_per_leaf
            makespans = self.get_single_level_predictions(jobs_per_leaf)
            self.logger.debug(
                "Num leaves: %d, Num_jobs: %s, Makespans: %s", 1, num_jobs, makespans
            )
        else:
            num_leaves = get_num_leaves(topology)
            num_jobs = np.array([jpl * num_leaves for jpl in jobs_per_leaf])
            self.logger.debug(
                "Num leaves: %d, Num_jobs: %s, Jobsperleaf: %s",
                num_leaves,
                num_jobs,
                jobs_per_leaf,
            )
            makespans = self.get_multilevel_predictions(topology, jobs_per_leaf)
        out_df = pd.DataFrame(data={"num_jobs": num_jobs, "makespan": makespans})
        out_df["num_levels"] = len(topology)
        out_df.sort_values(by="num_jobs")
        return out_df

    def get_multilevel_predictions(self, topology, jobs_per_leaf: np.ndarray):
        if len(topology) < 2:
            raise ValueError("Must be passed a multi-level topology")
        return [self.calc_cost(topology, jpl) for jpl in jobs_per_leaf]

    @staticmethod
    def interpolate(num_jobs, makespans):
        interp_num_jobs, interp_makespans = interpolate_real_data(
            num_jobs, makespans, degree=2
        )
        return pd.DataFrame(
            data={
                "num_jobs": interp_num_jobs,
                "makespan": interp_makespans,
                "throughput": interp_num_jobs / interp_makespans,
            }
        )

    def calc_cost(self, topology, jobs_per_leaf: np.int32):
        cost_to_launch_hierarchy = self.get_empty_hierarchy_init_cost(topology)
        # Only have to consider the cost at the last leaf (it is the straggler/on the critical path)
        cost_to_launch_jobs = self.get_job_launch_cost(topology, jobs_per_leaf)
        total_cost = cost_to_launch_hierarchy + cost_to_launch_jobs
        self.logger.log(
            9,
            "Level-{}, jobsperleaf: {:4d}, hierarchy_launch: {:4.1f}, job_launch: {:7.1f}, total: {:7.1f}".format(
                len(topology),
                jobs_per_leaf,
                cost_to_launch_hierarchy,
                cost_to_launch_jobs,
                total_cost,
            ),
        )
        return total_cost

    @abstractmethod
    def calc_model_cost(self):
        pass


class AnalyticalModel(Model):
    def __init__(
        self,
        sched_rate: SupportsFloat,
        sched_create_cost: SupportsFloat,
        resource_cap: SupportsFloat,
        max_jobs_per_leaf: SupportsInt,
        logger=None,
    ):
        """
        sched_rate is the maximum throughput, in jobs per second, that a given
        scheduler implementation can achieve on a resource unconstrained system.

        resource_cap is the maximum number of jobs that can be running
        simultaneously due to resource capacity constraints
        """
        super().__init__(logger=logger)
        self.sched_rate = float(sched_rate)
        self.sched_create_cost = float(sched_create_cost)
        self.resource_cap = float(resource_cap)
        self.max_jobs_per_leaf = int(max_jobs_per_leaf)

    @abstractmethod
    def get_single_level_predictions(self, num_jobs: np.ndarray, resource_cap=None, runtime=None):
        pass

    @abstractmethod
    def calc_model_cost(self):
        pass

    def get_valid_jobs_per_leaf(self):
        return pd.Series(data=np.arange(1, self.max_jobs_per_leaf + 1, dtype=np.int32))

    def get_interpolated_predictions(self, topology):
        # No need for interpolation with this inexpensive analytical model
        pred_df = self.get_predictions(topology)
        pred_df["throughput"] = pred_df.num_jobs / pred_df.makespan
        return pred_df

    def get_job_launch_cost(self, topology, jobs_per_leaf: pd.Series):
        num_leaves = get_num_leaves(topology)
        return self.get_single_level_predictions(
            np.array([jobs_per_leaf]), resource_cap=self.resource_cap / num_leaves
        )[0]

    def get_empty_hierarchy_init_cost(self, topology):
        if len(topology) == 0:
            return 0
        return self.get_single_level_predictions(np.array([topology[-1]]), runtime=0)[
            0
        ] + self.get_empty_hierarchy_init_cost(topology[:-1])


class PurelyAnalyticalModel(AnalyticalModel):
    def __init__(
        self,
        sched_rate: SupportsFloat,
        sched_create_cost: SupportsFloat,
        job_runtime: SupportsFloat,
        resource_cap: SupportsFloat,
        max_jobs_per_leaf=1024,
        logger=None,
    ):
        """
        sched_rate is the maximum throughput, in jobs per second, that a given
        scheduler implementation can achieve on a resource unconstrained system.

        job_runtime is the average runtime of each job

        resource_cap is the maximum number of jobs that can be running
        simultaneously due to resource capacity constraints
        """
        super().__init__(sched_rate, sched_create_cost, resource_cap, max_jobs_per_leaf, logger=logger)
        self.job_runtime = float(job_runtime)

    def get_single_level_predictions(self, num_jobs: np.ndarray, resource_cap=None, runtime=None):
        if resource_cap is None:
            resource_cap = self.resource_cap  # assume we are using the whole system
        if runtime is None:
            runtime = self.job_runtime
        sched_bottleneck = num_jobs / self.sched_rate
        resource_bottleneck = (num_jobs * runtime) / num_jobs.clip(
            max=resource_cap
        )  # clip(max=X) sets all values greater than X to X.  equivalent to .apply(lambda x: min(x, X))
        true_bottleneck = np.maximum(sched_bottleneck, resource_bottleneck)
        init_cost = self.sched_create_cost
        return true_bottleneck + init_cost

    def calc_model_cost(self):
        return (0, 0)


class AnalyticalModelContentedRuntime(AnalyticalModel):
    def __init__(
        self,
        sched_rate: SupportsFloat,
        sched_create_cost: SupportsFloat,
        resource_cap: SupportsFloat,
        avg_runtime_func,
        cores_per_node,
        max_jobs_per_leaf=1024,
        logger=None,
    ):
        """
        sched_rate is the maximum throughput, in jobs per second, that a given
        scheduler implementation can achieve on a resource unconstrained system.

        resource_cap is the maximum number of jobs that can be running
        simultaneously due to resource capacity constraints
        """
        super().__init__(sched_rate, sched_create_cost, resource_cap, max_jobs_per_leaf, logger=logger)

        self.get_avg_runtime = avg_runtime_func
        self.cores_per_node = cores_per_node

    def get_single_level_predictions(self, num_jobs: np.ndarray, resource_cap=None, runtime=None):
        if resource_cap is None:
            resource_cap = self.resource_cap  # assume we are using the whole system
        if runtime is None:
            runtime = self.get_avg_runtime(num_jobs, resource_cap)
        sched_bottleneck = num_jobs / self.sched_rate
        resource_bottleneck = (
            num_jobs * runtime
        ) / num_jobs.clip(
            max=resource_cap
        )  # clip(max=X) sets all values greater than X to X.  equivalent to .apply(lambda x: min(x, X))
        true_bottleneck = np.maximum(sched_bottleneck, resource_bottleneck)
        init_cost = self.sched_create_cost
        return true_bottleneck + init_cost

    def calc_model_cost(self):
        num_waves = 3
        contention_test = (
            self.get_avg_runtime(self.cores_per_node * num_waves, self.cores_per_node)
            * num_waves
        )
        return (contention_test, contention_test)


class SimpleModel(Model):
    def __init__(self, model_df, setup_df, sleep0_df, logger=None):
        super().__init__(logger=logger)

        self.depth_1_model_df = agg_by(
            model_df[(model_df.num_levels == 1) & (model_df.num_nodes == 1)],
            ["num_jobs"],
        )
        util.pretty_log_df(self.depth_1_model_df, "depth 1 model DF:", self.logger)

        self.empty_hierarchy_df = agg_by(
            setup_df[(setup_df["just_setup"] == True) & (setup_df["num_levels"] == 1)],
            ["num_levels"],
        )
        util.pretty_log_df(self.empty_hierarchy_df, "Empty hierarchy DF:", self.logger)
        assert len(self.empty_hierarchy_df) == 1  # should only contain first level

        # Just needed to know the raw sched overhead of launching num_jobs ==
        # topology branching factors, (to estimate hierarchy setup cost)
        self.sleep0_df = agg_by(
            sleep0_df[(sleep0_df.num_levels == 1) & (sleep0_df.num_nodes == 1)],
            ["num_jobs"],
        )
        util.pretty_log_df(self.sleep0_df, "Sleep0 Job Launch Model df", self.logger)

    def calc_model_cost(self):
        return calc_cost_to_generate_dfs(
            [self.depth_1_model_df, self.empty_hierarchy_df, self.sleep0_df],
            ["Depth-1", "Empty Hierarchy", "Sleep 0"],
        )

    def get_valid_jobs_per_leaf(self):
        return self.depth_1_model_df["num_jobs"]

    def get_single_level_predictions(self, num_jobs: pd.Series):
        model_level_df = self.depth_1_model_df[
            self.depth_1_model_df.num_jobs.isin(num_jobs)
        ]
        return model_level_df["makespan"]["median"]

    def get_job_launch_cost(self, topology, jobs_per_leaf):
        num_jobs = jobs_per_leaf
        series = self.get_interpolated_predictions([1])
        series = series[series.num_jobs == num_jobs]
        try:
            assert len(series) == 1
        except:
            self.logger.debug("Num jobs: %s, SERIES:\n%s", num_jobs, series)
            raise
        makespan = series["makespan"].iloc[0]
        return makespan

    def get_sleep0_launch_cost(self, topology, num_jobs):
        if num_jobs not in topology:
            raise RuntimeError(
                "Accessing data when we shouldn't: topo ({}), num_jobs ({})".format(
                    topology, num_jobs
                )
            )
        makespan = self.sleep0_df[self.sleep0_df.num_jobs == num_jobs]["makespan"][
            "median"
        ]
        assert len(makespan) == 1
        return makespan.iloc[0]

    def get_empty_hierarchy_init_cost(self, topology):
        num_levels = len(topology)
        if num_levels == 1:
            assert topology[0] == 1
            return self.empty_hierarchy_df[self.empty_hierarchy_df.num_levels == 1][
                "makespan"
            ]["median"].iloc[0]
        else:
            # calculate the time to launch the critical path of the hiearchy
            # (i.e., the last instance at each level)
            return self.get_sleep0_launch_cost(
                topology, topology[-1]
            ) + self.get_empty_hierarchy_init_cost(topology[:-1])


class EmpiricalModel(Model):
    def __init__(self, model_df, setup_df, depth_1_real_df, logger=None):
        super().__init__(logger=logger)

        self.depth_1_real_df = depth_1_real_df

        self.depth_two_leaf_perf_df = model_df[
            (model_df.num_levels == 1) & (model_df.num_nodes == 1)
        ]
        self.depth_two_leaf_perf_df = agg_by(self.depth_two_leaf_perf_df, ["num_jobs"])
        util.pretty_log_df(
            self.depth_two_leaf_perf_df, "Depth-2 leaf performance", self.logger
        )

        self.depth_three_leaf_perf_df = model_df[
            (model_df.num_levels == 2)
            & (model_df.num_nodes == 1)
            & (model_df.first_branch_factor == 36)
        ]
        self.depth_three_leaf_perf_df = agg_by(
            self.depth_three_leaf_perf_df, ["num_jobs"]
        )
        util.pretty_log_df(
            self.depth_three_leaf_perf_df, "Depth-3 leaf performance", self.logger
        )

        self.empty_hierarchy_df = agg_by(
            setup_df[(setup_df["just_setup"] == True)],
            ["num_levels", "first_branch_factor"],
        )
        util.pretty_log_df(self.empty_hierarchy_df, "Empty hierarchy DF:", self.logger)
        assert len(self.empty_hierarchy_df) == 3  # should contain all three levels

    def calc_model_cost(self):
        return calc_cost_to_generate_dfs(
            [
                self.depth_1_real_df,
                self.empty_hierarchy_df,
                self.depth_two_leaf_perf_df,
                self.depth_three_leaf_perf_df,
            ],
            ["Depth-1 Real", "Empty Hierarchy", "Depth-2 Leaf", "Depth-3 Leaf"],
        )

    def get_valid_jobs_per_leaf(self):
        return self.depth_1_real_df["num_jobs"]

    def get_single_level_predictions(self, num_jobs: pd.Series):
        return self.depth_1_real_df[self.depth_1_real_df.num_jobs.isin(num_jobs)][
            "makespan"
        ]["median"]

    def get_job_launch_cost(self, topology, jobs_per_leaf):
        num_levels = len(topology)
        if num_levels == 2:
            perf_df = self.depth_two_leaf_perf_df
        elif num_levels == 3:
            perf_df = self.depth_three_leaf_perf_df
        series = perf_df[perf_df.jobs_per_leaf["median"] == jobs_per_leaf]
        try:
            assert len(series) == 1
        except:
            self.logger.debug(
                "num_levels: {}, jobs_per_leaf: {}".format(num_levels, jobs_per_leaf)
            )
            self.logger.debug("SERIES:\n%s", series)
            raise
        makespan = series["makespan"]["median"].iloc[0]
        return makespan

    def get_empty_hierarchy_init_cost(self, topology):
        num_levels = len(topology)
        conditional = self.empty_hierarchy_df.num_levels == num_levels
        series = self.empty_hierarchy_df[conditional]
        try:
            assert len(series) == 1
        except:
            self.logger.debug("num_levels: {}".format(num_levels))
            self.logger.debug("SERIES:\n%s", series)
            raise
        makespan = series["makespan"]["median"].iloc[0]
        return makespan
