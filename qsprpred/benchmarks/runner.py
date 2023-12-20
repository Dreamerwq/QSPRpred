import itertools
import os
import random
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock
from typing import Generator

import pandas as pd

from .replica import Replica
from .settings.benchmark import BenchmarkSettings
from ..logs import logger

lock = Lock()


class BenchmarkRunner:
    """Class that runs benchmarking experiments as defined by
    `BenchmarkSettings`. It translates the settings into
    a list of `Replica` objects with its `iterReplicas` method and
    runs them in parallel. Each replica is processed by the `runReplica`
    method.

    The report from each replica is appended to a `resultsFile`, which
    is read and returned by the `run` method after the runners is finished
    with all replicas. All outputs generated by the replicas and the `BenchmarkSettings`
    used are saved in the `dataDir`.

    The random seed for each replica is determined in a pseudo-random way from
     `BenchmarkSettings.random_seed`. The `getSeedList` method is used to generate
     a list of seeds from this 'master' seed. There are some caveats to this method
     (see the docstring of `getSeedList`).

    Attributes:
        settings (BenchmarkSettings):
            Benchmark settings.
        nProc (int):
            Number of processes to use.
        resultsFile (str):
            Path to the results file.
        dataDir (str):
            Path to the directory to store data.

    """

    class ReplicaException(Exception):
        """Custom exception for errors in a replica.

        Attributes:
            replicaID (int):
                ID of the replica that caused the error.
            exception (Exception):
                Exception that was raised.
        """

        def __init__(self, replica_id: str, exception: Exception):
            """Initialize the exception.

            Args:
                replica_id (str):
                    ID of the replica that caused the error.
                exception (Exception):
                    Exception that was raised.
            """
            self.replicaID = replica_id
            self.exception = exception

    def __init__(
        self,
        settings: BenchmarkSettings,
        n_proc: int | None = None,
        data_dir: str = "./data",
        results_file: str | None = None,
    ):
        """Initialize the runner.

        Args:
            settings (BenchmarkSettings):
                Benchmark settings.
            n_proc (int, optional):
                Number of processes to use. Defaults to os.cpu_count().
            data_dir (str, optional):
                Path to the directory to store data. Defaults to "./data".
                If the directory does not exist, it will be created.
            results_file (str, optional):
                Path to the results file. Defaults to "{data_dir}/data/results.tsv".
        """
        logger.debug("Initializing BenchmarkRunner...")
        self.settings = settings
        self.nProc = n_proc or os.cpu_count()
        self.dataDir = data_dir
        self.resultsFile = results_file if results_file else f"{data_dir}/results.tsv"
        os.makedirs(self.dataDir, exist_ok=True)
        logger.debug(f"Saving settings to: {self.dataDir}/settings.json")
        self.settings.toFile(f"{self.dataDir}/settings.json")

    @property
    def nRuns(self) -> int:
        """Returns the total number of benchmarking runs. This is the product
        of the number of replicas, data sources, descriptors, target properties,
        data preparation settings and models as defined in the `BenchmarkSettings`.

        Returns:
            int:
                Total number of benchmarking runs.
        """
        benchmark_settings = self.settings
        benchmark_settings.checkConsistency()
        ret = (
            benchmark_settings.n_replicas
            * len(benchmark_settings.data_sources)
            * len(benchmark_settings.descriptors)
            * len(benchmark_settings.target_props)
            * len(benchmark_settings.prep_settings)
            * len(benchmark_settings.models)
        )
        if len(benchmark_settings.optimizers) > 0:
            ret *= len(benchmark_settings.optimizers)
        return ret

    def run(self, raise_errors=False) -> pd.DataFrame:
        """Runs the benchmarking experiments.

        Args:
            raise_errors (bool, optional):
                Whether to raise the first encountered `ReplicaException`
                and stop the benchmarking run. Defaults to `False`,
                in which case replicas that raise an exception are skipped
                and errors are logged.

        Returns:
            pd.DataFrame:
                Results from the benchmarking experiments.
        """
        logger.debug(f"Performing {self.nRuns} replica runs...")
        with ProcessPoolExecutor(max_workers=self.nProc) as executor:
            for result in executor.map(
                self.runReplica, self.iterReplicas(), itertools.repeat(self.resultsFile)
            ):
                if isinstance(result, self.ReplicaException):
                    if raise_errors:
                        raise result.exception
                    else:
                        logger.error(
                            f"Error in replica {result.replicaID}: {result.exception}"
                        )
        logger.debug("Finished all replica runs.")
        return pd.read_table(self.resultsFile)

    def getSeedList(self, seed: int | None = None) -> list[int]:
        """
        Get a list of seeds for the replicas from one 'master' randomSeed.
        The list of seeds is generated by sampling from the range of
        possible seeds (0 to 2^32 - 1) with the given randomSeed as the random
        randomSeed for the random module. This means that the list of seeds
        will be the same for each run of the benchmarking experiment
        with the same 'master' randomSeed. This is useful for reproducibility,
        but it also avoids recalculating replicas that were already calculated.

        Caveat: If the randomSeed in `BenchmarkSettings.randomSeed` is the same, but
        the number of replicas is different (i.e. the settings themselves change)
        then this code will still generate the same seeds for experiments that
        might not overlap with previous experiments. Therefore, take this into account
        when you already calculated some replicas, but decided to change your experiment
        settings.

        Args:
            seed (int, optional):
                'Master' randomSeed. Defaults to `BenchmarkSettings.randomSeed`.

        Returns:
            list[int]:
                list of seeds for the replicas
        """
        seed = seed or self.settings.random_seed
        random.seed(seed)
        return random.sample(range(2**32 - 1), self.nRuns)

    def iterReplicas(self) -> Generator[Replica, None, None]:
        """Generator that yields `Replica` objects for each benchmarking run.
        This is done by iterating over the product of the data sources, descriptors,
        target properties, data preparation settings, models and optimizers as defined
        in the `BenchmarkSettings`. The random randomSeed for each replica is determined
        in a pseudo-random way from `BenchmarkSettings.randomSeed` via
        the `getSeedList` method.

        Yields:
            Generator[Replica, None, None]:
                `Replica` objects for each benchmarking run.
        """
        benchmark_settings = self.settings
        benchmark_settings.checkConsistency()
        indices = [x + 1 for x in range(benchmark_settings.n_replicas)]
        optimizers = (
            benchmark_settings.optimizers
            if len(benchmark_settings.optimizers) > 0
            else [None]
        )
        product = itertools.product(
            indices,
            [benchmark_settings.name],
            benchmark_settings.data_sources,
            benchmark_settings.descriptors,
            benchmark_settings.target_props,
            benchmark_settings.prep_settings,
            benchmark_settings.models,
            optimizers,
        )
        seeds = self.getSeedList(benchmark_settings.random_seed)
        for idx, settings in enumerate(product):
            yield Replica(
                *settings,
                random_seed=seeds[idx],
                assessors=benchmark_settings.assessors,
            )

    @classmethod
    def runReplica(cls, replica: Replica, results_file: str):
        """Runs a single replica. This is executed in parallel by the `run` method.
        It is a classmethod so that it can be pickled and executed in parallel
        more easily.

        Args:
            replica (Replica):
                Replica to run.
            results_file (str):
                Path to the results file.

        Returns:
            pd.DataFrame | ReplicaException:
                Results from the replica or a `ReplicaException` if an error occurred.
        """
        try:
            with lock:
                df_results = None
                if os.path.exists(results_file):
                    df_results = pd.read_table(results_file)
                if (
                    df_results is not None
                    and df_results.ReplicaID.isin([replica.id]).any()
                ):
                    logger.warning(f"Skipping {replica.id}")
                    return
                replica.initData()
                replica.addDescriptors()
            replica.prepData()
            replica.initModel()
            replica.runAssessment()
            with lock:
                df_report = replica.createReport()
                df_report.to_csv(
                    results_file,
                    sep="\t",
                    index=False,
                    mode="a",
                    header=not os.path.exists(results_file),
                )
                return df_report
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            return cls.ReplicaException(replica.id, e)
