import itertools
import logging
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

lock_data = Lock()
lock_report = Lock()


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

    logLevel = logging.DEBUG

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
                else:
                    logger.debug(f"Return success from replica: {result}")
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
            yield self.makeReplica(
                *settings,
                random_seed=seeds[idx],
                assessors=benchmark_settings.assessors,
            )

    def makeReplica(self, *args, **kwargs) -> Replica:
        """Returns a `Replica` object for the given settings. This is useful
        for debugging.

        Returns:
            Replica:
                Replica object.
        """
        return Replica(*args, **kwargs)

    @classmethod
    def getLoggerForReplica(cls, replica: Replica, level: int = logging.DEBUG):
        """Returns a logger for the given replica.

        Args:
            replica (Replica):
                Replica to get the logger for.
            level (int, optional):
                Log level. Defaults to logging.DEBUG.
        """
        replica_logger = logging.getLogger(replica.id)
        if len(replica_logger.handlers) > 0:
            return replica_logger
        replica_logger.setLevel(level)
        sh = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        sh.setFormatter(formatter)
        sh.setLevel(level)
        replica_logger.addHandler(sh)
        return replica_logger

    @classmethod
    def checkReplicaInResultsFile(cls, replica: Replica, results_file: str) -> bool:
        """Checks if the replica is already present in the results file.
        This method is thread-safe.

        Args:
            replica (Replica):
                Replica to check.
            results_file (str):
                Path to the results file.

        Returns:
            bool:
                Whether the replica is already present in the results file.
        """
        if not os.path.exists(results_file):
            return False
        df_results = pd.read_table(results_file)
        return df_results.ReplicaID.isin([replica.id]).any()

    @classmethod
    def replicaToReport(cls, replica: Replica) -> pd.DataFrame:
        """Converts a replica to a report.

        Args:
            replica (Replica):
                Replica to convert.

        Returns:
            pd.DataFrame:
                Report from the replica.
        """
        return replica.createReport()

    @classmethod
    def appendReportToResults(cls, df_report: pd.DataFrame, results_file: str):
        """Appends a report to the results file. This method is thread-safe.

        Args:
            df_report (pd.DataFrame):
                Report to append.
            results_file (str):
                Path to the results file.
        """
        df_report.to_csv(
            results_file,
            sep="\t",
            index=False,
            mode="a",
            header=not os.path.exists(results_file),
        )

    @classmethod
    def initData(cls, replica: Replica):
        """Initializes the data set for this replica.
        This method is thread-safe.

        Args:
            replica (Replica):
                Replica to initialize.
        """
        logger = cls.getLoggerForReplica(replica, cls.logLevel)
        logger.debug("Initializing data set...")
        replica.initData()
        logger.debug("Done.")
        logger.debug("Adding descriptors...")
        replica.addDescriptors()
        logger.debug("Done.")

    @classmethod
    def runReplica(cls, replica: Replica, results_file: str) -> str | ReplicaException:
        """Runs a single replica. This is executed in parallel by the `run` method.
        It is a classmethod so that it can be pickled and executed in parallel
        more easily.

        Args:
            replica (Replica):
                Replica to run.
            results_file (str):
                Path to the results file.

        Returns:
            str | ReplicaException:
                ID of the replica that was run or a `ReplicaException` if an error
                was encountered.
        """
        logger = cls.getLoggerForReplica(replica, cls.logLevel)
        logger.debug(f"Starting replica: {replica.id}")
        try:
            with lock_data:
                logger.debug(f"Checking {replica.id} in results file: {results_file}")
                if cls.checkReplicaInResultsFile(replica, results_file):
                    logger.warning(f"Skipping {replica.id}. Already in results file.")
                    return replica.id
                logger.debug("Initializing data...")
                cls.initData(replica)
                logger.debug("Done.")
            logger.debug("Preparing data set...")
            replica.prepData()
            logger.debug("Done.")
            logger.debug("Initializing model...")
            replica.initModel()
            logger.debug("Done.")
            logger.debug("Running assessments...")
            replica.runAssessment()
            logger.debug("Done.")
            logger.debug("Creating report...")
            df_report = cls.replicaToReport(replica)
            logger.debug("Done.")
            with lock_report:
                logger.debug(f"Adding report to: {results_file}")
                cls.appendReportToResults(df_report, results_file)
                logger.debug("Done.")
            logger.debug(f"Finished replica: {replica.id}")
            return replica.id
        except Exception as e:
            logger.error(f"Error in replica '{replica}': {e}")
            traceback.print_exception(type(e), e, e.__traceback__)
            return cls.ReplicaException(replica.id, e)
