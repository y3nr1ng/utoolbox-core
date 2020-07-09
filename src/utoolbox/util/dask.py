import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional

from dask.distributed import Client, as_completed

__all__ = ["get_client", "wait_futures"]

logger = logging.getLogger("utoolbox.util.dask")

DEFAULT_WORKER_OPTIONS = {
    "memory_target_fraction": 0.5,
    "memory_spill_fraction": False,
    "memory_pause_fraction": 0.75,
}

ASZARR_SLURM_SPEC = {
    "cores": 8,
    "processes": 1,
    "memory": "32GB",
    "project": "aszarr",
    "queue": "CPU",  # TODO update to use merged queue
    "walltime": "24:00:00",  # 1d
}


class ManagedCluster(ABC):
    def __init__(self, n_workers=4, threads_per_worker=2, memory="16GB"):
        self._n_workers = n_workers
        self._threads_per_worker = threads_per_worker
        self._memory = memory

        self._cluster, self._client = None, None

    def __enter__(self):
        self._open()
        self._wait_ready()
        return self

    def __exit__(self, *exc):
        self.close()

    ##

    @property
    def cluster(self):
        assert self._cluster is not None, "cluster is not opened yet"
        return self._cluster

    @property
    def client(self) -> Client:
        return self._client

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def threads_per_worker(self) -> int:
        return self._threads_per_worker

    @property
    def memory(self) -> str:
        return self._memory

    @property
    def scheduler_address(self) -> str:
        return self.client.scheduler_info()["address"]

    @property
    def dashboard_address(self) -> Optional[str]:
        try:
            return self.client.scheduler_info()["services"]["dashboard"]
        except KeyError:
            return None

    ##

    @abstractmethod
    def open(self):
        """
        Start up a cluster instance.

        Returns:
            (object): a viable dask clusterf
        """

    def close(self):
        """Stop the managed cluster instance."""
        self.client.close()
        self._client = None
        logger.debug("client disconnected")

        self.cluster.close()
        self._cluster = None
        logger.debug("cluster shutdown")

    ##

    def _open(self):
        """Start up the client."""
        self.open()  # implementation from childs, start the cluster
        self._client = Client(self.cluster)

        logger.info(
            f"established cluster connection (scheduler: {self.scheduler_address})"
        )

        dashboard_address = self.dashboard_address
        if dashboard_address is None:
            logger.debug(f"no dashboard")
        else:
            logger.info(f"dashboard: {dashboard_address}")

    def _wait_ready(self):
        """Wait for the cluster to prepare all its workers."""
        self.client.wait_for_workers(self.n_workers)


class ManagedLocalCluster(ManagedCluster):
    def __init__(self, address=None, n_workers=None, threads_per_worker=None, **kwargs):
        # default to utilize all cores on local system
        super().__init__(
            n_workers=n_workers, threads_per_worker=threads_per_worker, **kwargs
        )

    def open(self):
        from dask.distributed import LocalCluster

        self._cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory,
        )


class ManagedSLURMCluster(ManagedCluster):
    """
    Args:
        project (str, optional): project name
        queue (str, optional): queue to submit to
        walltime (str, optional): maximum wall time
    """

    def __init__(self, project=None, queue="batch", walltime="24:00:00", **kwargs):
        super().__init__(**kwargs)
        self._project = project
        self._queue = queue
        self._walltime = walltime

    def open(self):
        from dask_jobqueue import SLURMCluster

        args = {
            "cores": self.threads_per_worker,
            "processes": 1,
            "memory": self.memory,
            "project": self._project,
            "queue": self._queue,
            "walltime": self._walltime,
            "log_directory": "/tmp",
        }
        self._cluster = SLURMCluster(**args)
        self._cluster.scale(self.n_workers)


@contextmanager
def get_client(
    address=None, auto_spawn=True, worker_log_level="ERROR", **clustser_kwargs
):
    """
    Args:
        address (str, optional): address of the cluster scheduler, or 'slurm' to launch
            a dask cluster through SLURM
        auto_spawn (bool, optional): automagically spawn cluster if not found
        work_log_level (str, optional): worker log level
    """
    cluster_klass = None
    if address == "slurm":
        # create SLURM jobs
        cluster_klass = ManagedSLURMCluster
    elif address is None:
        # nothing specified, use:
        #   - already connected client
        #   - spawn new local cluster
        try:
            # we try to acquire current session first
            client = Client.current()

            address = client.scheduler_info()["address"]
            logger.info(f"connect to existing cluster (scheduler: {address})")

            yield client
        except ValueError:
            # nothing exists, continue to spawn managed cluster
            if not auto_spawn:
                raise RuntimeError("please spawn a dask cluster first")

        cluster_klass = ManagedLocalCluster

        # local cluster needs address info
        clustser_kwargs.update({"address": address})
    else:
        # directly specify the scheduler to connect to
        yield Client(address)

    if not cluster_klass:
        # nothing to spawn and manage
        return

    with cluster_klass(**clustser_kwargs) as cluster:
        client = cluster.client

        # register loggers
        try:
            import coloredlogs
        except ImportError:
            logger.install("install `coloredlogs` to configure loggers automatically")
        else:

            def install_logger(dask_worker):
                # we know this is annoying, silence it
                logging.getLogger("tifffile").setLevel(logging.ERROR)

                coloredlogs.install(
                    level=worker_log_level,
                    fmt="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S",
                )

            logger.debug(f'install logger for workers, level="{worker_log_level}"')
            client.register_worker_callbacks(install_logger)

        yield client


def wait_futures(futures, return_failed=False, show_bar=True):
    """
    Monitor all the futures and return failed futures.

    Args:
        futures (list of Futures): list of futures
    """
    iterator = as_completed(futures)
    if show_bar:
        try:
            from tqdm import tqdm
        except ImportError:
            logger.warning('requires "tqdm" to show progress bar')
        else:
            iterator = tqdm(iterator, total=len(futures))

    failed_futures = []
    for future in iterator:
        try:
            future.result()
        except Exception as error:
            logger.exception(error)
            failed_futures.append(future)

        del future  # release

    if failed_futures:
        logger.error(f"{len(failed_futures)} task(s) failed")

    if return_failed:
        return failed_futures
