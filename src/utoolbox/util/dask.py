from functools import lru_cache
from dask.distributed import Client, LocalCluster, as_completed
import logging
import atexit

__all__ = ["get_local_cluster", "get_client", "wait_futures"]

logger = logging.getLogger("utoolbox.util.dask")

DEFAULT_WORKER_OPTIONS = {
    "memory_target_fraction": 0.5,
    "memory_spill_fraction": False,
    "memory_pause_fraction": 0.75,
}


@lru_cache(maxsize=1)
def get_local_cluster(*, scheduler_port=0, **kwargs):
    """
    Get a cached local cluster.
    """
    args = DEFAULT_WORKER_OPTIONS.copy()
    args.update(kwargs)
    cluster = LocalCluster(scheduler_port=scheduler_port, **args)
    port = cluster.scheduler.port
    logger.info(f'launch a local cluster at "localhost:{port}"')
    return cluster


def get_client(worker_log_level="ERROR", *args, **kwargs):
    cluster = get_local_cluster(*args, **kwargs)
    client = Client(cluster)
    logger.info(f'new client connection "{client}"')

    def close_client():
        logger.info(f'client "{client}" closed')
        client.close()

    atexit.register(close_client)

    # register loggers
    try:
        import coloredlogs
    except ImportError:
        logger.install("install `coloredlogs` to configure loggers automatically")
    else:

        def install_logger(dask_worker):
            coloredlogs.install(
                level=worker_log_level,
                fmt="%(asctime)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )

        logger.debug(f'install logger for workers, level="{worker_log_level}"')
        client.register_worker_callbacks(install_logger)

    return client


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
