from functools import lru_cache
from dask.distributed import Client, LocalCluster, as_completed
import logging
import atexit

__all__ = ["get_local_cluster", "get_client", "wait_futures"]

logger = logging.getLogger("utoolbox.utils.dask")

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


def get_client(*args, **kwargs):
    cluster = get_local_cluster(*args, **kwargs)
    client = Client(cluster, silence_logs="error")
    logger.info(f'new client connection "{client}"')
    atexit.register(client.close)
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
