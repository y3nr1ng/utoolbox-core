import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def roll_by_frame(data, wnd_size=40, col_name='frame'):
    """
    Roll over frames using specified constant size uniform window.

    Parameters
    ----------
    data : pd.DataFrame
        Data source.
    wnd_size : int
        Rolling window size.
    col_name : str
        The column to roll with.

    Notes
    -----
    Frame number should start from 1.
    """
    # sort on the columns for efficient retrieval
    logger.debug("start sorting by frames")
    data.sort_values(col_name, ascending=True, inplace=True)

    frames = data[col_name].values
    n_frames = frames.max()
    logger.debug("{} frames provided".format(n_frames))

    indices = np.where(frames[:-1] != frames[1:])[0]
    indices += 1
    indices = np.concatenate(([0], indices, [n_frames]))

    def _iter_func():
        for start, end in zip(indices[:-wnd_size], indices[wnd_size:]):
            yield data.iloc[start:end]

    n_rolled = n_frames-wnd_size+1
    logger.debug("rolled result contains {} frames".format(n_rolled))
    return _iter_func(), n_rolled
