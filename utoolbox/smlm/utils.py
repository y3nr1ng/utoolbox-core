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
    logger.debug("sorting by frames")
    data.sort_values(col_name, ascending=True, inplace=True)

    frames = data[col_name].values
    indices = np.where(frames[:-1] != frames[1:])[0]
    indices += 1

    def _iter_func():
        start = 0
        for end in indices[wnd_size-1::wnd_size]:
            yield data.iloc[start:end]
            start = end
        # last segment
        yield data.iloc[start:]

    max_frames = frames.max()
    n_rolled = max_frames-wnd_size+1
    logger.info("{} frames provided, rolled result contains {} frames".format(max_frames, n_rolled))
    return _iter_func(), n_rolled
