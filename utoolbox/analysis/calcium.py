import numpy as np

def _f0(data, r):
    """Generate the F0 image by averaging the first few frames."""
    return np.mean(data[r[0]:r[1]], axis=0)

def f_f0(data, f0_range):
    """Measure raw intensity (F) fluctuation.

    Parameters
    ----------
    data : TimeSeries
        Source dataset.
    f0_range : tuple or integer
        If f0_range is an integer, it denotes [0, T), otherwise, as a tuple, it
        denotes [T_0, T_1).
    """
    #TODO ensure data is of type TimeSeries

    if isinstance(f0_range, int):
        f0_range = (0, f0_range)
    else:
        f0_range = tuple(f0_range)
    f0 = _f0(data, f0_range)

    with np.errstate(divide='ignore'):
        data = np.nan_to_num(data/f0, copy=False)
    # blank the f0 baselines
    data[f0_range[0]:f0_range[1], ...] = 0

    # scale the result to [0., 1.]
    return (data - data.min()) / np.ptp(data)

def df_f0(data, f0_range, mode='increase'):
    """Measure fluorescence variation (delta-F) fluctuations.

    Parameters
    ----------
    data : TimeSeries
        Source dataset.
    f0_range : tuple or integer
        If f0_range is an integer, it denotes [0, T), otherwise, as a tuple, it
        denotes [T_0, T_1).
    mode : 'increase' or 'decrease'
        Determine the direction of detection, 'increase' indicates F_{n+1}-F_n,
        in order to have intensity increase as positive, vice versa, 'decrease'
        indicates F_n-F{n-1}.
    """
    pass

def delta_f(data):
    """Measure rapid frame-to-frame changes in intensity."""
    raise NotImplementedError
