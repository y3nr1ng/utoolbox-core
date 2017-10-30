import os.path
import glob

class TimeSeries():
    """
    Load series of time points.
    """
    def __init__(self, dtype, folder=None, pattern='*'):
        if folder is not None:
            folder_path = os.path.join(folder, pattern)
            file_list = glob.glob(folder_path)
            self._timepoints = [TimePoint(dtype, f) for f in file_list]
        else:
            self._timepoints = []

    def __del__(self):
        del self._timepoints[:]

class TimePoint():
    """
    Load a file as an object representation.
    """
    def __init__(self, dtype, file_path=None):
        pass

    def __del__(self):
        pass
