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

    def __len__(self):
        return len(self._timepoints)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._timepoints[i] for i in xrange(*key.indices(len(self)))]
        elif isinstance(key, int):
            return self._timepoints[key]
        else:
            raise TypeError('Invalid argument type')

    def __del__(self):
        del self._timepoints[:]

class TimePoint():
    """
    Load a file as an object representation.
    """
    def __init__(self, dtype, file_path=None):
        self.dtype = dtype

        #TODO call the constructor of the primitive type
        dtype(file_path)

    def __del__(self):
        pass
