import numpy as np

class SimpleVolume(np.ndarray):
    def __init__(self, file_path):
        print(file_path)

    def __del__(self):
        pass

class SIVolume(object):
    pass
