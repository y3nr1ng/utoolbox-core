import logging

from .dense import DenseDataset
from .timeseries import TimeSeriesDataset
from .serialize import SerializableDataset

logger = logging.getLogger(__name__)


class SpimDataset(DenseDataset, TimeSeriesDataset, SerializableDataset):
    pass


if __name__ == "__main__":
    path = "/scratch/20170718_U2Os_BLStimu/cell2_FTmChG6s_zp6um_20ms_interval_6s/raw/"
    dataset = SpimDataset(path)
