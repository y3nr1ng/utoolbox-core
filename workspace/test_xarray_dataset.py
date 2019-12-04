import logging
import os
from pprint import pprint

import coloredlogs

from utoolbox.io.dataset.mm import MicroManagerV1Dataset

if __name__ == "__main__":
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    src_dir = (
        "S:/Andy/20191119_ExM_kidney_10XolympusNA06_zp5_7x8_DKO_4-2_Nkcc2_488_slice_2_1"
    )
    ds = MicroManagerV1Dataset(src_dir)

    print(ds.dataset.coords)
