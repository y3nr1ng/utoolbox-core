import glob
import logging
import os

import imageio

from utoolbox.container import ImageDatastore
from . import Filename

logger = logging.getLogger(__name__)

class Dataset(object):
    """
    Representation of an acquisition result from LatticeScope, containing
    software setup and collected data.
    """
    def __init__(self, root, refactor=True):
        self.root = root

        #TODO parse _Settings.ini
        self._settings = None

        if refactor:
            #TODO refactor the dataset
            pass

        self._datastore = ImageDatastore(root, imageio.volread, sub_dir=False)
        #TODO partition the datastore by channels

        #TODO generate inventory file

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, src_dir):
        pathname = os.path.join(src_dir, "*.tif")
        filenames = glob.glob(pathname, recursive=False)
        Dataset._validate_data_format(filenames)
        self._root = src_dir

    @staticmethod
    def _validate_data_format(filenames):
        for filename in filenames:
            filename = os.path.basename(filename)
            try:
                _ = Filename(filename)
            except ValueError:
                logger.error("invalid filename format ({})".format(filename))
                raise
