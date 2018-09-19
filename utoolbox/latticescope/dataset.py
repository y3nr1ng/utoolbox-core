import copy
import glob
import logging
import os

import imageio

from utoolbox.container import ImageDatastore
from .parse import Filename
from .refactor import sort_timestamps, merge_fragmented_timestamps, rename_by_mapping
from .settings import Settings

logger = logging.getLogger(__name__)

class Dataset(object):
    """
    Representation of an acquisition result from LatticeScope, containing
    software setup and collected data.
    """
    def __init__(self, root, refactor=True):
        """
        Parameters
        ----------
        root : str
            Source directory of the dataset, flat layout.
        refactor : bool
            Refactor filenames, default is True.
        """
        if not os.path.exists(root):
            raise FileNotFoundError("invalid root folder")
        self._root = root

        settings, data_filenames = self._list_files()
        sort_timestamps(data_filenames)
        if refactor:
            old_data_filenames = copy.deepcopy(data_filenames)
            merge_fragmented_timestamps(data_filenames)
            rename_by_mapping(self.root, old_data_filenames, data_filenames)

        if not os.path.commonprefix([settings, data_filenames[0].name]):
            logger.warning("sample name mismatched, possibly mixed up")
        with open(settings, 'r') as fd:
            lines = fd.read()
        self.settings = Settings(lines)

        #TODO partition the datastore by channels
        n_channels = len(self.settings.waveform.channels)
        logger.info("{} channel(s) in settings".format(n_channels))

        self.datastore = ImageDatastore(root, imageio.volread, sub_dir=False)

        #TODO generate inventory file

    @property
    def root(self):
        return self._root

    def _list_files(self):
        settings_filename, data_filenames = None, []
        filenames = os.listdir(self.root)
        for filename in filenames:
            if filename.endswith('_Settings.txt'):
                if settings_filename is not None:
                    raise FileExistsError("multiple settings found")
                settings_filename = filename
            else:
                try:
                    parsed = Filename(filename)
                    data_filenames.append(parsed)
                except:
                    logger.warning("invalid format \"{}\", ignored".format(filename))
        return settings_filename, data_filenames
