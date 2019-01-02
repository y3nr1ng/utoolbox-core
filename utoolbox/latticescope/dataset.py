import copy
import glob
import logging
import os
import re

import imageio

from utoolbox.container import ImageDatastore
from .parse import Filename
from .refactor import (sort_timestamps, \
                       merge_fragmented_timestamps, \
                       rename_by_mapping)
from .settings import Settings

logger = logging.getLogger(__name__)

class Dataset(object):
    """
    Representation of an acquisition result from LatticeScope, containing
    software setup and collected data.
    """
    SETTINGS_PATTERN = '(?P<ds_name>.+)_Settings.txt$'

    def __init__(self, root, show_uri=False, refactor=True):
        """
        Parameters
        ----------
        root : str
            Source directory of the dataset, flat layout.
        show_uri: bool
            Return image URI when requested, default is False.
        refactor : bool
            Refactor filenames, default is True.
        """
        if not os.path.exists(root):
            raise FileNotFoundError("invalid root folder")
        self._root = root

        settings_file = self._find_settings_file()
        # NOTE some files have corrupted timestamp causing utf-8 decode error
        with open(settings_file, 'r', errors='ignore') as fd:
            lines = fd.read()
        self.settings = Settings(lines)

        n_channels = len(self.settings.waveform.channels)
        logger.info("{} channel(s) in settings".format(n_channels))




        data_files = self._list_data_files()
        if refactor:
            data_files_orig = copy.deepcopy(data_files)
            merge_fragmented_timestamps(data_files)
            rename_by_mapping(self.root, data_files_orig, data_files)

        #TODO partition the datastore by channels
        
        self._show_uri = show_uri
        self._datastore = None

        #TODO generate inventory file

    @property
    def datastore(self):
        if self._datastore is None:
            if self.show_uri:
                def read_func(x):
                    return (x, imageio.volread(x))
            else:
                read_func = imageio.volread
            self._datastore = ImageDatastore(self.root, read_func, sub_dir=False)
        return self._datastore

    @property
    def root(self):
        return self._root

    @property
    def show_uri(self):
        return self._show_uri

    @show_uri.setter
    def show_uri(self, state):
        if state != self.show_uri:
            self.show_uri = state
            self._datastore = None

    def _find_settings_file(self, extension='txt'):
        # settings are .txt files
        pattern = os.path.join(self.root, "*.{}".format(extension))
        filenames = [os.path.basename(fn) for fn in glob.glob(pattern)]

        ds_names = []
        for filename in filenames:
            matches = re.match(Dataset.SETTINGS_PATTERN, filename)
            if matches is None:
                continue
            ds_names.append(matches.group('ds_name'))
        
        if not ds_names:
            raise FileNotFoundError("no known settings file")
        elif len(ds_names) > 1:
            logger.warning("diverged dataset, attempting to resolve it")

            ds_names.sort()
            
            # use the longest common prefix to resolve it
            prefix = os.path.commonprefix(ds_names)
            if not prefix:
                raise RuntimeError(
                    "unable to determine which settings file to use"
                )
            ds_name = prefix
        else:
            ds_name = ds_names[0]

        return os.path.join(self.root, ds_name)

    def _list_data_files(self, sort=True):
        data_files = []
        filenames = os.listdir(self.root)
        for filename in filenames:
            _, extension = os.path.splitext(filename)
            if extension != '.tif':
                continue
            try:
                parsed = Filename(filename)
                data_files.append(parsed)
            except:
                logger.warning("invalid format \"{}\", ignored".format(filename))
        if sort:
            sort_timestamps(data_files)
        return data_files
