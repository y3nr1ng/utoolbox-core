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
        logger.debug("settings file \"{}\"".format(settings_file))
        # NOTE some files have corrupted timestamp causing utf-8 decode error
        with open(settings_file, 'r', errors='ignore') as fd:
            lines = fd.read()
        self.settings = Settings(lines)

        n_channels = len(self.settings.waveform.channels)
        logger.info("{} channel(s) in settings".format(n_channels))

        # if requested, wrap the reader before creating datastores
        if show_uri:
            def read_func(x):
                return (x, imageio.volread(x))
        else:
            read_func = imageio.volread

        # partition the dataset to different datastore by channels 
        self._datastore = dict()
        for channel in self.settings.waveform.channels:
            if channel.wavelength in self._datastore:
                logger.warning("duplicated wavelength, ignore")
                continue
            self._datastore[channel.wavelength] = ImageDatastore(
                self.root,
                read_func,
                sub_dir=False,
                pattern="*_ch{}_*".format(channel.id)
            )

        """
        if refactor:
            data_files_orig = copy.deepcopy(data_files)
            merge_fragmented_timestamps(data_files)
            rename_by_mapping(self.root, data_files_orig, data_files)
        """

        #TODO generate inventory file

    @property
    def datastore(self):
        return self._datastore

    @property
    def root(self):
        return self._root

    def _find_settings_file(self, extension='txt'):
        # settings are .txt files
        search_pattern = os.path.join(self.root, "*.{}".format(extension))
        filenames = glob.glob(search_pattern)

        ds_names = []
        for filename in filenames:
            basename = os.path.basename(filename)
            matches = re.match(Dataset.SETTINGS_PATTERN, basename)
            if matches is None:
                continue
            ds_names.append((matches.group('ds_name'), filename))
        
        if not ds_names:
            raise FileNotFoundError("no known settings file")
        elif len(ds_names) > 1:
            logger.warning("diverged dataset, attempting to resolve it")

            # sort by name of dataset instead of actual path
            ds_names.sort(key=lambda t: t[0])
            
            # use the longest common prefix to resolve it
            ds_names_tr = list(zip(*ds_names))
            prefix = os.path.commonprefix(ds_names_tr[0])
            try:
                index = ds_names_tr[0].index(prefix)
            except ValueError:
                raise RuntimeError(
                    "unable to determine which settings file to use"
                )
            return ds_names[index][1]
        else:
            return ds_names[0][1]

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
