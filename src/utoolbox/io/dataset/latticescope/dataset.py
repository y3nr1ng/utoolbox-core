import glob
import logging
import os
import re
from collections import defaultdict
from io import StringIO

import dask.array as da
import imageio
import numpy as np
import pandas as pd
from dask import delayed
from prompt_toolkit.shortcuts import input_dialog

from ..base import (
    TILE_INDEX_STR,
    DenseDataset,
    DirectoryDataset,
    MultiChannelDataset,
    MultiViewDataset,
    TiledDataset,
    TimeSeriesDataset,
)
from .error import MalformedSettingsFileError, MissingSettingsFileError
from .settings import AcquisitionMode, ScanType, Settings

__all__ = ["LatticeScopeDataset", "LatticeScopeTiledDataset"]

logger = logging.getLogger("utoolbox.io.dataset")

# internally coded camera pixel size
# FIXME provide an external lookup source
PIXEL_SIZE_LUT = {"Orca4.0": 6.5}


class LatticeScopeDataset(
    DirectoryDataset,
    DenseDataset,
    MultiChannelDataset,
    MultiViewDataset,
    TimeSeriesDataset,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_fragments = 1

    ##

    @property
    def is_fragmented(self) -> bool:
        return self._n_fragments > 1

    @property
    def n_fragments(self) -> int:
        """How many pieces are the file fragmented into?"""
        return self._n_fragments

    @property
    def read_func(self):
        def volread_np(uri):
            return np.array(imageio.volread(uri))

        def func(uri, shape, dtype):
            if self.is_fragmented:
                # build parts list
                fbase, fext = os.path.splitext(uri)
                uri = [uri]
                for i in range(1, self.n_fragments):
                    part_uri = f"{fbase}_part{i:04d}{fext}"
                    uri.append(part_uri)

                # probe the shape
                if not self._fragmented_shapes:
                    shapes = []
                    for part_uri in uri:
                        reader = imageio.get_reader(part_uri)
                        nz = reader.get_length()
                        nxy = next(reader.iter_data()).shape
                        shapes.append((nz,) + nxy)
                    self._fragmented_shapes = tuple(shapes)

                # load and concat file parts
                arrays = []
                for part_uri, part_shape in zip(uri, self._fragmented_shapes):
                    part_array = da.from_delayed(
                        delayed(volread_np, pure=True)(part_uri), part_shape, dtype
                    )
                    arrays.append(part_array)
                # fragments are 3D-only, we concat them at the slowest axis
                array = da.concatenate(arrays, axis=0)

                if array.shape != shape:
                    logger.warning(f'"{os.path.basename(uri[0])}" is incomplete')

                return array
            else:
                # simple array
                return da.from_delayed(
                    delayed(volread_np, pure=True)(uri), shape, dtype
                )

        return func

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def settings_path(self):
        return self._settings_path

    ##

    def _can_read(self):
        return bool(self.metadata)

    def _enumerate_files(self):
        search_path = os.path.join(self.root_dir, "*.tif")
        file_list, partial_suffix = [], set()
        for fname in glob.glob(search_path):
            result = re.search(r"_part(\d+).t", fname)
            if result:
                # this is a partial file
                partial_suffix.add(result.group(1))
            else:
                file_list.append(fname)

        # update fragments count
        self._n_fragments = 1 + len(partial_suffix)

        if self.is_fragmented:
            logger.info(
                f"fragmented TIFF, each data contains {self.n_fragments} pieces"
            )

        return file_list

    def _find_settings_path(self):
        # find common prefix
        file_list = []
        for ext in ("tif", "txt"):
            file_list.extend(glob.glob(os.path.join(self.root_dir, f"*.{ext}")))
        prefix = os.path.commonprefix(file_list)
        # strip until ends with an underscore
        if prefix[-1] == "_":
            prefix = prefix[:-1]
        logger.debug(f'dataet prefix "{prefix}"')

        # find settings
        settings_path = f"{prefix}_Settings.txt"
        if not os.path.exists(settings_path):
            raise MissingSettingsFileError()
        return settings_path

    def _load_array_info(self):
        camera = self.metadata["camera"]
        left, top, right, bottom = camera["roi"]
        bin_x, bin_y = camera["binning"]

        logger.debug(f"binning (x={bin_x}, y={bin_y})")
        # NOTE camera should coerce the binning range to proper range, using int-div
        shape = ((bottom - top + 1) // bin_y, (right - left + 1) // bin_x)

        if self.metadata["general"]["mode"] == AcquisitionMode.Z_STACK:
            scan_type = self.metadata["waveform"]["type"]
            key = {
                ScanType.OBJECTIVE: "obj_piezo_n_steps",
                ScanType.SAMPLE: "sample_piezo_n_steps",
            }[scan_type]
            nz = self.metadata["waveform"][key]
            if nz > 1:
                shape = (nz,) + shape

        if self.is_fragmented:
            # we will need this to store the precise shape later
            self._fragmented_shapes = None

        # NOTE assuming fixed at 16-bit
        return shape, np.uint16

    def _load_channel_info(self):
        channels = self.metadata["waveform"]["channels"]
        return [c.wavelength for c in channels]

    def _load_metadata(self):
        settings_path = self._find_settings_path()
        with open(settings_path, "r", errors="ignore") as fd:
            metadata = Settings(fd.read())
        self._settings_path = settings_path
        return metadata

    def _load_timestamps(self):
        pattern = r"_stack(\d+)_.*_(\d+)msec_"
        pairs = defaultdict(list)
        for filename in self.files:
            try:
                stackno, rel_ts = re.search(pattern, filename).groups()
                stackno, rel_ts = int(stackno), np.timedelta64(rel_ts, "ms")
            except (AttributeError, ValueError):
                # no match!
                logger.error(f'malformed filename "{os.path.basename(filename)}"')
            pairs[stackno].append(rel_ts)

        # deduplicate
        dedup_pairs = []
        for stackno, ts in pairs.items():
            if not all(t == ts[0] for t in ts):
                logger.warning(
                    f"a channel/view combination yields inconsistent timestamp"
                )
            dedup_pairs.append((stackno, ts[0]))

        # sort timestamps by stackno
        dedup_pairs.sort(key=lambda x: x[0])
        return [rel_ts for _, rel_ts in dedup_pairs]

    def _load_view_info(self):
        if self.metadata["hardware"]["detection"]["twin_cam"]["enabled"]:
            return ("CamA", "CamB")
        else:
            return None

    def _load_voxel_size(self):
        camera_type = set()
        for attrs in self.metadata["hardware"]["detection"]["cameras"].values():
            camera_type.add(attrs["type"])
        if len(camera_type) > 1:
            raise MalformedSettingsFileError(
                "when I write this stuff, it does not support twin-cam with different brand"
            )
        camera_type = next(iter(camera_type))

        try:
            value = PIXEL_SIZE_LUT[camera_type]
            logger.debug(f'camera identified as "{camera_type}", pixel size {value} um')
            mag = self.metadata["hardware"]["detection"]["magnification"]
            value /= mag
            logger.info(
                f"detection magnification {mag}, effective pixel size {value:.4f} um"
            )
        except KeyError:
            value = input_dialog(
                title="Unknown camera model",
                text=f'What is the size of a single pixel for "{camera_type}"? ',
            ).run()
            value = float(value)
        self._pixel_size = value
        size = (self._pixel_size,) * 2  # assuming pixel size is isotropic

        if self.metadata["general"]["mode"] == AcquisitionMode.Z_STACK:
            scan_type = self.metadata["waveform"]["type"]
            key = {
                ScanType.OBJECTIVE: "obj_piezo_step_size",
                ScanType.SAMPLE: "sample_piezo_step_size",
            }[scan_type]
            size = (self.metadata["waveform"][key],) + size

        return size

    def _lookup_channel_id(self, wavelength):
        for channel in self.metadata["waveform"]["channels"]:
            if channel.wavelength == wavelength:
                return channel.id
        else:
            raise ValueError(f'channel "{channel}" is not enlisted in the settings')

    def _retrieve_file_list(self, coord_dict, cascade=False):
        file_list = self.files

        # view
        if "view" in coord_dict:
            file_list = [f for f in file_list if coord_dict["view"] in f]

        # channel
        ich = self._lookup_channel_id(coord_dict["channel"])
        file_list = [f for f in file_list if f"ch{ich}" in f]

        # timepoint
        if "time" in coord_dict:
            its = coord_dict["time"].to_timedelta64()  # ns
            its = int(its) // 1000000  # ms
            file_list = [f for f in file_list if f"_{its:07d}msec_" in f]

        if not cascade:
            assert len(file_list) == 1, "multiple files match the search condition"
            return file_list[0]
        else:
            return file_list


class LatticeScopeTiledDataset(LatticeScopeDataset, TiledDataset):
    @property
    def script_path(self):
        return self._script_path

    ##

    def _can_read(self):
        if not super()._can_read():
            return False

        # find script file
        script_path = glob.glob(os.path.join(self.root_dir, "*.csv"))
        if len(script_path) == 0:
            return False
        elif len(script_path) > 1:
            logger.warning(f"found {len(script_path)} script file candidates")
            for path in script_path:
                with open(path, "r") as fd:
                    first_line = fd.readline()
                    if first_line.startswith("# Subvolume X"):
                        logger.info(f'using "{os.path.basename(path)}" as script file')
                        self._script_path = path
                        break
            else:
                logger.error("none of the candidates are valid")
                return False
        else:
            self._script_path = script_path[0]

        return True

    def _load_coordinates(self):
        """
        NOTE Do NOT use the Stack X/Y/Z in script.csv! They are might be inconsistent
        with coordinates when manually modified. This is also why I implement
        `_load_coordinates` instead of `_load_mapped_coordinates`.
        """
        # cleanup the file
        with open(self.script_path, "r", encoding="unicode_escape") as fd:
            ignore_start, ignore_end = -1, -1
            for i, line in enumerate(fd):
                if line.startswith("----"):
                    if ignore_start < 0:
                        # first encounter
                        ignore_start = i
                    else:
                        # latest encounter
                        ignore_end = i

            fd.seek(0)

            if ignore_start < 0:
                script_raw = StringIO(fd.read())
            else:
                logger.info("found correction scan info, filtering lines...")
                # requires filtering
                lines = []
                for i, line in enumerate(fd):
                    if i < ignore_start or i > ignore_end:
                        lines.append(line)
                script_raw = StringIO("".join(lines))

        # ln 3-N, position list
        coords = pd.read_csv(script_raw, skiprows=2)
        coords.dropna(how="all", axis="columns", inplace=True)

        # rename to internal header
        coords.rename(
            {
                "Absolute X (um)": "tile_x",
                "Absolute Y (um)": "tile_y",
                "Absolute Z (um)": "tile_z",
            },
            axis="columns",
            inplace=True,
        )
        # keep only these columns
        coords = coords[["tile_x", "tile_y", "tile_z"]]
        # ensure we have np.float32
        coords = coords.astype(np.float32)
        # keep the scanning order, LatticeScope use this as saving order
        return coords

    def _retrieve_file_list(self, coord_dict):
        file_list = super()._retrieve_file_list(coord_dict, cascade=True)

        # generate queries
        statements = [f"{k}=={coord_dict[k]}" for k in TILE_INDEX_STR]
        query_stmt = " & ".join(statements)
        # find tile linear index
        result = self.tile_coords.query(query_stmt)

        try:
            # stacked data, only 1 file
            index = self.tile_coords.index.get_loc(result.iloc[0].name)
        except IndexError:
            logger.debug(f'invalid statement "{query_stmt}"')
            # secondary filter failed
            return None
        else:
            return file_list[index]
