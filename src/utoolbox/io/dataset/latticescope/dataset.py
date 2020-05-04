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


class LatticeScopeDataset(
    DirectoryDataset,
    DenseDataset,
    MultiChannelDataset,
    MultiViewDataset,
    TimeSeriesDataset,
):
    @property
    def read_func(self):
        def func(uri, shape, dtype):
            print(f'read_func "{uri}"')
            array = da.from_delayed(
                delayed(imageio.volread, pure=True)(uri), shape, dtype
            )
            return array

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
        return glob.glob(search_path)

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
        shape = (bottom - top + 1, right - left + 1)

        if self.metadata["general"]["mode"] == AcquisitionMode.Z_STACK:
            scan_type = self.metadata["waveform"]["type"]
            key = {
                ScanType.OBJECTIVE: "obj_piezo_n_steps",
                ScanType.SAMPLE: "sample_piezo_n_steps",
            }[scan_type]
            nz = self.metadata["waveform"][key]
            if nz > 1:
                shape = (nz,) + shape

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
        # TODO dirty patch, fix this
        with open(self.settings_path, "r", errors="ignore") as fd:
            for line in fd:
                if line.startswith("Twin cam mode?"):
                    _, flag = line.split("=")
                    flag = flag.strip()
                    if flag == "TRUE":
                        return ("CamA", "CamB")
                    else:
                        return None
            else:
                raise MalformedSettingsFileError("cannot find twin camera flag")

    def _load_voxel_size(self):
        value = input_dialog(
            title="Missing Info", text="What is the size of a single pixel? "
        ).run()
        self._pixel_size = float(value)
        size = (self._pixel_size,) * 2

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

    def _retrieve_file_list(self, coord_dict, single=True):
        # filter by view, might only has single view
        if "view" in coord_dict:
            filtered = [f for f in self.files if coord_dict["view"] in f]
        else:
            filtered = self.files
        # filter by channel and timestamp
        ich = self._lookup_channel_id(coord_dict["channel"])
        its = coord_dict["time"].to_timedelta64()  # ns
        its = int(its) // 1000000  # ms
        filtered = [f for f in filtered if f"ch{ich}" in f and f"_{its:07d}msec_" in f]

        if single:
            assert len(filtered) == 1, "found more than 1 file matches, stmt error"
            return filtered[0]
        else:
            return filtered


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

    def _load_tiling_coordinates(self):
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
        file_list = super()._retrieve_file_list(coord_dict)

        # generate queries
        statements = [f"{k}=={coord_dict[k]}" for k in ("tile_x", "tile_y", "tile_z")]
        query_stmt = " & ".join(statements)
        # find tile linear index
        index = self.tile_coords.query(query_stmt).index.values

        try:
            # stacked data, only 1 file
            index = index[0]
            return file_list[index]
        except IndexError:
            logger.debug(f'invalid statement "{query_stmt}"')
            # secondary filter failed
            return None
