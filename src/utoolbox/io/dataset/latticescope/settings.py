import configparser
import logging
import re
from collections import abc, namedtuple
from datetime import datetime
from enum import Enum

from .error import MalformedSettingsFileError

logger = logging.getLogger("utoolbox.io.dataset")


class AcquisitionMode(Enum):
    Z_STACK = "Z stack"
    SIM = "SI Scan SLM comb"


class ScanType(Enum):
    SAMPLE = "Sample piezo"
    OBJECTIVE = "Z galvo & piezo"


class TriggerMode(Enum):
    SLM = "SLM -> Cam"
    FPGA = "FPGA"


Channel = namedtuple(
    "Channel", ["id", "filter", "wavelength", "power", "exposure", "stacks"]
)


class Settings(dict):
    """
    Object-oriented modeling of the Settings.txt from LatticeScope.

    Args:
        lines (str): raw string from the file
    """

    section_pattern = re.compile(
        r"^(?:\*{5}\s{1}){3}\s*(?P<title>[^\*]+)(?:\s{1}\*{5}){3}", re.MULTILINE
    )

    def __init__(self, lines):
        super().__init__()

        sections = Settings.identify_sections(lines)
        for title, start, end in sections:
            try:
                section, parsed = {
                    "General": Settings.parse_general,
                    "Waveform": Settings.parse_waveform,
                    "Camera": Settings.parse_camera,
                    "Advanced Timing": Settings.parse_timing,
                    ".ini File": Settings.parse_hardware_settings,
                }[title](lines[start:end])
                self[section] = parsed
            except KeyError:
                logger.warning('unknown section "{}", ignored'.format(title))

    @property
    def path(self):
        return self._path

    @staticmethod
    def identify_sections(lines):
        """Determine sections and their positions (line number) in file."""
        titles, starts, ends = [], [], None
        for match in Settings.section_pattern.finditer(lines):
            titles.append(match.group("title").rstrip())
            starts.append(match.start())
        ends = starts[1:] + [len(lines)]

        return [(s, i0, i1) for s, i0, i1 in zip(titles, starts, ends)]

    @staticmethod
    def parse_general(lines):
        patterns = {
            "date": r"^Date :\t(\d+/\d+/\d+) .*",
            "time": r"^Date :\t.* (\d+:\d+:\d+).*",
            "mode": r"^Acq Mode :\t(.*)",
        }

        converter = {"mode": AcquisitionMode}

        parsed = {}
        for field, pattern in patterns.items():
            value = re.findall(pattern, lines, re.MULTILINE)[0]
            try:
                value = converter[field](value)
            except KeyError:
                # no need to convert
                pass
            parsed[field] = value

        # merge timestamp
        datetime_str = " ".join([parsed["date"], parsed["time"]])
        for timestamp_fmt in ("%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
            try:
                parsed["timestamp"] = datetime.strptime(datetime_str, timestamp_fmt)
                del parsed["date"], parsed["time"]
            except ValueError:
                pass
        else:
            logger.warning("unable to parse the timestamp")
            parsed["timestamp"] = datetime_str

        return "general", parsed

    @staticmethod
    def parse_waveform(lines):
        patterns = {
            "type": r"^Z motion :\t(.*)",
            "obj_piezo_step_size": r"Z PZT .* Interval \(um\), .* :\t\d*\.?\d+\t(\d*\.?\d+)\t\d+",
            "obj_piezo_n_steps": r"Z PZT .* Interval \(um\), .* :\t\d*\.?\d+\t\d*\.?\d+\t(\d+)",
            "sample_piezo_step_size": r"^S PZT .* Interval \(um\), .* :\t\d*\.?\d+\t(\d*\.?\d+)\t\d+",
            "sample_piezo_n_steps": r"^S PZT .* Interval \(um\), .* :\t\d*\.?\d+\t\d*\.?\d+\t(\d+)",
        }

        converter = {
            "type": ScanType,
            "obj_piezo_step_size": float,
            "obj_piezo_n_steps": int,
            "sample_piezo_step_size": float,
            "sample_piezo_n_steps": int,
        }

        parsed = {}
        for field, pattern in patterns.items():
            # NOTE exception here, multiple channels may exist
            value = re.findall(pattern, lines, re.MULTILINE)[0]
            try:
                value = converter[field](value)
            except KeyError:
                # no need to convert
                pass
            parsed[field] = value

        # NOTE exception, deal with multi-channel
        # TODO allow N/A filter
        ch_settings = re.findall(
            r"^Excitation Filter, Laser, Power \(%\), Exp\(ms\) \((\d+)\) :\t([\w/]+)\t(\d+)\t(\d+)\t(\d+(?:\.\d+)?)",
            lines,
            re.MULTILINE,
        )
        # sort by channel id
        ch_settings.sort(key=lambda t: t[0])

        ch_stacks = re.findall(r"^# of stacks \((\d+)\) :\t(\d+)", lines, re.MULTILINE)
        # sort by channel id
        ch_stacks.sort(key=lambda t: t[0])

        # map number of stacks
        ch_settings[:] = [
            settings + (n_stacks[1],)
            for settings, n_stacks in zip(ch_settings, ch_stacks)
        ]

        # convert power, exposure, stacks to numbers
        ch_settings[:] = [
            (id_, filter_, laser, float(power), float(exposure), int(stacks))
            for id_, filter_, laser, power, exposure, stacks in ch_settings
        ]

        parsed["channels"] = [Channel(*value) for value in ch_settings]

        return "waveform", parsed

    @staticmethod
    def parse_camera(lines):
        patterns = {
            "model": r"^Model :\t(.*)",
            "exposure": r"^Exp\(s\)\D+([\d\.]+)",
            "cycle": r"^Cycle\(s\)\D+([\d\.]+)",
            "roi": r"^ROI :\tLeft=(\d+) Top=(\d+) Right=(\d+) Bot=(\d+)",
            "binning": r"^Binning :\tX=(\d+) Y=(\d+)",
        }

        converter = {
            "exposure": lambda x: float(x),
            "cycle": lambda x: float(x),
            "roi": lambda x: tuple([int(i) for i in x]),
            "binning": lambda x: tuple([int(i) for i in x]),
        }

        parsed = {}
        for field, pattern in patterns.items():
            value = re.findall(pattern, lines, re.MULTILINE)[0]
            try:
                value = converter[field](value)
            except KeyError:
                # no need to convert
                pass
            parsed[field] = value

        return "camera", parsed

    @staticmethod
    def parse_timing(lines):
        patterns = {"mode": r"^Trigger Mode :\t(.*)"}

        converter = {"mode": TriggerMode}

        parsed = {}
        for field, pattern in patterns.items():
            value = re.findall(pattern, lines, re.MULTILINE)[0]
            try:
                value = converter[field](value)
            except KeyError:
                # no need to convert
                pass
            parsed[field] = value

        return "timing", parsed

    @staticmethod
    def parse_hardware_settings(lines):
        # drop the header line
        lines = lines.split("\n", 1)[1]

        return "hardware", HardwareSettings(lines)


def update_nested_dict(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            # update new value in dict
            d[k] = update_nested_dict(d.get(k, {}), v)
        elif isinstance(v, list):
            # append new value
            try:
                d[k].append(v)
            except AttributeError:
                d[k] = v
        # NOTE do we really need set in hardware dict?
        # elif isinstance(v, set):
        #    # add new value to set
        #    try:
        #        d[k].add(v)
        #    except AttributeError:
        #        d[k] = v
        else:
            # override pure value
            d[k] = v
    return d


class HardwareSettings(dict):
    """
    Args:
        lines (str): lines read from the .ini file

    Attributes:
        raw_config (ConfigParser): the main configuration parser from lines
    """

    def __init__(self, lines):
        super().__init__()

        config = configparser.ConfigParser()
        config.read_string(lines)

        def parse_section(section, func_table):
            try:
                parse_funcs = func_table[section]

                if not isinstance(parse_funcs, list):
                    parse_funcs = [parse_funcs]

                # call the parsers
                for parse_func in parse_funcs:
                    if not isinstance(parse_func, tuple):
                        parse_func = (parse_func,)

                    func, *args = parse_func
                    parsed = func(config[section], *args)
                    if parsed:
                        update_nested_dict(self, parsed)
            except KeyError:
                logger.debug('ignore section "{}"'.format(section))
            except Exception as error:
                logger.exception(error)

        # 1st pass
        for section in config.sections():
            parse_section(
                section,
                {
                    "Cam 1": (self.parse_camera_type, "Cam 1"),
                    "Cam 2": (self.parse_camera_type, "Cam 2"),
                    "Cam 3": (self.parse_camera_type, "Cam 3"),
                    "Cam 4": (self.parse_camera_type, "Cam 4"),
                    "Detection optics": self.parse_magnification,
                    "General": [self.parse_trigger_mode, self.parse_twin_cam_mode],
                    "Twin Cam Saving": self.parse_twin_cam_saving,
                },
            )

        # 2nd pass
        # NOTE these sections are optional, dependes on results from 1st pass
        for section in config.sections():
            parse_section(
                section,
                {"Hamamatsu Camera Settings": self.parse_hamamatsu_camera_settings},
            )

    ##

    def parse_camera_type(self, section, camera):
        if section["Enabled?"] == "TRUE":
            return {
                "detection": {"cameras": {camera: {"type": section["Type"].strip('"')}}}
            }
        else:
            return None

    def parse_hamamatsu_camera_settings(self, section):
        serials = {}
        mapping = {
            ("Cam 1", "Orca4.0"): "Orca 4.0 SN",
            ("Cam 2", "Orca4.0"): "Orca 4.0 SN (Twin Camera)",
            ("Cam 1", "Orca2.8"): "Orca 2.8 SN",
        }
        for camera, values in self["detection"]["cameras"].items():
            camera_type = values["type"]
            try:
                key = mapping[(camera, camera_type)]
                serials[camera] = {"serial": section[key].strip('"')}
            except KeyError:
                raise MalformedSettingsFileError(
                    f'unknown Hamamatsu combination "{camera}" ({camera_type})'
                )
        return {"detection": {"cameras": serials}}

    def parse_trigger_mode(self, section):
        return {"detection": {"trigger_mode": section["Cam Trigger Mode"].strip('"')}}

    def parse_twin_cam_mode(self, section):
        enabled = section["Twin cam mode?"] == "TRUE"
        return {"detection": {"twin_cam": {"enabled": enabled}}}

    def parse_magnification(self, section):
        return {"detection": {"magnification": float(section["magnification"])}}

    def parse_twin_cam_saving(self, section):
        partial_save = {}
        for key, save in section.items():
            camera, channel = re.search(
                r"saving camera ([a-zA-Z]{1}) (.*)", key
            ).groups()
            camera = f"Cam{camera.upper()}"
            partial_save[(camera, channel)] = save == "TRUE"
        return {"detection": {"twin_cam": {"partial_save": partial_save}}}
