from collections import namedtuple
from datetime import datetime
from enum import Enum
import logging
import re

from utoolbox.util import AttrDict

logger = logging.getLogger(__name__)


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


class Settings(AttrDict):
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
                    # ".ini File": None, # TODO analyze setup file
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

        parsed = AttrDict()
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

        parsed = AttrDict()
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
        }

        converter = {
            "exposure": lambda x: float(x),
            "cycle": lambda x: float(x),
            "roi": lambda x: tuple([int(i) for i in x]),
        }

        parsed = AttrDict()
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

        parsed = AttrDict()
        for field, pattern in patterns.items():
            value = re.findall(pattern, lines, re.MULTILINE)[0]
            try:
                value = converter[field](value)
            except KeyError:
                # no need to convert
                pass
            parsed[field] = value

        return "timing", parsed


class HardwareSettings(object):
    def __init__(self, lines):
        pass
