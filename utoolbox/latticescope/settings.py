import configparser
from datetime import datetime
from enum import Enum
import logging
import os
import re

from utoolbox.container import AttrDict

logger = logging.getLogger(__name__)

class AcquisitionMode(Enum):
    Z_STACK = "Z stack"

class ScanType(Enum):
    SAMPLE = "Sample piezo"
    OBJECTIVE = "Z objective & galvo"

class TriggerMode(Enum):
    SLM = "SLM -> Cam"
    FPGA = "FPGA"

class Settings(AttrDict):
    section_pattern = re.compile(
        '^(?:\*{5}\s{1}){3}\s*(?P<title>[^\*]+)(?:\s{1}\*{5}){3}',
        re.MULTILINE
    )

    def __init__(self, lines):
        super(Settings, self).__init__()

        sections = Settings.identify_sections(lines)
        for title, start, end in sections:
            try:
                section, parsed = {
                    'General': Settings.parse_general,
                    'Waveform': Settings.parse_waveform,
                    'Camera': Settings.parse_camera,
                    'Advanced Timing': Settings.parse_timing,
                    '.ini File': None
                }[title](lines[start:end])
                self[section] = parsed
            except:
                logger.warning("unknown section \"{}\", ignored".format(title))

    @property
    def path(self):
        return self._path

    @staticmethod
    def identify_sections(lines):
        """Determine sections and their positions (line number) in file."""
        titles, starts, ends = [], [], None
        for match in Settings.section_pattern.finditer(lines):
            titles.append(match.group('title').rstrip())
            starts.append(match.start())
        ends = starts[1:] + [len(lines)]

        return [
            (s, i0, i1) for s, i0, i1 in zip(titles, starts, ends)
        ]

    @staticmethod
    def parse_general(lines):
        patterns = {
            'timestamp': '^Date :\t(\d+/\d+/\d+ \d+:\d+:\d+ [A|P]M)',
            'mode': '^Acq Mode :\t(.*)'
        }

        converter = {
            'timestamp': lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'),
            'mode': AcquisitionMode
        }

        parsed = AttrDict()
        for field, pattern in patterns.items():
            value = re.findall(pattern, lines, re.MULTILINE)[0]
            try:
                value = converter[field](value)
            except:
                # no need to convert
                pass
            parsed[field] = value

        return 'general', parsed

    @staticmethod
    def parse_waveform(lines):
        patterns = {
            'type': '^Z motion :\t(.*)',
            'obj_piezo_step':
                '^Z PZT .* Interval \(um\), .* :\t\d+\.*\d*\t(\d+\.*\d*)\t\d+$',
            'sample_piezo_step':
                '^S PZT .* Interval \(um\), .* :\t\d+\.*\d*\t(\d+\.*\d*)\t\d+$',
        }

        converter = {
            'type': ScanType
        }

        parsed = AttrDict()
        for field, pattern in patterns.items():
            # NOTE exception here, multiple channels may exist
            value = re.findall(pattern, lines, re.MULTILINE)[0]
            try:
                value = converter[field](value)
            except:
                # no need to convert
                pass
            parsed[field] = value

        # NOTE exception, deal with multi-channel
        values = re.findall(
            '^Excitation Filter, Laser, Power \(%\), Exp\(ms\) \((?P<channel>\d+)\) :\t(?P<filter>\D+)\t(?P<wavelength>\d+)\t(?P<power>\d+)\t(?P<exposure>\d+.\d+)',
            lines,
            re.MULTILINE
        )
        parsed['channels'] = values

        return 'waveform', parsed

    @staticmethod
    def parse_camera(lines):
        patterns = {
            'model': '^Model :\t(.*)',
            'exposure': '^Exp\(s\)\D+([\d\.]+)',
            'cycle': '^Cycle\(s\)\D+([\d\.]+)',
            'roi': '^ROI :\tLeft=(\d+) Top=(\d+) Right=(\d+) Bot=(\d+)'
        }

        converter = {
            'exposure': lambda x: float(x),
            'cycle': lambda x: float(x),
            'roi': lambda x: tuple([int(i) for i in x])
        }

        parsed = AttrDict()
        for field, pattern in patterns.items():
            value = re.findall(pattern, lines, re.MULTILINE)[0]
            try:
                value = converter[field](value)
            except:
                # no need to convert
                pass
            parsed[field] = value

        return 'camera', parsed

    @staticmethod
    def parse_timing(lines):
        patterns = {
            'mode': '^Trigger Mode :\t(.*)'
        }

        converter = {
            'mode': TriggerMode
        }

        parsed = AttrDict()
        for field, pattern in patterns.items():
            value = re.findall(pattern, lines, re.MULTILINE)[0]
            try:
                value = converter[field](value)
            except:
                # no need to convert
                pass
            parsed[field] = value

        return 'timing', parsed

class HardwareSettings(object):
    def __init__(self, lines):
        pass
