import logging
import re

import numpy as np

__all__ = ["Amira"]

logger = logging.getLogger(__name__)


class Amira(object):
    def __init__(self, path):
        self._path = path

        with open(path, "r") as fd:
            file_def = fd.readline()

            if not self.is_file_valid(file_def):
                raise TypeError("not an Amira file")
            self._encoding = self.encoding(file_def)

            # scan until start of the data section
            lines = []
            for line in fd:
                if line.startswith("# Data section follows"):
                    break
                else:
                    lines.append(line.strip())
            lines = "".join(lines)

        self._parameters = self.parse_parameters(lines)
        self._data = self.parse_data_definitions(lines)

    ##

    @property
    def data(self):
        return self._data

    @property
    def is_ascii(self):
        return self._encoding == "ascii"

    @property
    def is_binary(self):
        return self._encoding == "binary"

    @property
    def parameters(self):
        return self._parameters

    @property
    def path(self):
        return self._path

    ##

    @classmethod
    def parse_parameters(cls, lines, pattern=r"Parameters {([^{]+)}"):
        parameters = dict()
        lines = re.findall(pattern, lines)[0].split(",")
        for line in lines:
            name, *options = tuple(line.split(" "))
            # force as tuple
            options = tuple(options)
            parameters[name] = {
                "ContentType": cls._parse_content_type,
                "MinMax": cls._parse_min_max,
            }[name](*options)
        return parameters

    @classmethod
    def _parse_content_type(cls, text):
        return text[1:-1]

    @classmethod
    def _parse_min_max(cls, vmin, vmax):
        return float(vmin), float(vmax)

    ##

    @classmethod
    def parse_data_definitions(cls, lines):
        data = dict()
        for name, shape in cls._parse_tag_sections(lines):
            fmt, tag = re.findall(name + " {([^{]+)} (@\d+)", lines)[0]
            fmt = fmt.strip()
            data[name] = (tag, cls._preallocate_memory(shape, fmt))
        return data

    @classmethod
    def _parse_tag_sections(cls, lines, pattern=r"define ([^\s]+) ([\d\s]+)"):
        lines = re.findall(pattern, lines)
        for name, shape in lines:
            yield name, cls._parse_shape(shape)

    @classmethod
    def _parse_shape(cls, shape):
        return tuple(int(s) for s in shape.split(" "))

    @classmethod
    def _preallocate_memory(cls, shape, fmt, pattern=r"(\S+)\[(\d+)\]"):
        dtype, nelem = re.findall(pattern, fmt)[0]
        dtype = {"float": np.float32}[dtype]
        nelem = (int(nelem),)
        return np.empty(shape + nelem, dtype=dtype)

    ##

    @classmethod
    def is_file_valid(cls, line):
        line = line.strip()
        for keyword in ("Amira", "Avizo"):
            if keyword in line:
                return True
        return False

    @classmethod
    def encoding(cls, line):
        if "ASCII" in line:
            return "ascii"
        elif "BINARY" in line:
            return "binary"
        else:
            raise ValueError("unable to determine encoding")

