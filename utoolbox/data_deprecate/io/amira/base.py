import logging

import numpy as np
from pyparsing import (
    CaselessKeyword,
    Combine,
    Dict,
    Group,
    Literal,
    MatchFirst,
    OneOrMore,
    Optional,
    QuotedString,
    Word,
    alphanums,
    nestedExpr,
    restOfLine,
)
from pyparsing import pyparsing_common as pc

__all__ = ["Amira"]

logger = logging.getLogger(__name__)


# comment
# .. file format
file_format_options = CaselessKeyword("BINARY-LITTLE-ENDIAN") | CaselessKeyword(
    "3D ASCII"
).setResultsName("format")
file_format_version = pc.real.setResultsName("version")
file_format = Group(
    CaselessKeyword("AmiraMesh").suppress() + file_format_options + file_format_version
).setResultsName("file_format")
# .. comment
comment_string = restOfLine
comment = Group(Word("#", min=1).suppress() + MatchFirst([file_format, comment_string]))


# declaration
object_type = (
    CaselessKeyword("lattice")
    | CaselessKeyword("vertex")
    | CaselessKeyword("edge")
    | CaselessKeyword("point")
    | CaselessKeyword("points")
).setResultsName("object_type")
object_size = pc.integer.setResultsName("size")
declaration = Group(CaselessKeyword("define").suppress() + object_type + object_size)

# parameter
key = CaselessKeyword("ContentType") | CaselessKeyword("MinMax")
value = (
    QuotedString('"', '"')
    | Group(OneOrMore(pc.integer | pc.real))
    | nestedExpr("{", "}")
)
parameter = Dict(
    Group(key + value + Optional(",").suppress())
    | Group(Word("_" + alphanums) + value + Optional(",")).suppress()
)
parameters = CaselessKeyword("Parameters").suppress() + nestedExpr(
    "{", "}", parameter
).setResultsName("parameters")

# prototype
element_type = (CaselessKeyword("float") | CaselessKeyword("int")).setResultsName(
    "type"
)
# .. array
element_counts = pc.integer.setResultsName("counts")
# .. structure
element_name = Word(alphanums).setResultsName("name")
element = (
    element_type
    + Optional(Word("[").suppress() + element_counts + Word("]").suppress())
    + element_name
)
section_id = Combine(Literal("@") + pc.integer).setResultsName("section_id")
# .. prototype
prototype = Group(
    object_type + nestedExpr("{", "}", element).setResultsName("data_type") + section_id
)

grammar = OneOrMore(
    comment.setResultsName("comments", listAllMatches=True)
    | declaration.setResultsName("declarations", listAllMatches=True)
    | parameters
    | prototype.setResultsName("prototypes", listAllMatches=True)
)


class Amira(object):
    def __init__(self, path):
        self._path = path

        self._metadata = self._parse_metadata()
        self._validate_file()

        self._data = self._parse_data_prototype()

    ##

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata

    @property
    def path(self):
        return self._path

    ##

    def _parse_metadata(self):
        lines = []
        with open(self.path, "r", errors="ignore") as fd:
            for line in fd:
                if line.startswith("# Data section follows"):
                    break
                lines.append(line)
        lines = "".join(lines)

        metadata = grammar.parseString(lines).asDict()

        # simplify blocks nested 1 level deeper in a list
        # TODO how to fix this in grammar?
        def extract(d, key):
            d[key] = d[key][0]

        extract(metadata, "parameters")
        for prototype in metadata["prototypes"]:
            extract(prototype, "data_type")

        return metadata

    def _validate_file(self):
        for comment in self.metadata["comments"]:
            if "file_format" in comment:
                break
        else:
            raise RuntimeError("not an Amira-generated file")

    def _parse_data_prototype(self):
        """Parse data block info, but NOT loaded yet."""
        types = self._parse_object_types()

        data = dict()
        for source in self.metadata["prototypes"]:
            name = source["data_type"]["name"]
            if name in data:
                raise RuntimeError(f'data block "{name}" already exists')
            size = types[source["object_type"]]
            sid = source["section_id"]

            shape, dtype = self._interpret_data_layout(size, source["data_type"])
            data[name] = (sid, shape, dtype)
        return data

    def _parse_object_types(self):
        types = dict()
        for source in self.metadata["declarations"]:
            name, size = source["object_type"], source["size"]
            if name in types:
                raise RuntimeError(f'malformed declaration, "{name}" already exists')
            types[name] = size
        return types

    @classmethod
    def _interpret_data_layout(cls, size, element):
        dtype = {"float": np.float32, "int": np.int32}[element["type"]]

        # element
        shape = (element.get("counts", 1),)
        # overall
        shape = (size,) + shape

        return shape, dtype


if __name__ == "__main__":
    from pprint import pprint
    import logging

    logging.basicConfig(level=logging.DEBUG)

    files = ["pureGreen.col", "c6_rawpoints_0042.am", "c6_spatialgraph_0042.am"]
    for path in files:
        print(path)
        am = Amira(path)
        pprint(am.metadata)
        pprint(am._data)
        print()
