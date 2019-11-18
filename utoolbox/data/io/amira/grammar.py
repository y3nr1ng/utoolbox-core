from pprint import pprint

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
object_size = pc.integer.setResultsName("object_size")
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
element_array = Group(
    element_type + Word("[") + element_counts + Word("]")
).setResultsName("array")
# .. single
element_single = element_type
# .. structure
element_name = Word(alphanums).setResultsName("name")
element = (element_single ^ element_array) + element_name
section_id = Combine(Literal("@") + pc.integer).setResultsName("section_id")
# .. prototype
prototype = Group(
    object_type + nestedExpr("{", "}", element).setResultsName("data_type") + section_id
).setDebug()

grammar = OneOrMore(
    comment.setResultsName("comments", listAllMatches=True)
    | declaration.setResultsName("declarations", listAllMatches=True)
    | parameters
    | prototype.setResultsName("prototypes", listAllMatches=True)
)


def test(path):
    # locate metadata section
    with open(path, "r", errors="ignore") as fd:
        lines = []
        for line in fd:
            if line.startswith("# Data section follows"):
                break
            lines.append(line)

    lines = "".join(lines)
    parsed = grammar.parseString(lines)

    return parsed.asDict()


if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level=logging.DEBUG)

    files = ["pureGreen.col", "c6_rawpoints_0042.am", "c6_spatialgraph_0042.am"]
    for path in files:
        print(path)
        result = test(path)
        print(json.dumps(result, indent=2))
        print()
