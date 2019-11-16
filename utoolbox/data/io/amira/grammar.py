from pprint import pprint

from pyparsing import (
    Or,
    Group,
    CaselessKeyword,
    alphas,
    nestedExpr,
    nums,
    Word,
    OneOrMore,
    QuotedString,
    Optional,
    Suppress,
    Literal,
)

object_type = (
    CaselessKeyword("lattice")
    | CaselessKeyword("vertex")
    | CaselessKeyword("edge")
    | CaselessKeyword("point")
)
object_definition = Group(CaselessKeyword("define") + object_type + Word(nums))

parameter_keyword = CaselessKeyword("ContentType") | CaselessKeyword("MinMax")
single_value = parameter_keyword + QuotedString('"', '"')
numeric_array = parameter_keyword + Group(OneOrMore(Word(nums)))
unknown = Word(alphas + "_") + nestedExpr("{", "}")
parameter_definition = Group(single_value | numeric_array | unknown) + Suppress(
    Optional(",")
)
parameters = Group(
    CaselessKeyword("Parameters") + nestedExpr("{", "}", parameter_definition)
)

comment = Suppress(CaselessKeyword("#"))

header_flag = CaselessKeyword("AmiraMesh")
file_type = CaselessKeyword("BINARY-LITTLE-ENDIAN") | CaselessKeyword("3D ASCII")
file_type_version = Word(nums + ".")
file_format = Group(comment + header_flag + file_type + file_type_version)

data_type = CaselessKeyword("float") | CaselessKeyword("int")
array = data_type + nestedExpr("[", "]", Word(nums))
single = data_type
data_section_flag = Suppress(Literal("@")) + nums
data_shape = Group((array | single) + Word(alphas))
data_definition = Group(object_type + nestedExpr("{", "}", data_shape))

data = r"""
# AmiraMesh BINARY-LITTLE-ENDIAN 3.0

define VERTEX 748
define EDGE 832
define POINT 12870

Parameters {
    _symbols {
    }
    HistoryLogHead {
        HistoryLog {
            UID:2282e508-4780-4f85-a2f3-b01f90056530 {
            }
        }
    }
    ContentType "Colormap",
    MinMax 0 255
}

Points { float[3] Coordinates } @1
Points { int Ids } @2
"""


def test():
    result = (
        OneOrMore(file_format | object_definition | parameters | data_definition)
        .parseString(data)
        .asList()
    )

    pprint(result)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    test()
