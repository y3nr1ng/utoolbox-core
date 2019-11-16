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
parameter_definition = Group(single_value | numeric_array) + Optional(",")
parameters = Group(
    CaselessKeyword("Parameters") + nestedExpr("{", "}", parameter_definition)
)


data = r"""
define VERTEX 748
define EDGE 832
define POINT 12870

Parameters {
    ContentType "Colormap",
    MinMax 0 255
}

"""


def test():

    result = OneOrMore(object_definition | parameters).parseString(data).asList()

    pprint(result)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    test()
