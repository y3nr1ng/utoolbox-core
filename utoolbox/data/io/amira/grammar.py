from lark import Lark

am_grammar = r"""
?start: format
        | define

?format: FORMAT_KW FORMAT FLOAT 
FORMAT_KW.2: "# AmiraMesh"
FORMAT: "3D ASCII" | "BINARY-LITTLE-ENDIAN"

?define: DEFINE_KW CLASS INT 
DEFINE_KW: "define"i
CLASS: "lattice"i | "vertex"i | "edge"i | "point"i

COMMENT: "#" /.*/

%import common (WS, INT, FLOAT)
%ignore WS
%ignore COMMENT
"""


def test():
    parser = Lark(am_grammar, parser="lalr")

    lines = r"""
# AmiraMesh 3D ASCII 2.0
# CreationDate: Mon May  8 14:12:39 2000

define VERTEX 748
define EDGE 832
define POINT 12870
    """

    print("===")
    for i, line in enumerate(lines.split("\n")):
        print(f"{i:03d}\t{line}")
    print("===")

    result = parser.parse(lines)

    print(result.pretty())


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    test()
