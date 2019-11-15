from lark import Lark

am_grammar = r"""
?start: section

?section: CNAME content

?content: CNAME value
        | "{" [section*] "}"

?value: NUMBER+

%import common (WS, NUMBER, CNAME)
%ignore WS
"""


def test():
    parser = Lark(am_grammar, parser="lalr")

    lines = r"""
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
