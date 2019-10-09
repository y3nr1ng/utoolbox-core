from prompt_toolkit import prompt

from utoolbox.cli.prompt.validators import NumberValidator

__all__ = ["prompt_float", "prompt_int", "prompt_str"]


def prompt_float(message, validator=NumberValidator()):
    return _prompt_simple(message, float, validator=validator)


def prompt_int(message, validator=NumberValidator()):
    return _prompt_simple(message, int, validator=validator)


def prompt_str(message, validator=None):
    return _prompt_simple(message, validator=validator)


def _prompt_simple(message, converter=str, validator=None):
    return converter(prompt(message, validator=validator))
