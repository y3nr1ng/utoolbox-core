from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

__all__ = ["prompt_options"]


def prompt_options(message, options, show_options=True):
    # convert options to string
    options[:] = [str(option) for option in options]

    if show_options:
        print()
        for option in options:
            print(f".. {option}")

    # ask
    completer = WordCompleter(options)
    return prompt(message, completer=completer)
