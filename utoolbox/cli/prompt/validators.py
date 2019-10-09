import re

from prompt_toolkit.validation import Validator, ValidationError


class NumberValidator(Validator):
    pattern = re.compile(r"\d+(\.\d+)?")

    def __init__(self, message=None):
        self.message = message if message else "input contains non-numeric characters"

    def validate(self, document):
        text = document.text

        if text and (NumberValidator.pattern.match(text) is None):
            i = 0

            for i, c in enumerate(text):
                if not c.isdigit():
                    break
            raise ValidationError(message=self.message, cursor_position=i)
