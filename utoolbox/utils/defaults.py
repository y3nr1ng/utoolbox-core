"""
Provides all sorts of default parameters in different scenarios.
"""

class DefaultFormat(dict):
    def __missing__(self, key):
        return '{' + key + '}'
