import logging


__all__ = ["change_logging_level"]


class change_logging_level:
    """Temporarily change the logging level of a command block."""

    def __init__(self, level, logger=None):
        self._target_level = level
        self._logger = logger if logger else logging.getLogger()

    def __enter__(self):
        self._original_level = self._logger.getEffectiveLevel()
        self.logger.setLevel(self._target_level)
        return self

    def __exit__(self, *exc):
        self.logger.setLevel(self._original_level)

    ##

    @property
    def logger(self):
        return self._logger
