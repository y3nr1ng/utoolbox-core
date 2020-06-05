import logging

from tqdm import tqdm

__all__ = ["TqdmLoggingHandler", "change_logging_level"]


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


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
