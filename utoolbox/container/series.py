import logging
logger = logging.getLogger(__name__)

import pandas as pd

from .registry import BaseContainer

class TimeSeries(BaseContainer):
    """
    Container for a time series.

    Parameters
    ----------
    ctype : utoolbox.container.BaseContainer
        Container type for each time point.
    """
    @property
    def _constructor(self):
        return TimeSeries

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError

    def __init__(self, ctype, source=None, **kwargs):
        if isinstance(source, list):
            pass

        super(TimeSeries, self).__init__(*args, **args)
        self.metadata.ctype = ctype
