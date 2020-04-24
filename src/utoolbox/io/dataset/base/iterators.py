from typing import Iterable, Union

from .generic import BaseDataset

__all__ = ["DatasetIterator"]


class DatasetIterator:
    """
    Iterate over a specific index.

    Args:
        ds (BaseDataset): dataset to iterate upon
        index (str or list of str): the index to iterate
        ascending (bool, optional): False to sort in descending order
        return_key (bool, optional): True to return the index and the value
    """

    def __init__(
        self,
        dataset: BaseDataset,
        index: Union[str, Iterable[str]],
        ascending: bool = True,
        return_key: bool = True,
    ):
        self.dataset = dataset

        if isinstance(index, str):
            self._index = index
            self._iter_func = self._single_iter
        else:
            self._index = list(index)
            self._iter_func = self._nested_iter
        self._ascending = ascending
        self._return_key = return_key

    def __iter__(self):
        yield self._iter_func()

    def _nested_iter(self):
        raise NotImplementedError  # TODO

    def _single_iter(self):
        dataset = self.dataset.sort_index(
            axis="index",
            level=self.index,
            ascending=self.ascending,
            sort_remaining=False,
        )
        for key, selected in dataset.groupby(self.index):
            if self.return_key:
                yield key, selected
            else:
                yield selected

    ##

    @property
    def ascending(self) -> bool:
        return self._ascending

    @property
    def index(self):
        return self._index

    @property
    def return_key(self) -> bool:
        return self._return_key
