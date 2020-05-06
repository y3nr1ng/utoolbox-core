from typing import Iterable, Union, List

from .generic import BaseDataset

__all__ = ["DatasetIterator"]


class DatasetIterator:
    """
    Iterate over a specific index. The return value is another MultiIndex 
    dataframe, allowing one to chain different iterators.

    Args:
        ds (BaseDataset): dataset to iterate upon
        index (str or list of str): the index to iterate
        ascending (bool, optional): False to sort in descending order
        return_key (bool, optional): True to return the index and the value
        strict (bool, optional): iterator must able to support this dataset
    """

    def __init__(
        self,
        dataset: BaseDataset,
        index: Union[str, Iterable[str]],
        ascending: bool = True,
        return_key: bool = True,
        strict: bool = False,
    ):
        self.dataset = dataset

        if isinstance(index, str):
            self._index = index
        else:
            self._index = list(index)  # ensure it is a list
        self._ascending = ascending
        self._return_key = return_key

        self._strict = strict

    def __iter__(self):
        print(
            f">> {self.__class__.__name__}.__iter__, type(dataset)={type(self.dataset).__name__}"
        )

        original_inventory = self.dataset.inventory.copy()

        try:
            dataset = self.dataset.inventory.sort_index(
                axis="index",
                level=self.index,
                ascending=self.ascending,
                sort_remaining=False,
            )
            iterator = dataset.groupby(self.index)
        except (KeyError, IndexError):
            # not a supported dataset
            #   - KeyError: inventory does not has this column
            #   - IndexError: unusable index, such as empty list
            if self.strict:
                raise ValueError("iterator does not support this dataset")
            else:
                iterator = [(None, self.dataset)]

        # use original dataset as skeleton, iterate over stuff
        modified = False
        for key, selected in iterator:
            # TODO refactor here
            # TODO if selected only has 1 row, return dataset[selected] instead of wrapped dataset
            if self.return_key:
                if key:
                    self.dataset.inventory = selected
                    modified = True
                yield key, self.dataset
            else:
                yield self.dataset

        # restore
        if modified:
            self.dataset.inventory = original_inventory

    ##

    @property
    def ascending(self) -> bool:
        return self._ascending

    @property
    def index(self) -> Union[str, List[str]]:
        return self._index

    @property
    def return_key(self) -> bool:
        return self._return_key

    @property
    def strict(self) -> bool:
        return self._strict
