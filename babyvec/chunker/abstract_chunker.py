import abc
from typing import Iterable


class AbstractChunker(abc.ABC):
    def __init__(
        self,
        delimiter: str = " ",
    ):
        self.delimiter = delimiter
        return

    @abc.abstractmethod
    def chunkify_document(self, document: str) -> Iterable[str]:
        pass
