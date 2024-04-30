import abc
from typing import Iterable


class AbstractChunker(abc.ABC):
    @abc.abstractmethod
    def chunkify_document(self, document: str) -> Iterable[str]:
        pass
