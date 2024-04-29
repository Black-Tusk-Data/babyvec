import abc

from babyvec.computer.abstract_embedding_computer import AbstractEmbeddingComputer
from babyvec.index.abstract_index import AbstractIndex
from babyvec.store.abstract_embedding_store import AbstractEmbeddingStore


class AbstractIndexFactory(abc.ABC):
    def __init__(
        self,
        *,
        store: AbstractEmbeddingStore,
        computer: AbstractEmbeddingComputer,
    ):
        self.store = store
        self.computer = computer
        return

    @abc.abstractmethod
    def build_index(self) -> AbstractIndex:
        pass
