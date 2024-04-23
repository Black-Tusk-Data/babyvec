import abc

from babyvec.models import Embedding


class AbstractEmbeddingStore(abc.ABC):
    @abc.abstractmethod
    def get(self, text: str) -> Embedding | None:
        pass

    @abc.abstractmethod
    def put(self, text: str, embedding: Embedding) -> None:
        pass
