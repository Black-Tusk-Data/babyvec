import abc

from babyvec.models import Embedding, EmbeddingId


class AbstractEmbeddingStore(abc.ABC):
    @abc.abstractmethod
    def get(self, text: str) -> Embedding | None:
        pass

    @abc.abstractmethod
    def put(self, text: str, embedding: Embedding) -> None:
        pass

    @abc.abstractmethod
    def get_text_map(self) -> dict[str, EmbeddingId]:
        pass
