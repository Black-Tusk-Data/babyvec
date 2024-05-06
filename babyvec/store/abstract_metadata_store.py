import abc
import os

from babyvec.models import EmbeddingId, PersistenceOptions


class AbstractMetadataStore(abc.ABC):

    def __init__(
        self,
        options: PersistenceOptions,
    ):
        self.options = options
        self.persist_dir = options.persist_dir
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
        return

    @abc.abstractmethod
    def add_text(self, *, text: str, embedding_id: EmbeddingId, metadata: dict) -> None:
        pass

    @abc.abstractmethod
    def get_embedding_id(self, text: str) -> EmbeddingId | None:
        pass

    @abc.abstractmethod
    def get_embedding_text_and_metadata(
        self, embedding_id: EmbeddingId
    ) -> tuple[str, dict]:
        pass
