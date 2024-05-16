import abc

import numpy.typing as npt

from babyvec.models import (
    Embedding,
    EmbeddingId,
    EmbeddingScalarType,
    PersistenceOptions,
)
from babyvec.store.abstract_metadata_store import AbstractMetadataStore


class AbstractEmbeddingStore(abc.ABC):
    embed_table: npt.NDArray[EmbeddingScalarType]

    def __init__(
        self,
        *,
        metadata_store: AbstractMetadataStore,
        persist_options: PersistenceOptions,
    ):
        self.metadata_store = metadata_store
        self.persist_options = persist_options
        return

    @abc.abstractmethod
    def get(self, text: str) -> Embedding | None:
        pass

    @abc.abstractmethod
    def put(self, *, text: str, embedding: Embedding) -> EmbeddingId:
        pass

    @abc.abstractmethod
    def put_many(
        self, *, texts: list[str], embeddings: list[Embedding]
    ) -> list[EmbeddingId]:
        pass

    @abc.abstractmethod
    def delete_embeddings(self, embedding_ids: list[EmbeddingId]) -> None:
        pass

    pass
