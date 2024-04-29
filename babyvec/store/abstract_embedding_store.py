import abc
import os
from typing import NamedTuple, Type

from babyvec.models import Embedding, EmbeddingId, PersistenceOptions
from babyvec.store.abstract_metadata_store import AbstractMetadataStore


class EmbeddingPersistenceOptions(NamedTuple):
    persist_options: PersistenceOptions
    metadata_store_type: Type[AbstractMetadataStore]


class AbstractEmbeddingStore(abc.ABC):
    def __init__(
        self,
        options: EmbeddingPersistenceOptions,
    ):
        self.persist_dir = options.persist_options.persist_dir
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
        self.metadata_store = options.metadata_store_type(
            options.persist_options,
        )
        return

    @abc.abstractmethod
    def get(self, text: str) -> Embedding | None:
        pass

    @abc.abstractmethod
    def put(
        self, *, text: str, embedding: Embedding, metadata: dict | None = None
    ) -> None:
        pass
