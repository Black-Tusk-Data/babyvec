import logging
import os

import hickle as hkl
import numpy as np

from babyvec.abstract_embedding_store import AbstractEmbeddingStore
from babyvec.common import FileRef
from babyvec.models import Embedding


INIT_SIZE = 5000


class EmbeddingStoreNumpy(AbstractEmbeddingStore):
    def __init__(
            self,
            embedding_size: int,
    ):
        self.insert_idx = 1
        self.text_table: dict[str, int] = {}
        self.embed_table = np.empty((INIT_SIZE, embedding_size))
        return

    def get(self, text: str) -> Embedding | None:
        idx = self.text_table.get(text, None)
        if idx is None:
            return None
        return self.embed_table[idx]

    def put(self, text: str, embedding: Embedding) -> None:
        if text in self.text_table:
            idx = self.text_table.pop(text)
            self.embed_table[idx] = embedding
            return
        if self.insert_idx >= self.embed_table.shape[0]:
            self.embed_table = np.resize(
                self.embed_table,
                (self.embed_table.shape[0] * 2, self.embed_table.shape[1])
            )
        self.text_table[text] = self.insert_idx
        self.embed_table[self.insert_idx] = embedding
        self.insert_idx += 1
        return

    def persist_to_disk(self, storage_path: FileRef) -> None:
        if not os.path.exists(storage_path.absdir):
            os.makedirs(storage_path.absdir)
        hkl.dump(self.embed_table, os.path.join(
            storage_path.absdir,
            "embeddings.hkl",
        ))
        hkl.dump(self.text_table, os.path.join(
            storage_path.absdir,
            "text_table.hkl",
        ))
        return

    @staticmethod
    def initialize(*, embedding_size: int) -> "EmbeddingStoreNumpy":
        return EmbeddingStoreNumpy(embedding_size=embedding_size)

    @staticmethod
    def load_from_disk(storage_path: FileRef) -> "EmbeddingStoreNumpy":
        embed_table = hkl.load(os.path.join(
            storage_path.absdir,
            "embeddings.hkl",
        ))
        store = EmbeddingStoreNumpy(embedding_size=embed_table.shape[1])
        store.embed_table = embed_table
        store.text_table = hkl.load(os.path.join(
            storage_path.absdir,
            "text_table.hkl",
        ))
        return store
