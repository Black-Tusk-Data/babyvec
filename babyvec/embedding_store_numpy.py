import logging
import os
import shutil

import hickle as hkl
import numpy as np

from babyvec.abstract_embedding_store import AbstractEmbeddingStore
from babyvec.common import FileRef
from babyvec.models import Embedding


INIT_SIZE = 5


class EmbeddingStoreNumpy(AbstractEmbeddingStore):
    def __init__(
            self,
            *,
            embedding_size: int,
            embed_table_fref: FileRef,
            text_table: dict[str, int] | None = None
    ):
        self.embed_table_fref = embed_table_fref
        self.insert_idx = 1
        self.text_table = text_table or {}
        if not os.path.exists(self.embed_table_fref.abspath):
            if not os.path.exists(self.embed_table_fref.absdir):
                os.makedirs(self.embed_table_fref.absdir)
            np.memmap(
                self.embed_table_fref.abspath,
                dtype='float32',
                mode='w+',
                shape=(INIT_SIZE, embedding_size)
            ).flush()
        self.embed_table = np.memmap(
            self.embed_table_fref.abspath,
            dtype='float32',
            mode='r+',
            shape=(max(INIT_SIZE, len(self.text_table) + 1), embedding_size)
        )
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
            logging.debug("resizing...")
            new_fref = FileRef.parse(
                os.path.join(
                    self.embed_table_fref.absdir,
                    "bigger.dat",
                )
            )
            new_table = np.memmap(
                new_fref.abspath,
                dtype='float32',
                mode='w+',
                shape=(
                    2 * self.embed_table.shape[0],
                    self.embed_table.shape[1],
                )
            )
            new_table.flush()
            shutil.copy(
                new_fref.abspath,
                self.embed_table_fref.abspath,
            )
            self.embed_table = np.memmap(
                self.embed_table_fref.abspath,
                dtype='float32',
                mode='r+',
                shape=new_table.shape,
            )
            os.remove(new_fref.abspath)

        self.text_table[text] = self.insert_idx
        self.embed_table[self.insert_idx] = embedding
        self.insert_idx += 1
        return

    def persist_to_disk(self, storage_path: FileRef) -> None:
        if not os.path.exists(storage_path.absdir):
            os.makedirs(storage_path.absdir)
        self.embed_table.flush()
        hkl.dump(self.text_table, os.path.join(
            storage_path.absdir,
            "text_table.hkl",
        ))
        return

    @staticmethod
    def initialize(*, embedding_size: int, embed_table_fref: FileRef) -> "EmbeddingStoreNumpy":
        return EmbeddingStoreNumpy(embedding_size=embedding_size, embed_table_fref=embed_table_fref)

    @staticmethod
    def load_from_disk(
            embed_table_fref: FileRef,
            embedding_size: int,
    ) -> "EmbeddingStoreNumpy":
        text_table = hkl.load(os.path.join(
            embed_table_fref.absdir,
            "text_table.hkl",
        ))
        return EmbeddingStoreNumpy(
            embedding_size=embedding_size,
            embed_table_fref=embed_table_fref,
            text_table=text_table,
        )
