import os

from babyvec.models import Embedding, EmbeddingScalarType
from babyvec.store.abstract_embedding_store import (
    AbstractEmbeddingStore,
    EmbeddingPersistenceOptions,
)
from npy_append_array import NpyAppendArray  # type: ignore
import numpy as np
import numpy.typing as npt


EMBED_TABLE_FNAME = "embed-table.npy"


class EmbeddingStoreNumpy(AbstractEmbeddingStore):
    def __init__(
        self,
        options: EmbeddingPersistenceOptions,
    ):
        super().__init__(options)
        self.embed_table_path = os.path.join(
            self.persist_dir,
            EMBED_TABLE_FNAME,
        )

        self.embed_table: npt.NDArray[EmbeddingScalarType]
        if os.path.exists(self.embed_table_path):
            self.embed_table = np.load(self.embed_table_path, mmap_mode="r")
        else:
            self.embed_table = np.array([])
        return

    def get(self, text: str) -> Embedding | None:
        embed_id = self.metadata_store.get_embedding_id(text)
        if embed_id is None:
            return None
        return self.embed_table[embed_id]

    def put(self, *, text: str, embedding: Embedding) -> None:
        self.put_many(
            texts=[text],
            embeddings=[embedding],
        )
        return

    def put_many(
        self,
        *,
        texts: list[str],
        embeddings: list[Embedding],
    ) -> None:
        assert len(texts) == len(embeddings)
        if not texts:
            return
        existing_embed_ids = [
            self.metadata_store.get_embedding_id(text) for text in texts
        ]

        insert_offset = len(self.embed_table)

        missing_indices = [
            i for i in range(len(existing_embed_ids)) if existing_embed_ids[i] is None
        ]
        new_texts: list[str] = []
        new_embeddings = np.empty(
            (
                len(missing_indices),
                len(embeddings[0]),
            )
        )
        for i, idx in enumerate(missing_indices):
            new_embeddings[i] = embeddings[idx]
            new_texts.append(texts[idx])
        pass

        # Note!  We do not support adjusting an existing embedding!
        # If we want to do this, need to look at loading the mmap in write mode.
        with NpyAppendArray(self.embed_table_path, delete_if_exists=False) as npaa:
            npaa.append(new_embeddings)
            for i, text in enumerate(new_texts):
                self.metadata_store.add_text_embedding(
                    text=text,
                    embedding_id=insert_offset + i,
                )

        self.embed_table = np.load(
            self.embed_table_path,
            mmap_mode="r",
        )
        return
