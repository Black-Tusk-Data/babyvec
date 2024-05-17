import logging
import os
from typing import cast

import numpy as np
import numpy.typing as npt
from npy_append_array import NpyAppendArray  # type: ignore

from babyvec.models import (
    Embedding,
    EmbeddingId,
    EmbeddingScalarType,
    PersistenceOptions,
)
from babyvec.store.abstract_embedding_store import (
    AbstractEmbeddingStore,
)
from babyvec.store.abstract_metadata_store import AbstractMetadataStore

EMBED_TABLE_FNAME = "embed-table.npy"


class EmbeddingStoreNumpy(AbstractEmbeddingStore):
    def __init__(
        self,
        *,
        metadata_store: AbstractMetadataStore,
        persist_options: PersistenceOptions,
    ):
        super().__init__(metadata_store=metadata_store, persist_options=persist_options)

        self.embed_table_path = os.path.join(
            self.persist_options.persist_dir,
            EMBED_TABLE_FNAME,
        )
        self.embed_table: npt.NDArray[EmbeddingScalarType]
        if os.path.exists(self.embed_table_path):
            self.embed_table = np.load(self.embed_table_path, mmap_mode="r")
        else:
            self.embed_table = np.array([])
            pass
        return

    def get(self, text: str) -> Embedding | None:
        embed_id = self.metadata_store.get_embedding_id(text)
        if embed_id is None:
            return None
        return self.embed_table[embed_id]

    def put(self, *, text: str, embedding: Embedding) -> EmbeddingId:
        return self.put_many(
            texts=[text],
            embeddings=[embedding],
        )[0]

    def put_many(
        self,
        *,
        texts: list[str],
        embeddings: list[Embedding],
    ) -> list[EmbeddingId]:
        assert len(texts) == len(embeddings)
        if not texts:
            return []
        embed_ids = [self.metadata_store.get_embedding_id(text) for text in texts]

        insert_offset = len(self.embed_table)

        missing_indices = [i for i in range(len(embed_ids)) if embed_ids[i] is None]
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
            for i, idx in enumerate(missing_indices):
                text = texts[idx]
                embedding_id = insert_offset + i
                embed_ids[idx] = embedding_id
                self.metadata_store.add_text_embedding(
                    text=text,
                    embedding_id=embedding_id,
                )
                pass
            pass

        self.embed_table = np.load(
            self.embed_table_path,
            mmap_mode="r",
        )
        return cast(list[EmbeddingId], embed_ids)

    def delete_embeddings(self, embedding_ids: list[EmbeddingId]) -> None:
        embed_table = np.load(
            self.embed_table_path,
            mmap_mode="c",
        )
        last_embed_id = len(embed_table) - 1

        for embedding_id in reversed(sorted(embedding_ids)):
            if embedding_id > last_embed_id:
                logging.debug("embedding %d does not exist", embedding_id)
                continue
            if embedding_id == last_embed_id:
                last_embed_id -= 1
                continue
            embed_table[embedding_id] = embed_table[last_embed_id]
            self.metadata_store.migrate_embedding_id(
                from_embedding_id=last_embed_id,
                to_embedding_id=embedding_id,
            )
            last_embed_id -= 1
            pass

        # now embed_table is meaningless after the 'last_embed_id'th entry
        np.save(self.embed_table_path, embed_table[: last_embed_id + 1])
        self.embed_table = np.load(
            self.embed_table_path,
            mmap_mode="r",
        )
        return

    pass
