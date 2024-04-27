import os
import pickle
import shelve

from babyvec.models import Embedding, EmbeddingId
from babyvec.store.abstract_embedding_store import AbstractEmbeddingStore
from npy_append_array import NpyAppendArray
import numpy as np
import numpy.typing as npt


TEXT_MAP_FNAME = "text-map.dat"
EMBED_TABLE_FNAME = "embed-table.npy"


class EmbeddingStoreNumpy(AbstractEmbeddingStore):
    def __init__(
            self,
            *,
            persist_dir: str
    ):
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        self.text_map_path = os.path.join(
            persist_dir,
            TEXT_MAP_FNAME,
        )
        self.embed_table_path = os.path.join(
            persist_dir,
            EMBED_TABLE_FNAME,
        )
        self.text_map = shelve.open(self.text_map_path)
        self.embed_table: npt.ArrayLike
        if os.path.exists(self.embed_table_path):
            self.embed_table = np.load(self.embed_table_path, mmap_mode="r")
        else:
            self.embed_table = np.array([])
        return

    def get(self, text: str) -> Embedding | None:
        idx = self.text_map.get(text)
        if idx is None:
            return None
        return self.embed_table[idx]

    def put(self, text: str, embedding: Embedding) -> None:
        self.put_many([text], [embedding])
        return

    def put_many(self, texts: list[str], embeddings: list[Embedding]) -> None:
        assert len(texts) == len(embeddings)
        if not texts:
            return
        existing = [
            self.text_map.get(text)
            for text in texts
        ]

        insert_offset = len(self.embed_table)

        missing_indices = [
            i for i in range(len(existing)) if existing[i] is None
        ]
        new_texts: list[str] = []
        new_embeddings = np.empty((
            len(missing_indices),
            len(embeddings[0]),
        ))
        for i, idx in enumerate(missing_indices):
            new_texts.append(texts[idx])
            new_embeddings[i] = embeddings[idx]

        # Note!  We do not support adjusting an existing embedding!
        # If we want to do this, need to look at loading the mmap in write mode.
        with NpyAppendArray(self.embed_table_path, delete_if_exists=False) as npaa:
            npaa.append(new_embeddings)
            for i, text in enumerate(new_texts):
                self.text_map[text] = insert_offset + i

        self.embed_table = np.load(
            self.embed_table_path,
            mmap_mode="r",
        )
        self.text_map.close()
        self.text_map = shelve.open(self.text_map_path)
        return

    def get_text_map(self) -> dict[str, EmbeddingId]:
        return dict(self.text_map)
