import logging
import os

from babyvec.common import FileRef
from babyvec.embedding_store_numpy import EmbeddingStoreNumpy
from babyvec.local_embedder import LocalEmbedder
from babyvec.models import Embedding


class BabyVecLocalEmbedder:
    def __init__(
            self,
            *,
            persist_path: str,
            embedding_size: int,
            model: str,
            device: str,
    ):
        self.persist_fref = FileRef.parse(persist_path)
        if not os.path.exists(self.persist_fref.absdir):
            logging.debug("no persistent record found...")
            EmbeddingStoreNumpy.initialize(
                embedding_size=embedding_size,
                embed_table_fref=self.persist_fref,
            ).persist_to_disk(self.persist_fref)

        self._store = EmbeddingStoreNumpy.load_from_disk(
            embedding_size=embedding_size,
            embed_table_fref=self.persist_fref,
        )
        self._embedder = LocalEmbedder(
            store=self._store,
            model=model,
            device=device,
        )
        return

    def get_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embedder.get_embeddings(texts)

    def close(self):
        logging.info("writing to disk...")
        self._store.persist_to_disk(self.persist_fref)
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return
