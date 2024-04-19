import os

from babyvec.common import FileRef
from babyvec.embedding_store import EmbeddingStore
from babyvec.local_embedder import LocalEmbedder


class BabyVecLocalEmbedder:
    def __init__(
            self,
            *,
            persist_path: str,
            embedding_size: int,
            model: str,
            device: str,
    ):
        persist_fref = FileRef.parse(persist_path)
        if not os.path.exists(persist_fref.abspath):
           EmbeddingStore.initialize(
               embedding_size=embedding_size,
           ).persist_to_disk(persist_fref)
        self.store = EmbeddingStore.load_from_disk(storage_path=persist_fref)
        self.embedder = LocalEmbedder(
            store=self.store,
            model=model,
            device=device,
        )
        return
