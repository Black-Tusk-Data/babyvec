from babyvec.computer.embedding_computer_jina_bert import EmbeddingComputerJinaBert
from babyvec.embed_provider.cached_embed_provider import CachedEmbedProvider
from babyvec.models import *
from babyvec.store.embedding_store_numpy import EmbeddingStoreNumpy


# 'packaged' providers
class CachedParallelJinaEmbedder(CachedEmbedProvider):
    def __init__(
            self,
            persist_dir: str,
            n_computers: int,
            device: str,
    ):
        computer = EmbeddingComputerJinaBert(
            compute_options=EmbedComputeOptions(
                device=device,
                n_computers=n_computers,
            )
        )
        store = EmbeddingStoreNumpy(persist_dir=persist_dir)
        super().__init__(
            computer=computer,
            store=store
        )
