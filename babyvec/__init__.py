from babyvec.computer.embedding_computer_jina_bert import EmbeddingComputerJinaBert
from babyvec.computer.parallelized_embedding_computer import ParallelizedEmbeddingComputer
from babyvec.embed_provider.cached_embed_provider import CachedEmbedProvider
from babyvec.embed_provider.parallelized_cached_embed_provider import ParallelizedCachedEmbedProvider
from babyvec.models import *
from babyvec.store.embedding_store_numpy import EmbeddingStoreNumpy


# 'packaged' providers
class CachedParallelJinaEmbedder(ParallelizedCachedEmbedProvider):
    def __init__(
            self,
            persist_dir: str,
            n_computers: int,
            device: str,
    ):
        store = EmbeddingStoreNumpy(persist_dir=persist_dir)
        super().__init__(
            n_computers=n_computers,
            compute_opttions=EmbedComputeOptions(
                device=device,
            ),
            computer_type=EmbeddingComputerJinaBert,
            store=store,
        )
