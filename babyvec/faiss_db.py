from babyvec.computer.abstract_embedding_computer import AbstractEmbeddingComputer
from babyvec.computer.embedding_computer_jina_bert import EmbeddingComputerJinaBert
from babyvec.embed_provider.abstract_embed_provider import AbstractEmbedProvider
from babyvec.embed_provider.cached_embed_provider import CachedEmbedProvider
from babyvec.embed_provider.parallelized_cached_embed_provider import (
    ParallelizedCachedEmbedProvider,
)
from babyvec.index.faiss_index import FaissIndex
from babyvec.models import (
    CorpusFragment,
    DbSearchResult,
    EmbedComputeOptions,
    PersistenceOptions,
)
from babyvec.store.abstract_embedding_store import (
    AbstractEmbeddingStore,
    EmbeddingPersistenceOptions,
)
from babyvec.store.abstract_metadata_store import AbstractMetadataStore
from babyvec.store.embedding_store_numpy import EmbeddingStoreNumpy
from babyvec.store.metadata_store_sqlite import MetadataStoreSQLite


class FaissDb:
    def __init__(
        self,
        *,
        persist_dir: str,
        device: str,
        n_computers: int = 1,
        computer_type: type[AbstractEmbeddingComputer] = EmbeddingComputerJinaBert,
        metadata_store_type: type[AbstractMetadataStore] = MetadataStoreSQLite,
        embed_store_type: type[EmbeddingStoreNumpy] = EmbeddingStoreNumpy,
    ):
        self.persist_options = EmbeddingPersistenceOptions(
            persist_options=PersistenceOptions(persist_dir=persist_dir),
            metadata_store_type=metadata_store_type,
        )
        self.compute_options = EmbedComputeOptions(
            device=device,
        )
        self.embed_store = embed_store_type(self.persist_options)
        # TODO: this is messy
        self.embedding_provider = ParallelizedCachedEmbedProvider(
            n_computers=n_computers,
            compute_options=self.compute_options,
            computer_type=computer_type,
            store=self.embed_store,
        )
        self.index: FaissIndex | None = None
        return

    def ingest_fragments(self, fragments: list[CorpusFragment]) -> None:
        # TODO: The following is hackish and indicates a weakness of the current abstractions.
        #       Embedding provider should expose 'persist_embeddings', returning the embed IDs
        self.embedding_provider.get_embeddings(
            [fragment.text for fragment in fragments]
        )
        for fragment in fragments:
            self.embed_store.metadata_store.add_fragment(
                embedding_id=self.embed_store.metadata_store.get_embedding_id(fragment.text),  # type: ignore
                fragment=fragment,
            )
        return

    def index_existing_fragments(self) -> None:
        self.index = FaissIndex(
            computer=self.embedding_provider.computer,
            vectors=self.embed_store.embed_table,
        )
        return

    def search(self, query: str, k_nearest: int) -> list[DbSearchResult]:
        assert self.index
        index_hits = self.index.search(query, k_nearest)
        results: list[DbSearchResult] = []
        for hit in index_hits:
            for fragment in self.embed_store.metadata_store.get_fragments_for_embedding(
                hit.embedding_id
            ):
                results.append(
                    DbSearchResult(
                        index_search_result=hit,
                        fragment=fragment,
                    )
                )
                pass
            pass
        return results

    def shutdown(self):
        self.embedding_provider.shutdown()
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return

    pass
