from babyvec.computer.abstract_embedding_computer import AbstractEmbeddingComputer
from babyvec.computer.embedding_computer_jina_bert import EmbeddingComputerJinaBert
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
        embed_store_type: type[AbstractEmbeddingStore] = EmbeddingStoreNumpy,
    ):
        self.persist_options = PersistenceOptions(persist_dir=persist_dir)
        self.compute_options = EmbedComputeOptions(
            device=device,
        )
        self.metadata_store = metadata_store_type(self.persist_options)
        self.embed_store = embed_store_type(
            metadata_store=self.metadata_store,
            persist_options=self.persist_options,
        )
        self.embedding_provider = ParallelizedCachedEmbedProvider(
            n_computers=n_computers,
            compute_options=self.compute_options,
            computer_type=computer_type,
            store=self.embed_store,
        )
        self.index: FaissIndex | None = None
        return

    def ingest_fragments(self, fragments: list[CorpusFragment]) -> None:
        embedding_ids = self.embedding_provider.persist_embeddings(
            [fragment.text for fragment in fragments]
        )

        for embed_id, fragment in zip(embedding_ids, fragments):
            self.embed_store.metadata_store.ingest_fragment(
                embedding_id=embed_id,
                fragment=fragment,
            )
            pass
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
