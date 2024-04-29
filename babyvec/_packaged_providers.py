from torch import index_reduce
from babyvec.computer.embedding_computer_jina_bert import EmbeddingComputerJinaBert
from babyvec.embed_provider.parallelized_cached_embed_provider import ParallelizedCachedEmbedProvider
from babyvec.index.abstract_index import AbstractIndex
from babyvec.index.numpy_faiss_index_factory import NumpyFaissIndexFactory
from babyvec.models import *
from babyvec.store.abstract_embedding_store import EmbeddingPersistenceOptions
from babyvec.store.abstract_metadata_store import AbstractMetadataStore
from babyvec.store.embedding_store_numpy import EmbeddingStoreNumpy
from babyvec.store.metadata_store_sqlite import MetadataStoreSQLite


# 'packaged' providers
class CachedParallelJinaEmbedder(ParallelizedCachedEmbedProvider):
    def __init__(
            self,
            persist_dir: str,
            n_computers: int,
            device: str,
    ):
        store = EmbeddingStoreNumpy(EmbeddingPersistenceOptions(
            persist_options=PersistenceOptions(persist_dir=persist_dir),
            metadata_store_type=MetadataStoreSQLite,
        ))
        super().__init__(
            n_computers=n_computers,
            compute_options=EmbedComputeOptions(
                device=device,
            ),
            computer_type=EmbeddingComputerJinaBert,
            store=store,
        )


class SemanticDb:
    def __init__(
            self,
            *,
            metadata_store: AbstractMetadataStore,
            index: AbstractIndex,
    ):
        self.metadata_store = metadata_store
        self.index = index
        return

    def search(self, query: str, n: int) -> list[DbSearchResult]:
        index_results = self.index.search(query, n)
        results: list[DbSearchResult] = []
        for r in index_results:
            text, metadata = self.metadata_store.get_embedding_text_and_metadata(
                r.embedding_id
            )
            results.append(DbSearchResult(
                index_search_result=r,
                text=text,
                metadata=metadata,
            ))
        return results


def FaissNumpyJinaSemanticDb(
        *,
        persist_dir: str,
        device: str,
):
    """
    MAY NEED TO SET:
      export KMP_DUPLICATE_LIB_OK='True'
    """
    store = EmbeddingStoreNumpy(EmbeddingPersistenceOptions(
        persist_options=PersistenceOptions(persist_dir=persist_dir),
        metadata_store_type=MetadataStoreSQLite,
    ))
    computer = EmbeddingComputerJinaBert(compute_options=EmbedComputeOptions(
        device=device,
    ))

    index_factory = NumpyFaissIndexFactory(
        store=store,
        computer=computer,
    )

    return SemanticDb(
        metadata_store=store.metadata_store,
        index=index_factory.build_index(),
    )
