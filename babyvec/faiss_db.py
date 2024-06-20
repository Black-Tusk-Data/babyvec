import logging

import numpy as np
import numpy.typing as npt

from babyvec.computer.abstract_embedding_computer import AbstractEmbeddingComputer
from babyvec.computer.embedding_computer_jina_bert import EmbeddingComputerJinaBert
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
    FragmentFilter,
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
        track_for_compacting: bool = False,
    ):
        """
        Init a 'DB-like' interface, featuring:
         - an embedding computer
         - a persitent store for embeddings
         - a liked metadata store for 'text fragments', which is many-to-one with stored embeddings
         - the ability to define an Faiss index over the stored embeddings, to enable a nearest-nerighbour search over the embedding space
        Options:
         - track_for_compacting: If True, when the DB is closed, any fragment that was not ingested while it was open will be deleted, in addition to any orphaned embeddings.
        """
        self.persist_options = PersistenceOptions(persist_dir=persist_dir)
        self.compute_options = EmbedComputeOptions(
            device=device,
        )
        self.metadata_store = metadata_store_type(self.persist_options)
        self.embed_store = embed_store_type(
            metadata_store=self.metadata_store,
            persist_options=self.persist_options,
        )
        self.embedding_provider: CachedEmbedProvider
        if n_computers == 1:
            self.embedding_provider = CachedEmbedProvider(
                computer=computer_type(self.compute_options),
                store=self.embed_store,
            )
            pass
        else:
            self.embedding_provider = ParallelizedCachedEmbedProvider(
                n_computers=n_computers,
                compute_options=self.compute_options,
                computer_type=computer_type,
                store=self.embed_store,
            )
            pass
        self.track_for_compacting = track_for_compacting
        self.tracked_ingested_fragment_ids: set[str] = set()
        return

    def ingest_fragments(self, fragments: list[CorpusFragment]) -> None:
        embedding_ids = self.embedding_provider.persist_embeddings(
            [fragment.text for fragment in fragments]
        )

        for embed_id, fragment in zip(embedding_ids, fragments):
            if self.track_for_compacting:
                self.tracked_ingested_fragment_ids.add(fragment.fragment_id)
                pass
            self.embed_store.metadata_store.ingest_fragment(
                embedding_id=embed_id,
                fragment=fragment,
            )
            pass
        return

    def _get_index(
        self, embedding_ids: npt.NDArray[np.int64] | None = None
    ) -> FaissIndex:
        vectors = (
            self.embed_store.embed_table
            if embedding_ids is None
            else self.embed_store.embed_table[embedding_ids]
        )
        return FaissIndex(
            computer=self.embedding_provider.computer,
            vectors=vectors,
        )

    def search(
        self,
        query: str,
        k_nearest: int,
        *,
        fragment_filter: FragmentFilter | None = None,
    ) -> list[DbSearchResult]:
        embedding_ids: npt.NDArray[np.int64] | None = None
        if fragment_filter:
            embedding_ids = (
                self.embed_store.metadata_store.get_embedding_ids_for_fragment_filter(
                    fragment_filter
                )
            )
            if not embedding_ids.shape[0]:
                return []
            pass

        index = self._get_index(embedding_ids)
        index_hits = index.search(query, k_nearest)
        results: list[DbSearchResult] = []
        for hit in index_hits:
            if hit.embedding_id < 0:
                continue
            for fragment in self.embed_store.metadata_store.get_fragments_for_embedding(
                hit.embedding_id
                if embedding_ids is None
                else int(embedding_ids[hit.embedding_id])
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
        self.tracked_ingested_fragment_ids = set()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        if not self.track_for_compacting:
            return
        all_fragment_ids = set(self.metadata_store.get_all_fragment_ids())
        delete_fragment_ids = all_fragment_ids.difference(
            self.tracked_ingested_fragment_ids
        )
        logging.info("deleting %d fragments", len(delete_fragment_ids))
        for fragment_id in delete_fragment_ids:
            self.metadata_store.delete_fragment(fragment_id)
            pass
        delete_embed_ids = self.metadata_store.compact_embeddings()
        logging.info("deleting %d embeddings", len(delete_embed_ids))
        self.embed_store.delete_embeddings(delete_embed_ids)
        return

    pass
