import abc
import os

from babyvec.models import CorpusFragment, EmbeddingId, PersistenceOptions


class AbstractMetadataStore(abc.ABC):

    def __init__(
        self,
        options: PersistenceOptions,
    ):
        self.options = options
        self.persist_dir = options.persist_dir
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
        return

    @abc.abstractmethod
    def add_text_embedding(self, *, text: str, embedding_id: EmbeddingId) -> None:
        pass

    @abc.abstractmethod
    def get_embedding_id(self, text: str) -> EmbeddingId | None:
        pass

    # TODO: may not need this, in favor of fetching relevant fragments
    @abc.abstractmethod
    def get_embedding_text(self, embedding_id: EmbeddingId) -> str:
        pass

    @abc.abstractmethod
    def get_fragments_for_embedding(
        self, embedding_id: EmbeddingId
    ) -> list[CorpusFragment]:
        pass

    @abc.abstractmethod
    def delete_fragment(
        self,
        fragment_id: str,
    ) -> None:
        pass

    @abc.abstractmethod
    def ingest_fragment(
        self,
        *,
        embedding_id: EmbeddingId,
        fragment: CorpusFragment,
    ) -> None:
        pass

    @abc.abstractmethod
    def compact_embeddings(self) -> list[EmbeddingId]:
        """
        Deletes all embeddings that are not used by a fragment, and returns their embedding IDs.
        """
        pass

    @abc.abstractmethod
    def migrate_embedding_id(
        self,
        *,
        from_embedding_id: EmbeddingId,
        to_embedding_id: EmbeddingId,
    ) -> None:
        pass

    @abc.abstractmethod
    def get_all_fragment_ids(self) -> list[str]:
        pass

    pass
