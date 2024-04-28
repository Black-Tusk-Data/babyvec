from babyvec.models import EmbeddingId
from babyvec.store.abstract_metadata_store import AbstractMetadataStore


class MetadataStoreSQLite(AbstractMetadataStore):
    def set_embedding_id(self, *, text: str, embedding_id: EmbeddingId) -> None:
        return

    def get_embedding_id(self, text: str) -> EmbeddingId | None:
        return None

    def get_embedding_metadata(self, embedding_id: EmbeddingId) -> dict:
        return {}
