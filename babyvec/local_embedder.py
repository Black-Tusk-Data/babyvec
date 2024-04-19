from enum import StrEnum
from transformers import AutoModel

from babyvec.abstract_embedder import AbstractEmbedder
from babyvec.embedding_store import EmbeddingStore
from babyvec.models import Embedding


class LocalEmbedModel(StrEnum):
    JINA_AI_V2 = "jinaai/jina-embeddings-v2-base-en"


class LocalEmbedder(AbstractEmbedder):
    def __init__(
            self,
            *,
            store: EmbeddingStore,
            model: str,
            device: str | None = None
    ):
        super().__init__(store=store)
        self.model = model
        self.embedding_model = AutoModel.from_pretrained(
            model,
            trust_remote_code=True
        )
        self.device = device
        return

    def _compute_embeddings(self, texts: list[str]) -> list[Embedding]:
        kwargs = {"device": self.device} if self.device else {}
        return self.embedding_model.encode(texts, **kwargs).tolist()
