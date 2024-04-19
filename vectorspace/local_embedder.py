from transformers import AutoModel

from vectorspace.abstract_embedder import AbstractEmbedder
from vectorspace.embedding_store import EmbeddingStore
from vectorspace.models import Embedding


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
        return self.embedding_model.encode(texts, **kwargs)
