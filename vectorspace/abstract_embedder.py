import abc

from vectorspace.embedding_store import EmbeddingStore
from vectorspace.models import Embedding


class AbstractEmbedder(abc.ABC):
    def __init__(
            self,
            store: EmbeddingStore,
    ):
        self.store = store
        return


    @abc.abstractmethod
    def _compute_embeddings(self, texts: list[str]) -> list[Embedding]:
        return

    def get_embeddings(self, texts: list[str]) -> list[Embedding]:
        cached = [
            self.store.get(text)
            for text in texts
        ]
        to_compute = {
            text: i
            for i, text in enumerate(texts) if not cached[i]
        }
        to_compute_uniq = list(set(to_compute.keys()))
        new_embeddings = self._compute_embeddings(to_compute_uniq)
        for text, embed in zip(to_compute_uniq, new_embeddings):
            self.store.put(text, embed)
            cached[to_compute[text]] = embed
        return cached
