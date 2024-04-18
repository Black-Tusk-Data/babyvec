import abc

from vectorspace.models import Embedding


class _EmbedImpl(abc.ABC):
    @abc.abstractmethod
    def _compute_embedderings(self, texts: list[str]) -> list[Embedding]:
        return
    
    def get_embeddings(self, texts: list[str]) -> list[Embedding]:
        return
