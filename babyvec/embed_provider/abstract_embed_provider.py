import abc

from babyvec.models import CorpusFragment, Embedding


class AbstractEmbedProvider(abc.ABC):
    # TODO: this name is bad.
    #   We rely on 'get_embeddings' to be cached in the'CachedEmbedProvider'
    #   for a number of use-cases, when this is not obvious that it is the case
    @abc.abstractmethod
    def get_embeddings(self, texts: list[str]) -> list[Embedding]:
        pass
