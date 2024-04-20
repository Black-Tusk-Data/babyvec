import abc
import logging
import time

from babyvec.models import Embedding


class AbstractEmbedProvider(abc.ABC):
    @abc.abstractmethod
    def get_embeddings(self, texts: list[str]) -> list[Embedding]:
        pass
