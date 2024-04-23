import abc
import logging
import time

from babyvec.models import Embedding


class AbstractEmbeddingComputer(abc.ABC):
    @abc.abstractmethod
    def compute_embeddings(self, texts: list[str]) -> list[Embedding]:
        pass
