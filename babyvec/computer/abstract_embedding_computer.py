import abc

from babyvec.models import EmbedComputeOptions, Embedding


class AbstractEmbeddingComputer(abc.ABC):
    def __init__(self, compute_options: EmbedComputeOptions):
        self.compute_options = compute_options
        return

    @abc.abstractmethod
    def compute_embeddings(self, texts: list[str]) -> list[Embedding]:
        pass

    def shutdown(self) -> None:
        return
