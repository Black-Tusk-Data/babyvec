from babyvec.computer.abstract_embedding_computer import AbstractEmbeddingComputer
from babyvec.index.abstract_index_factory import AbstractIndexFactory
from babyvec.index.faiss_index import FaissIndex
from babyvec.store.embedding_store_numpy import EmbeddingStoreNumpy


class NumpyFaissIndexFactory(AbstractIndexFactory):
    store: EmbeddingStoreNumpy

    def __init__(
        self,
        *,
        store: EmbeddingStoreNumpy,  # TODO: ideally would be a generic in the factory
        computer: AbstractEmbeddingComputer,
    ):
        super().__init__(store=store, computer=computer)
        self.store = store
        return

    def build_index(self) -> FaissIndex:
        store: EmbeddingStoreNumpy = self.store
        return FaissIndex(
            computer=self.computer,
            vectors=store.embed_table,
        )
