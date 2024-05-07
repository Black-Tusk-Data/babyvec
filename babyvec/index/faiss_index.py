import logging

import faiss  # type: ignore
import numpy as np
import numpy.typing as npt

from babyvec.computer.abstract_embedding_computer import AbstractEmbeddingComputer
from babyvec.index.abstract_index import AbstractIndex
from babyvec.models import EmbeddingScalarType, IndexSearchResult


class FaissIndex(AbstractIndex):
    def __init__(
        self,
        computer: AbstractEmbeddingComputer,
        vectors: npt.NDArray[EmbeddingScalarType],
    ):
        assert len(vectors.shape) == 2, "expected to add a 2-dimensional tensor"
        self.computer = computer
        embed_size = vectors.shape[1]
        self.index = faiss.IndexFlatL2(embed_size)
        # pylint: disable=no-value-for-parameter
        self.index.add(vectors)  # type: ignore
        logging.info("loaded FAISS index with %d vectors", vectors.shape[0])
        return

    def search(self, query: str, k_nearest: int) -> list[IndexSearchResult]:
        """
        Would be straightforward to search over multiple queries if necessary...
        """
        #
        query_embed = self.computer.compute_embeddings([query])
        # pylint: disable=no-value-for-parameter
        distances, embedding_ids = self.index.search(np.array(query_embed), k_nearest)  # type: ignore

        return [
            IndexSearchResult(
                embedding_id=embedding_id.item(),
                distance=distance.item(),
            )
            for embedding_id, distance in zip(embedding_ids[0], distances[0])
        ]
