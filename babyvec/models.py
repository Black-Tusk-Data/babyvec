from typing import NamedTuple

import numpy as np
import numpy.typing as npt


Embedding = npt.NDArray
EmbeddingId = int


class EmbedComputeOptions(NamedTuple):
    device: str


class PersistenceOptions(NamedTuple):
    persist_dir: str


class IndexSearchResult(NamedTuple):
    embedding_id: EmbeddingId
    distance: float


class DbSearchResult(NamedTuple):
    index_search_result: IndexSearchResult
    text: str
    metadata: dict
