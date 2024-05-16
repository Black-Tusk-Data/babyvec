from typing import NamedTuple

import numpy as np
import numpy.typing as npt

EmbeddingScalarType = np.float32
Embedding = npt.NDArray[EmbeddingScalarType]
EmbeddingId = int


class EmbedComputeOptions(NamedTuple):
    device: str
    pass


class PersistenceOptions(NamedTuple):
    persist_dir: str
    pass


class IndexSearchResult(NamedTuple):
    embedding_id: EmbeddingId
    distance: float
    pass


class CorpusFragment(NamedTuple):
    fragment_id: str
    text: str
    metadata: dict
    pass


class DbSearchResult(NamedTuple):
    index_search_result: IndexSearchResult
    fragment: CorpusFragment
    pass
