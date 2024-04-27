from typing import NamedTuple
import numpy.typing as npt

Embedding = npt.ArrayLike
EmbeddingId = int


class EmbedComputeOptions(NamedTuple):
    device: str


class IndexSearchResult(NamedTuple):
    text: str
    distance: float
