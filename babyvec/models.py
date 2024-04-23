from typing import NamedTuple
import numpy.typing as npt

# Embedding = list[float]
Embedding = npt.ArrayLike


# class PersistenceOptions(NamedTuple):
#     persist_dir: str


class EmbedComputeOptions(NamedTuple):
    device: str
