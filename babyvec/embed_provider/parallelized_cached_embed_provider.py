from typing import Type

from babyvec.computer.abstract_embedding_computer import AbstractEmbeddingComputer
from babyvec.computer.parallelized_embedding_computer import (
    ParallelizedEmbeddingComputer,
)
from babyvec.embed_provider.cached_embed_provider import CachedEmbedProvider
from babyvec.models import EmbedComputeOptions
from babyvec.store.abstract_embedding_store import AbstractEmbeddingStore


class ParallelizedCachedEmbedProvider(CachedEmbedProvider):
    computer: ParallelizedEmbeddingComputer

    def __init__(
        self,
        *,
        n_computers: int,
        compute_options: EmbedComputeOptions,
        computer_type: Type[AbstractEmbeddingComputer],
        store: AbstractEmbeddingStore,
    ):
        self.n_computers = n_computers
        self.compute_options = compute_options
        self.computer_type = computer_type
        computer = ParallelizedEmbeddingComputer(
            n_computers=n_computers,
            compute_options=compute_options,
            computer_type=computer_type,
        )

        super().__init__(computer=computer, store=store)
        return
