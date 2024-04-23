from unittest import TestCase

import numpy as np

from babyvec.computer.embedding_computer_jina_bert import EmbeddingComputerJinaBert
from babyvec.models import EmbedComputeOptions, Embedding

from ..parallelized_embedding_computer import ParallelizedEmbeddingComputer


class CachedEmbedProviderJina_Test(TestCase):
    def test_multiple_computers(self):
        compute_options = EmbedComputeOptions(device="cpu")
        texts = [
            f"howdy! number {i}"
            for i in range(100)
        ]

        embeddings_single: list[Embedding]
        embeddings_multi: list[Embedding]
        with ParallelizedEmbeddingComputer(
                compute_options=compute_options,
                n_computers=1,
                computer_type=EmbeddingComputerJinaBert,
        ) as single_provider:
            embeddings_single = single_provider.compute_embeddings(texts)

        with ParallelizedEmbeddingComputer(
                compute_options=compute_options,
                n_computers=2,
                computer_type=EmbeddingComputerJinaBert,
        ) as multi_provider:
            embeddings_multi = multi_provider.compute_embeddings(texts)

        for single_embed, multi_embed in zip(embeddings_single, embeddings_multi):
            np.testing.assert_allclose(
                single_embed,
                multi_embed,
                atol=0.0001,
            )

        return
