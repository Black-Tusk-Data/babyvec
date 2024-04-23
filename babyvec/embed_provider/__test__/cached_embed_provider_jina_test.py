from unittest import TestCase

import numpy as np

from babyvec.models import Embedding

from ..cached_embed_provider_jina import CachedEmbedProviderJina


def parse_embedding(embed):
    return "".join(chr(x) for x in embed)


class CachedEmbedProviderJina_Test(TestCase):
    def test_multiple_computers(self):
        texts = [
            f"howdy! number {i}"
            for i in range(100)
        ]

        embeddings_single: list[Embedding]
        embeddings_multi: list[Embedding]
        with CachedEmbedProviderJina(
            persist_dir="/tmp/tmp-jina-provider-test-1",
            device="cpu",
            n_computers=1,
        ) as single_provider:
            embeddings_single = single_provider.get_embeddings(texts)

        with CachedEmbedProviderJina(
            persist_dir="/tmp/tmp-jina-provider-test-2",
            device="cpu",
            n_computers=2,
        ) as multi_provider:
            embeddings_multi = multi_provider.get_embeddings(texts)

        for single_embed, multi_embed in zip(embeddings_single, embeddings_multi):
            np.testing.assert_allclose(
                single_embed,
                multi_embed,
                atol=0.0001,
            )

        return
