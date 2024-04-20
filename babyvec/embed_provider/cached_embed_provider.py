import abc
import logging
import time

from babyvec.embed_provider.abstract_embed_provider import AbstractEmbedProvider
from babyvec.models import Embedding


class CachedEmbedProvider(AbstractEmbedProvider):
    def get_embeddings(self, texts: list[str]) -> list[Embedding]:
        cached = [
            self.store.get(text)
            for text in texts
        ]
        to_compute = {
            text: i
            for i, text in enumerate(texts) if cached[i] is None
        }
        logging.debug("found %d cached embeddings", len(cached) - len(to_compute))
        if not to_compute:
            return cached

        to_compute_uniq = list(set(to_compute.keys()))
        t0 = time.time()
        new_embeddings = self._compute_embeddings(to_compute_uniq)
        t1 = time.time()
        logging.debug(
            "computed %d embeddings in %d s",
            len(to_compute_uniq),
            round(t1 - t0, 2)
        )
        for text, embed in zip(to_compute_uniq, new_embeddings):
            self.store.put(text, embed)
            cached[to_compute[text]] = embed
        t2 = time.time()
        logging.debug(
            "stored %d embeddings in %d s",
            len(to_compute_uniq),
            round(t2 - t1, 2)
        )
        return cached
