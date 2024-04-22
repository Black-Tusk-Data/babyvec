import os
import shutil
import unittest
from unittest import TestCase

import numpy as np

from ..embedding_store_numpy import EmbeddingStoreNumpy

np.random.seed(11)

PERSIST_DIR = "./.tmp"
EMBED_LENGTH = 123

embeddings = {
    "anything": np.random.random(EMBED_LENGTH),
    "another": np.random.random(EMBED_LENGTH),
}

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)


class EmbeddingStoreNumpy_Test(TestCase):
    def test_basic_getting_and_setting(self):
        store = EmbeddingStoreNumpy(persist_dir=PERSIST_DIR)
        self.assertIsNone(store.get("anything"))
        store.put("anything", embeddings["anything"])
        res = store.get("anything")
        np.testing.assert_array_equal(
            embeddings["anything"],
            store.get("anything"),
        )

        self.assertIsNone(store.get("another"))
        store.put("another", embeddings["another"])
        np.testing.assert_array_equal(
            embeddings["another"],
            store.get("another"),
        )
        return

    def test_multiple_put(self):
        store = EmbeddingStoreNumpy(persist_dir=PERSIST_DIR)
        embeds = [
            np.random.random(EMBED_LENGTH)
            for i in range(1000)
        ]
        for i in range(0, len(embeds), 100):
            chunk = embeds[i:i+100]
            store.put_many(
                [f"thing-{i + k}" for k in range(len(chunk))],
                chunk,
            )

        for i, embed in enumerate(embeds):
            np.testing.assert_array_equal(
                store.get(f"thing-{i}"),
                embed,
            )
        return
