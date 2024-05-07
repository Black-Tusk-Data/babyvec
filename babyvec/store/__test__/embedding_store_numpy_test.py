import os
import shutil
from unittest import TestCase

import numpy as np

from babyvec.models import PersistenceOptions
from babyvec.store.metadata_store_sqlite import MetadataStoreSQLite
from ..embedding_store_numpy import EmbeddingStoreNumpy

np.random.seed(11)

PERSIST_DIR = "/tmp/.tmp-store-test"
EMBED_LENGTH = 123

rng = np.random.default_rng()


embeddings = {
    "anything": rng.standard_normal(size=EMBED_LENGTH, dtype=np.float32),
    "another": rng.standard_normal(size=EMBED_LENGTH, dtype=np.float32),
}

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    pass


persist_options = PersistenceOptions(persist_dir=PERSIST_DIR)


def get_store():
    return EmbeddingStoreNumpy(
        persist_options=persist_options,
        metadata_store=MetadataStoreSQLite(persist_options),
    )


shutil.rmtree(PERSIST_DIR, ignore_errors=True)


class EmbeddingStoreNumpy_Test(TestCase):
    def test_basic_getting_and_setting(self):
        store = get_store()

        self.assertIsNone(store.get("anything"))
        store.put(text="anything", embedding=embeddings["anything"])
        res = store.get("anything")
        np.testing.assert_array_equal(
            embeddings["anything"],
            store.get("anything"),  # type: ignore
        )

        self.assertIsNone(store.get("another"))
        store.put(text="another", embedding=embeddings["another"])
        np.testing.assert_array_equal(
            embeddings["another"],
            store.get("another"),  # type: ignore
        )
        return

    def test_multiple_put(self):
        store = get_store()
        embeds = [
            rng.standard_normal(size=EMBED_LENGTH, dtype=np.float32)
            for i in range(1000)
        ]
        for i in range(0, len(embeds), 100):
            chunk = embeds[i : i + 100]
            store.put_many(
                texts=[f"thing-{i + k}" for k in range(len(chunk))],
                embeddings=chunk,
            )

        for i, embed in enumerate(embeds):
            np.testing.assert_array_equal(
                store.get(f"thing-{i}"),  # type: ignore
                embed,
            )
        return

    def test_persistence(self):
        embeds = [
            rng.standard_normal(size=EMBED_LENGTH, dtype=np.float32) for _ in range(100)
        ]
        store1 = get_store()
        for i, embed in enumerate(embeds):
            store1.put(text=f"test-{i}", embedding=embed)

        store2 = get_store()
        for i, embed in enumerate(embeds):
            np.testing.assert_array_equal(
                store2.get(f"test-{i}"),  # type: ignore
                embed,
            )
        return
