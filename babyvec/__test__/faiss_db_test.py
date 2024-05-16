import os
import shutil
from unittest import TestCase

from babyvec.faiss_db import FaissDb
from babyvec.models import CorpusFragment

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

PERSIST_DIR = "/tmp/CorpusFragmentFaissDb_Test"


FRAGMENTS = [
    CorpusFragment(
        fragment_id="cat",
        text="cat",
        metadata={"furry": True},
    ),
    CorpusFragment(
        fragment_id="dog",
        text="dog",
        metadata={"furry": True},
    ),
    CorpusFragment(
        fragment_id="dolphin",
        text="dolphin",
        metadata={"furry": False},
    ),
    CorpusFragment(
        fragment_id="seal",
        text="seal",
        metadata={"furry": False},
    ),
]


class FaissDb_Test(TestCase):
    def setUp(self) -> None:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        return super().setUp()

    def test_ingesting_into_and_searching_db(self):
        with FaissDb(
            persist_dir=PERSIST_DIR,
            device="cpu",
        ) as db:
            db.ingest_fragments(FRAGMENTS)
            db.index_existing_fragments()
            result = db.search(
                "house pet",
                4,
            )
            self.assertEqual(len(result), 4)
            # first two should be our cat and dog
            self.assertEqual(
                list(sorted(res.fragment.text for res in result[:2])), ["cat", "dog"]
            )
            self.assertEqual(
                [res.fragment.metadata["furry"] for res in result[:2]],
                [True, True],
            )
            self.assertEqual(
                [res.fragment.metadata["furry"] for res in result[2:]],
                [False, False],
            )
        return

    def test_non_duplicate_inserts(self):
        with FaissDb(
            persist_dir=PERSIST_DIR,
            device="cpu",
        ) as db:
            db.ingest_fragments(FRAGMENTS)
            db.index_existing_fragments()
            db.ingest_fragments(FRAGMENTS)
            db.index_existing_fragments()
            result = db.search(
                "house pet",
                4,
            )
            self.assertEqual(
                set(res.fragment.text for res in result),
                set(["cat", "dog", "dolphin", "seal"]),
            )
        return

    def test_compacting(self):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

        with FaissDb(
            persist_dir=PERSIST_DIR,
            device="cpu",
        ) as db:
            db.ingest_fragments(FRAGMENTS)
            pass

        with FaissDb(
            persist_dir=PERSIST_DIR, device="cpu", track_for_compacting=True
        ) as db:
            db.ingest_fragments(FRAGMENTS[:2])
            pass
        return

        with FaissDb(
            persist_dir=PERSIST_DIR,
            device="cpu",
        ) as db:
            db.index_existing_fragments()
            result = db.search("house pet", 4)
            self.assertEqual(len(result), 2)

            db.ingest_fragments(FRAGMENTS[2:])
            db.index_existing_fragments()
            result = db.search("house pet", 4)
            self.assertEqual(len(result), 4)
            pass
        return

    pass
