import os
import shutil
from unittest import TestCase

from babyvec.faiss_db import FaissDb
from babyvec.models import CorpusFragment


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

PERSIST_DIR = "/tmp/CorpusFragmentFaissDb_Test"
shutil.rmtree(PERSIST_DIR, ignore_errors=True)


FRAGMENTS = [
    CorpusFragment(
        text="cat",
        metadata={"furry": True},
    ),
    CorpusFragment(
        text="dog",
        metadata={"furry": True},
    ),
    CorpusFragment(
        text="dolphin",
        metadata={"furry": False},
    ),
    CorpusFragment(
        text="seal",
        metadata={"furry": False},
    ),
]


class FaissDb_Test(TestCase):
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

    pass