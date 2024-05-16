#!/usr/bin/env python3.12

import logging
import time
from uuid import uuid4

from babyvec.faiss_db import FaissDb
from babyvec.models import CorpusFragment

from common import get_python_documentation_fragments, setup_logging


N_COMPUTERS = 1


setup_logging()


def main():
    fragments = get_python_documentation_fragments()
    logging.info("computing %d embeddings...", len(fragments))
    chunk_size = N_COMPUTERS * 1
    t0 = time.time()
    with FaissDb(
        persist_dir="./persist",
        n_computers=1,
        device="mps",
        track_for_compacting=True,
    ) as vector_db:
        for lo in range(0, len(fragments), chunk_size):
            chunk = [
                CorpusFragment(
                    fragment_id=str(uuid4()),
                    text=fragment,
                    metadata={},
                )
                for fragment in fragments[lo : lo + chunk_size]
            ]
            vector_db.ingest_fragments(chunk)
            t1 = time.time()
            rate = round((lo + chunk_size) / (t1 - t0), 2)
            logging.debug("computing %f embeddings / second", rate)
            pass
    logging.info(
        "computed %d embeddings in %f seconds",
        len(fragments),
        round(time.time() - t0, 2),
    )
    return


if __name__ == "__main__":
    main()
