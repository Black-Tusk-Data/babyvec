#!/usr/bin/env python3

import json
import logging
import os
import time

from babyvec import FaissNumpyJinaSemanticDb
from common import setup_logging


setup_logging()

# This is likely required on macOS, see https://github.com/kyamagu/faiss-wheels/issues/73.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():
    t0 = time.time()
    db = FaissNumpyJinaSemanticDb(
        persist_dir="./persist",
        device="mps",
    )
    t1 = time.time()
    logging.info(
        "loaded index in %f seconds",
        round(t1 - t0, 2)
    )
    while True:
        query = input("Search Python documentation: ")
        if not query:
            return
        matches = db.search(
            query,
            5,
        )
        logging.info("results: \n%s", json.dumps([
            m._asdict()
            for m in matches
        ], indent=2))

    return


if __name__ == '__main__':
    main()
