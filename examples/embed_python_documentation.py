#!/usr/bin/env python3.12

import logging
import time

from babyvec import CachedParallelJinaEmbedder

from common import get_python_documentation_fragments, setup_logging


N_COMPUTERS = 1


setup_logging()


def main():
    fragments = get_python_documentation_fragments()
    logging.info("computing %d embeddings...", len(fragments))
    chunk_size = N_COMPUTERS * 1
    t0 = time.time()

    with CachedParallelJinaEmbedder(
        persist_dir="./persist",
        n_computers=N_COMPUTERS,
        device="mps",
    ) as embedder:
        for lo in range(0, len(fragments), chunk_size):
            chunk = fragments[lo:lo+chunk_size]
            embedder.get_embeddings(chunk)
            t1 = time.time()
            rate = round((lo + chunk_size) / (t1 - t0), 2)
            logging.debug(
                "computing %f embeddings / second",
                rate
            )
    logging.info(
        "computed %d embeddings in %f seconds",
        len(fragments),
        round(time.time() - t0, 2),
    )
    return


if __name__ == '__main__':
    main()
