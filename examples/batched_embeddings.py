#!/usr/bin/env python3.12

import itertools
import json
import logging
import os
import time

from babyvec import CachedParallelJinaEmbedder


def setup_logging():
    LOG_FMT = (
        f"%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
        format=LOG_FMT,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    return


setup_logging()


def load_fragments():
    files = [
        os.path.join("./transcribed", fname)
        for fname in os.listdir("./transcribed")
    ]
    t0 = time.time()
    for i, fname in enumerate(files):
        with open(fname, "r") as f:
            contents = json.loads(f.read())
            for chunk in contents["transcription"]:
                yield chunk["text"].strip()
        t1 = time.time()
        rate = round((i + 1) / (t1 - t0), 2)
        logging.info("completed %d at rate %f", i, rate)
    return



def main():
    N_COMPUTERS = 3
    CHUNK_SIZE = 100 * N_COMPUTERS
    with CachedParallelJinaEmbedder(
        persist_dir="./persist",
        device="mps",
        n_computers=N_COMPUTERS,
    ) as embedder:
        for texts in itertools.batched(load_fragments(), CHUNK_SIZE):
            embs = embedder.get_embeddings(list(texts))
    return


if __name__ == '__main__':
    main()
