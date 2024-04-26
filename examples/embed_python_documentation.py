#!/usr/bin/env python3.12

import time

from babyvec import CachedParallelJinaEmbedder

from common import get_python_documentation_sentences


N_COMPUTERS = 2


def main():
    sentences = get_python_documentation_sentences()
    embedder = CachedParallelJinaEmbedder(
        persist_dir="./persist",
        n_computers=N_COMPUTERS,
        device="mps",
    )
    chunk_size = N_COMPUTERS * 100
    t0 = time.time()
    for lo in range(0, len(sentences), chunk_size):
        chunk = sentences[lo:lo+chunk_size]
        embedder.get_embeddings(chunk)
        t1 = time.time()
        rate = round((lo + chunk_size) / t1 - t0, 2)
        print("computing ", rate, "embeddings / second", end="\r")
    print(
        "computed ",
        len(sentences),
        "embeddings in",
        time.time() - t0,
        "seconds"
    )
    return


if __name__ == '__main__':
    main()
