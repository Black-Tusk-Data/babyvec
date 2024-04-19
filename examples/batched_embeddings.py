#!/usr/bin/env python3.12

import itertools
import json
import os

from babyvec import BabyVecLocalEmbedder


def load_fragments():
    files = [
        os.path.join("./transcribed", fname)
        for fname in os.listdir("./transcribed")
    ]
    for fname in files:
        with open(fname, "r") as f:
            contents = json.loads(f.read())
            for chunk in contents["transcription"]:
                yield chunk["text"]
        print("finished", fname)
    return



def main():
    CHUNK_SIZE = 1000
    with BabyVecLocalEmbedder(
        persist_path="./db.sq3",
        embedding_size=768,
        model="jinaai/jina-embeddings-v2-base-en",
        device="mps",
    ) as bv:
        for texts in itertools.batched(load_fragments(), CHUNK_SIZE):
            bv.get_embeddings(list(texts))
    return


if __name__ == '__main__':
    main()
