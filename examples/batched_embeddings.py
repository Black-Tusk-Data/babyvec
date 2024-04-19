#!/usr/bin/env python3

from babyvec import BabyVecLocalEmbedder


def main():
    bv = BabyVecLocalEmbedder(
        persist_path="./db.sq3",
        embedding_size=768,
        model="jinaai/jina-embeddings-v2-base-en",
        device="mps",
    )
    return


if __name__ == '__main__':
    main()
