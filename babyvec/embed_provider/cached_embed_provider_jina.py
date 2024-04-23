import abc
import json
import logging
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import time

import numpy as np

from babyvec.computer.embedding_computer_jina_bert import EmbeddingComputerJinaBert
from babyvec.embed_provider.abstract_embed_provider import AbstractEmbedProvider
from babyvec.models import Embedding
from babyvec.store.embedding_store_numpy import EmbeddingStoreNumpy


KILL_COMMAND = b"STOP"


def worker_process(
        i: int,
        device: str,
        child_con: Connection,
):
    computer = EmbeddingComputerJinaBert(device=device)
    logging.debug("worker %d coming online...", i)
    while True:
        cmd = child_con.recv_bytes()
        logging.debug("worker %d caught work...", i)
        if cmd == KILL_COMMAND:
            return
        texts = json.loads(cmd)
        embeddings = computer.compute_embeddings(texts)
        flattened = np.concatenate(embeddings, axis=0)
        child_con.send_bytes(flattened.tobytes())
    return


class CachedEmbedProviderJina(AbstractEmbedProvider):
    def __init__(
            self,
            *,
            persist_dir: str,
            device: str,
            n_computers: int = 1,
    ) -> None:
        self.computer_processes = []
        self.computer_connections = []

        for i in range(n_computers):
            parent_con, child_con = Pipe()
            self.computer_connections.append(parent_con)
            self.computer_processes.append(Process(
                target=worker_process,
                args=(i, device, child_con),
            ))
            self.computer_processes[-1].start()

        self.computer = EmbeddingComputerJinaBert(device=device)
        self.store = EmbeddingStoreNumpy(persist_dir=persist_dir)
        return

    def _compute_embeddings(self, texts: list[str]) -> list[Embedding]:
        n = len(texts)
        chunk_size = n // len(self.computer_connections)
        chunks = [
            texts[i:i+chunk_size]
            for i in range(0, n, chunk_size)
        ]

        for con, chunk in zip(self.computer_connections, chunks):
            con.send_bytes(json.dumps(chunk).encode("utf-8"))

        res = []
        for con, chunk in zip(self.computer_connections, chunks):
            n_chunk = len(chunk)
            arr_buffer = con.recv_bytes()
            flattened = np.frombuffer(arr_buffer, dtype=np.float32)
            embeddings = flattened.reshape((n_chunk, -1))
            for i in range(n_chunk):
                res.append(embeddings[i])
        return res

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return

    def shutdown(self):
        for con in self.computer_connections:
            con.send_bytes(KILL_COMMAND)
        return
