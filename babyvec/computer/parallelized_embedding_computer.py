import json
import json
import logging
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Type

import numpy as np

from babyvec.computer.abstract_embedding_computer import AbstractEmbeddingComputer
from babyvec.models import EmbedComputeOptions, Embedding

KILL_COMMAND = b"STOP"


def worker_process(
    i: int,
    child_con: Connection,
    computer_type: Type[AbstractEmbeddingComputer],
    compute_options: EmbedComputeOptions,
):
    computer = computer_type(compute_options)
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


class ParallelizedEmbeddingComputer(AbstractEmbeddingComputer):
    def __init__(
        self,
        n_computers: int,
        compute_options: EmbedComputeOptions,
        computer_type: Type[AbstractEmbeddingComputer],
    ):
        super().__init__(compute_options)
        self.computer_processes = []
        self.computer_connections = []
        self.computer_type = computer_type

        for i in range(n_computers):
            parent_con, child_con = Pipe()
            self.computer_connections.append(parent_con)
            self.computer_processes.append(
                Process(
                    target=worker_process,
                    args=(i, child_con, computer_type, compute_options),
                )
            )
            self.computer_processes[-1].start()
        return

    def compute_embeddings(self, texts: list[str]) -> list[Embedding]:
        n = len(texts)
        chunk_size = max(n // len(self.computer_connections), 1)
        chunks = [texts[i : i + chunk_size] for i in range(0, n, chunk_size)]

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
        logging.info("shutdown")
        self.shutdown()
        return

    def shutdown(self):
        for con in self.computer_connections:
            con.send_bytes(KILL_COMMAND)
        return
