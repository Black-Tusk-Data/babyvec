import abc
import contextlib
import json
import os
import sqlite3
from string import Template
from uuid import uuid4

import sqlite_vss

from babyvec.common import FileRef
from babyvec.models import Embedding


def init_in_mem_db() -> sqlite3.Connection:
    mem_db = sqlite3.connect(":memory:")
    mem_db.enable_load_extension(True)
    sqlite_vss.load(mem_db)
    return mem_db


class AbstractEmbeddingStore(abc.ABC):
    @abc.abstractmethod
    def get(self, text: str) -> Embedding | None:
        pass

    # @abc.abstractmethod
    # def persist_to_disk(self, storage_path: FileRef):
    #     pass

    @abc.abstractmethod
    def put(self, text: str, embedding: Embedding) -> None:
        pass

    # @staticmethod
    # @abc.abstractmethod
    # def initialize(
    #         *,
    #         embedding_size: int,
    # ) -> "AbstractEmbeddingStore":
    #     pass

    # @staticmethod
    # @abc.abstractmethod
    # def load_from_disk(storage_path: FileRef) -> "AbstractEmbeddingStore":
    #     pass
