import contextlib
import sqlite3

from vectorspace.models import Embedding


class EmbeddingStore:
    def __init__(
            self,
            dbcon: sqlite3.Connection, # assumed to be in memory
    ):
        self.dbcon = dbcon
        self.dbcon.row_factory = sqlite3.Row
        return

    @classmethod
    @contextlib.contextmanager
    def _cursor(cls):
        try:
            print("CLS:", cls)
            cursor = cls.dbcon.cursor()
            yield cursor
        finally:
            cursor.close()
        return

    def get(self, text: str) -> Embedding | None:
        with self._cursor as cur:
            cur.execute(
                """
                select 1
                """
            )
            return cur.fetchall()
        return

    def put(self, text: str, embedding: Embedding) -> None:
        return
