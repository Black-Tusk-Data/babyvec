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


class EmbeddingStore:
    def __init__(
            self,
            dbcon: sqlite3.Connection, # assumed to be in memory
    ):
        self.dbcon = dbcon
        self.dbcon.row_factory = sqlite3.Row
        self._index_seq = self._query("select max(rowid) rowid from fragment_embed")[0]["rowid"] or 1
        return

    @contextlib.contextmanager
    def _cursor(self):
        try:
            cursor = self.dbcon.cursor()
            yield cursor
        finally:
            cursor.close()
        return

    def _query(self, *args, **kwargs) -> list[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute(*args, **kwargs)
            return cur.fetchall()
        return

    def get(self, text: str) -> Embedding | None:
        rows = self._query(
            """
            select vector_to_json(fe.embedding) embedding
              from fragment f
                   join fragment_embed fe
                       on fe.rowid = f.embed_id
             where f.text = :text
            """, {
                "text": text,
            }
        )
        if not rows:
            return None
        return json.loads(rows[0]["embedding"])

    def persist_to_disk(self, storage_path: FileRef):
        to_db = sqlite3.connect(storage_path.abspath)
        self.dbcon.backup(to_db)
        return

    def put(self, text: str, embedding: Embedding) -> str:
        """
        returns fragment ID
        """
        fragment_id = str(uuid4())
        with self._cursor() as cur:
            # cur.execute("""
            # with text_match as (
            #   select embed_id
            #     from fragment
            #     where text = :text
            # )  delete from fragment_embed
            #   where rowid in (select embed_id from text_match)
            # """)
            cur.execute("""
            insert into fragment (
              id,
              text,
              embed_id
            ) values (
              :id,
              :text,
              :embed_id
            )
            """, {
                "id": fragment_id,
                "text": text,
                "embed_id": self._index_seq,
            })

            cur.execute("""
            insert into fragment_embed (
                  rowid,
                  embedding
                )
                values (
                  :rowid,
                  :embedding
                )
            """, {
                "rowid": self._index_seq,
                "embedding": json.dumps(embedding),
            })

            self._index_seq += 1
            self.dbcon.commit()

        return fragment_id

    @staticmethod
    def initialize(
            *,
            embedding_size: int,
    ) -> "EmbeddingStore":
        schema_file = os.path.join(
            FileRef.parse(__file__).absdir,
            "schema.sql",
        )
        mem_db = init_in_mem_db()
        with open(schema_file, "r") as f:
            schema_template = Template(f.read())
            cur = mem_db.cursor()
            cur.executescript(schema_template.substitute({
                "EMBEDDING_SIZE": embedding_size,
            }))
            cur.close()
        return EmbeddingStore(mem_db)

    @staticmethod
    def load_from_disk(storage_path: FileRef) -> "EmbeddingStore":
        assert os.path.exists(storage_path.abspath)

        mem_db = init_in_mem_db()
        disk_db = sqlite3.connect(storage_path.abspath)
        # disk_db.enable_load_extension(True)
        # sqlite_vss.load(disk_db)
        disk_db.backup(mem_db)
        return EmbeddingStore(mem_db)
