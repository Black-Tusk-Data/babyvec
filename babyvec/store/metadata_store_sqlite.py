import contextlib
import json
import os
import sqlite3

from babyvec.common import FileRef
from babyvec.models import EmbeddingId, PersistenceOptions
from babyvec.store.abstract_metadata_store import AbstractMetadataStore


DBNAME = "bbvec.sq3"

SCHEMA = """
  CREATE TABLE IF NOT EXISTS fragment (
    embed_id  INTEGER  NOT NULL  PRIMARY KEY,
    text  TEXT  NOT NULL  UNIQUE,
    metadata_json  TEXT
  )
"""


class MetadataStoreSQLite(AbstractMetadataStore):
    def __init__(self, options: PersistenceOptions):
        super().__init__(options)
        db_path = os.path.join(self.persist_dir, DBNAME)
        is_fresh = not os.path.exists(db_path)

        self.dbcon = sqlite3.connect(db_path)
        self.dbcon.row_factory = sqlite3.Row

        if is_fresh:
            with self._cursor() as cur:
                cur.executescript(SCHEMA)
        return

    @contextlib.contextmanager
    def _cursor(self, autocommit=True):
        try:
            cursor = self.dbcon.cursor()
            yield cursor
            if autocommit:
                self.dbcon.commit()
        finally:
            cursor.close()
        return

    def _query(self, *args, **kwargs) -> list[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute(*args, **kwargs)
            return cur.fetchall()
        return

    def add_text(self, *, text: str, embedding_id: EmbeddingId, metadata: dict) -> None:
        with self._cursor() as cur:
            cur.execute("""
            INSERT INTO fragment (
              embed_id,
              text,
              metadata_json
            ) VALUES (
              :embed_id,
              :text,
              :metadata_json
            )
            ON CONFLICT (text) DO UPDATE
            SET embed_id = :embed_id,
                metadata_json = :metadata_json
          WHERE text = :text
            """, {
                "embed_id": embedding_id,
                "text": text,
                "metadata_json": json.dumps(metadata),
            })
        return

    def get_embedding_id(self, text: str) -> EmbeddingId | None:
        rows = self._query("""
        select embed_id
          from fragment
         where text = :text
        """, {"text": text})
        if not rows:
            return None
        return rows[0]["embed_id"]

    def get_embedding_text_and_metadata(self, embedding_id: EmbeddingId) -> tuple[str, dict]:
        rows = self._query("""
        select text, metadata_json
          from fragment
         where embed_id = :embed_id
        """, {"embed_id": embedding_id})
        assert rows
        return (
            rows[0]["text"],
            json.loads(rows[0]["metadata_json"])
        )
