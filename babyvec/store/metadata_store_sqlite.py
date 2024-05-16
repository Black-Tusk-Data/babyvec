import json
import logging
import os
from uuid import uuid4

from babyvec.lib.sqlitedb import SQLiteDB
from babyvec.models import CorpusFragment, EmbeddingId, PersistenceOptions
from babyvec.store.abstract_metadata_store import AbstractMetadataStore

DBNAME = "bbvec.sq3"

SCHEMA = """
  CREATE TABLE IF NOT EXISTS text_embedding (
    embed_id  INTEGER  NOT NULL  PRIMARY KEY,
    text  TEXT  NOT NULL  UNIQUE
  );

  CREATE TABLE IF NOT EXISTS fragment (
    fragment_id  TEXT  NOT NULL  PRIMARY KEY,
    embed_id  INTEGER  NOT NULL,
    text  TEXT  NOT NULL,
    metadata_json  TEXT,

    FOREIGN KEY (embed_id) REFERENCES text_embedding (embed_id)
  );
"""


class MetadataStoreSQLite(AbstractMetadataStore):
    def __init__(self, options: PersistenceOptions):
        super().__init__(options)
        self.db = SQLiteDB(
            dbfile_path=os.path.join(self.persist_dir, DBNAME),
            schema=SCHEMA,
        )
        return

    def add_text_embedding(self, *, text: str, embedding_id: EmbeddingId) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
            INSERT INTO text_embedding (
              embed_id,
              text
            ) VALUES (
              :embed_id,
              :text
            )
            ON CONFLICT (text) DO UPDATE
            SET embed_id = :embed_id
          WHERE text = :text
            """,
                {
                    "embed_id": embedding_id,
                    "text": text,
                },
            )
        return

    def get_embedding_id(self, text: str) -> EmbeddingId | None:
        rows = self.db.query(
            """
        select embed_id
          from text_embedding
         where text = :text
        """,
            {"text": text},
        )
        if not rows:
            return None
        return rows[0]["embed_id"]

    def get_embedding_text(self, embedding_id: EmbeddingId) -> str:
        rows = self.db.query(
            """
        select text
          from text_embedding
         where embed_id = :embed_id
        """,
            {"embed_id": embedding_id},
        )
        assert rows
        return rows[0]["text"]

    def ingest_fragment(
        self,
        *,
        embedding_id: EmbeddingId,
        fragment: CorpusFragment,
    ) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                insert or replace into fragment (
                  fragment_id,
                  embed_id,
                  text,
                  metadata_json
                ) values (
                  :fragment_id,
                  :embed_id,
                  :text,
                  :metadata_json
                )
                """,
                {
                    "fragment_id": fragment.fragment_id,
                    "embed_id": embedding_id,
                    "text": fragment.text,
                    "metadata_json": json.dumps(fragment.metadata),
                },
            )
        return

    def get_fragments_for_embedding(
        self, embedding_id: EmbeddingId
    ) -> list[CorpusFragment]:
        rows = self.db.query(
            """
        select f.fragment_id,
               f.text,
               f.metadata_json
          from fragment f
         where embed_id = :embed_id
        """,
            {
                "embed_id": embedding_id,
            },
        )
        return [
            CorpusFragment(
                fragment_id=row["fragment_id"],
                text=row["text"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def delete_fragment(self, fragment_id: str) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                DELETE
                  FROM fragment
                 WHERE fragment_id = :fragment_id
                """,
                {"fragment_id": fragment_id},
            )
            pass
        return

    def migrate_embedding_id(
        self, *, from_embedding_id: EmbeddingId, to_embedding_id: EmbeddingId
    ) -> None:
        logging.info(
            "migrating embedding %d to %d",
            from_embedding_id,
            to_embedding_id,
        )
        with self.db.cursor() as cur:
            cur.execute(
                """
            INSERT OR REPLACE INTO text_embedding (
              embed_id,
              text
            ) SELECT :to_embedding_id,
                     text
                FROM text_embedding
               WHERE embed_id = :from_embedding_id
            """,
                {
                    "to_embedding_id": to_embedding_id,
                    "from_embedding_id": from_embedding_id,
                },
            )
            cur.execute(
                """
            UPDATE fragment
               SET embed_id = :to_embedding_id
             WHERE embed_id = :from_embedding_id
            """,
                {
                    "to_embedding_id": to_embedding_id,
                    "from_embedding_id": from_embedding_id,
                },
            )
            cur.execute(
                """
            DELETE FROM text_embedding
            WHERE embed_id = :from_embedding_id
            """,
                {
                    "from_embedding_id": from_embedding_id,
                },
            )
            pass
        return

    def compact_embeddings(self) -> list[EmbeddingId]:
        rows = self.db.query(
            """
            SELECT te.embed_id
              FROM text_embedding te
         LEFT JOIN fragment f
                ON f.embed_id = te.embed_id
             WHERE f.fragment_id IS NULL
            """
        )
        embed_ids = [r["embed_id"] for r in rows]
        with self.db.cursor() as cur:
            for embed_id in embed_ids:
                cur.execute(
                    """
                    DELETE FROM text_embedding
                    WHERE embed_id = :embed_id
                    """,
                    {"embed_id": embed_id},
                )
                pass
            pass
        return embed_ids

    def get_all_fragment_ids(self) -> list[str]:
        return [
            r["fragment_id"]
            for r in self.db.query(
                """
            SELECT fragment_id
              FROM fragment
            """
            )
        ]

    pass
