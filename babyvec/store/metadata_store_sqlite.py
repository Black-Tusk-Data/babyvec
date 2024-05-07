import json
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

    def add_fragment(
        self,
        *,
        embedding_id: EmbeddingId,
        fragment: CorpusFragment,
    ) -> str:
        fragment_id = str(uuid4())
        with self.db.cursor() as cur:
            cur.execute(
                """
                insert into fragment (
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
                    "fragment_id": fragment_id,
                    "embed_id": embedding_id,
                    "text": fragment.text,
                    "metadata_json": json.dumps(fragment.metadata),
                },
            )
        return fragment_id

    def get_fragments_for_embedding(
        self, embedding_id: EmbeddingId
    ) -> list[CorpusFragment]:
        rows = self.db.query(
            """
        select f.text,
               f.metadata_json
          from fragment f
         where embed_id = :embed_id
        """,
            {
                "embed_id": embedding_id,
            },
        )
        return [
            CorpusFragment(text=row["text"], metadata=json.loads(row["metadata_json"]))
            for row in rows
        ]
