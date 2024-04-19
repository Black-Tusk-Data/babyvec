CREATE VIRTUAL TABLE fragment_embed USING vss0(
  -- integer key 'rowid' automatically defined
  embedding(${EMBEDDING_SIZE})
);

CREATE TABLE IF NOT EXISTS fragment (
  id  TEXT  NOT NULL  PRIMARY KEY,
  text  TEXT  NOT NULL  UNIQUE,
  embed_id  INTEGER  NOT NULL,


  FOREIGN KEY (embed_id) REFERENCES fragment_embed (rowid)
    ON DELETE CASCADE
);
