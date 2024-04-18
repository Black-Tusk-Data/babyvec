CREATE TABLE IF NOT EXISTS fragment (
  id  TEXT  NOT NULL  PRIMARY KEY,
  text  TEXT  NOT NULL,
  embed_id  INTEGER  NOT NULL
);

CREATE VIRTUAL TABLE chunk_embed USING vss0(
  -- integer key 'rowid' automatically defined
  embedding(768)
);
