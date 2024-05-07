import contextlib
import os
import sqlite3


class SQLiteDB:
    def __init__(
        self,
        *,
        dbfile_path: str,
        schema: str,
    ):
        is_fresh = not os.path.exists(dbfile_path)

        self.dbcon = sqlite3.connect(dbfile_path)
        self.dbcon.row_factory = sqlite3.Row

        if is_fresh:
            with self.cursor() as cur:
                cur.executescript(schema)
        return

    @contextlib.contextmanager
    def cursor(self, autocommit=True):
        cursor = self.dbcon.cursor()
        try:
            yield cursor
            if autocommit:
                self.dbcon.commit()
        finally:
            cursor.close()
        return

    def query(self, *args, **kwargs) -> list[sqlite3.Row]:
        with self.cursor() as cur:
            cur.execute(*args, **kwargs)
            return cur.fetchall()
        pass
