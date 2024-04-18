import os
import sqlite3

import sqlite_vss

from vectorspace.common import FileRef


class PersistenceManager:
    def __init__(
            self,
            storage_path: FileRef,
    ):
        self._storage_path = storage_path
        return

    def persist_to_disk(self, db: sqlite3.Connection):
        to_db = sqlite3.connect(self._storage_path)
        db.backup(to_db)
        return

    def load_from_disk(self) -> sqlite3.Connection:
        mem_db = sqlite3.connect(":memory:")
        mem_db.enable_load_extension(True)
        sqlite_vss.load(mem_db)

        if os.path.exists(self._storage_path):
            disk_db = sqlite3.connect(self._storage_path)
            disk_db.enable_load_extension(True)
            sqlite_vss.load(disk_db)
            disk_db.backup(mem_db)
            return mem_db

        # o/w must build and define schema
        schema_file = os.path.join(
            FileRef.parse(__file__).absdir,
            "schema.sql",
        )
        with open(schema_file, "r") as f:
            mem_db
        return
