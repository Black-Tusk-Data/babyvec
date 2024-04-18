import dataclasses as dc
from urllib.parse import quote_plus
import os


@dc.dataclass
class FileRef:
    orig_name: str
    sanitized_name: str
    abspath: str
    absdir: str
    ext: str

    @staticmethod
    def _sanitize_fname(fname: str):
        return quote_plus(fname)

    @staticmethod
    def parse(file_path: str) -> "FileRef":
        rel_name, ext = os.path.splitext(file_path)
        ext = ext[1:]
        orig_name = os.path.basename(rel_name)
        absdir = os.path.abspath(
            os.path.dirname(rel_name),
        )
        abspath = f"{os.path.join(absdir, orig_name)}.{ext}"
        return FileRef(
            sanitized_name=FileRef._sanitize_fname(os.path.basename(rel_name)),
            orig_name=orig_name,
            abspath=abspath,
            absdir=absdir,
            ext=ext,
        )
