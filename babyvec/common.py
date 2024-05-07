import argparse
import dataclasses as dc
import logging
import os
from typing import (
    Callable,
    NamedTuple,
    Type,
    TypeVar,
)
from urllib.parse import quote_plus


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


BaseArgs = NamedTuple
T = TypeVar("T", bound=BaseArgs)


def make_cli_arg_parser(
    *,
    name: str,
    desc: str,
    args_shape: Type[T],
) -> Callable[[], T]:
    type_map = args_shape.__annotations__
    defaults = args_shape._field_defaults

    parser = argparse.ArgumentParser(
        prog=name,
        description=desc,
    )
    for name, dtype in type_map.items():
        cli_argname = name.replace("_", "-")
        kwargs = {
            "dest": name,
            "required": name not in defaults,
            "type": dtype,
        }
        if name in defaults:
            kwargs["default"] = defaults[name]
        parser.add_argument(f"--{cli_argname}", **kwargs)

    def parse_args() -> T:
        args = parser.parse_args()
        return args_shape(**{k: v for k, v in vars(args).items() if v})  # type: ignore

    return parse_args


def setup_logging():
    LOG_FMT = (
        f"%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
        format=LOG_FMT,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    return
