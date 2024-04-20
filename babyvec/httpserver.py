#!/usr/bin/env python3

import os

from babyvec.common import BaseArgs, make_cli_arg_parser


DEFAULT_PORT = 9999


class Args(BaseArgs):
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"


argparser = make_cli_arg_parser(
    name="BabyVec HTTP server",
    desc="HTTP interface into BabyVec",
    args_shape=Args,
)


def main():
    def get_run_cmd(args: Args) -> list[str]:
        runner = "uvicorn"
        app = "babyvec._http_impl:app"
        return [
            runner,
            app,
            "--port",
            str(args.port),
            "--host",
            args.host,
        ]

    args = argparser()
    run_cmd = " ".join(get_run_cmd(args))
    print(run_cmd)
    os.system(run_cmd)
    return


if __name__ == "__main__":
    main()
