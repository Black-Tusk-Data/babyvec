#!/usr/bin/env python3

import uvicorn

from babyvec._http_impl import build_app, HttpServerArgs
from babyvec.common import make_cli_arg_parser, setup_logging

setup_logging()


argparser = make_cli_arg_parser(
    name="BabyVec HTTP server",
    desc="HTTP interface into BabyVec",
    args_shape=HttpServerArgs,
)


def main():
    args = argparser()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
    return


if __name__ == "__main__":
    main()
