#!/bin/bash


lint() {
    python -m pylint --errors-only ./babyvec
}

typecheck_mypy() {
    python -m mypy  --check-untyped-defs ./babyvec
}

typecheck_pyright() {
    python -m pyright ./babyvec
}


lint && \
    typecheck_mypy && \
    typecheck_pyright

