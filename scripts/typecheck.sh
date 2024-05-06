#!/bin/bash


lint() {
    python -m pylint --errors-only ./babyvec
}

typecheck_mypy() {
    python -m mypy ./babyvec
}


lint && \
    typecheck_mypy

