#!/usr/bin/env python

import os
from setuptools import find_packages, setup


lib_folder = os.path.dirname(os.path.realpath(__file__))


def read_requirements_txt():
    with open(os.path.join(lib_folder, "requirements.txt")) as f:
        return [line.strip() for line in f.readlines() if line.strip()]
    pass


requirements = [
    *read_requirements_txt(),
]

setup(
    name="babyvec",
    version="v0.2.1",
    description="Natural language embedding tools",
    author="Liam Tengelis",
    author_email="liam@blacktuskdata.com",
    url="https://github.com/lummm/babyvec",
    packages=find_packages(),
    package_data={
        "": ["*.yaml"],
        "babyvec": ["py.typed"],
    },
    install_requires=requirements,
    extras_require={
        "http": [
            "uvicorn",
            "fastapi",
            "pydantic",
        ],
    },
)
