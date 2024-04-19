#!/usr/bin/env python

import os
from setuptools import find_packages, setup

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"

requirements = [
    "numpy",
    "sqlite_vss",
    "torch",
    "transformers",
]

setup(name='babyvec',
      version='v0.1.0',
      description='Natural language embedding tools',
      author='Liam Tengelis',
      author_email='liam@blacktuskdata.com',
      url='https://github.com/lummm/babyvec',
      packages=find_packages(),
      package_data={
          '': ['*.yaml'],
          "pyjobq": ["py.typed"],
      },
      install_requires=requirements,
      extras_require = {
          "http": [
              "uvicorn",
              "fastapi",
          ],
      },
)
