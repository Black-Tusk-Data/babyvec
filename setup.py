#!/usr/bin/env python

import os
from setuptools import find_packages, setup


requirements = [
    "faiss-cpu",
    "nltk",
    "npy_append_array",
    "numpy",
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
          "babyvec": ["py.typed"],
      },
      install_requires=requirements,
      extras_require = {
          "http": [
              "uvicorn",
              "fastapi",
              "pydantic",
          ],
      },
)
