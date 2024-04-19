#!/usr/bin/env python

from setuptools import find_packages, setup


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
      extras_require = {
          "http": [
              "uvicorn",
              "fastapi",
          ],
      },
)
