#!/bin/bash

TOKENIZERS_PARALLELISM=true find ./babyvec -name '*_test.py' | xargs python -m unittest
