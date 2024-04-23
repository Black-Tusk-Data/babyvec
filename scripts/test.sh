#!/bin/bash

find ./babyvec -name '*_test.py' | xargs python -m unittest
