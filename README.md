# BabyVec
'BabyVec' is intended to provide a simple API around common operations over embeddings of text.

It supports an optional HTTP interface via uvicorn and FastAPI.

Some [examples](./examples) of usage:

1. Building an embedding 'store' for later use in an index.

Code: [./examples/embed_python_documentation.py](./examples/embed_python_documentation.py)

Since it is useful to compute embeddings on a beefy machine for later use on a smaller one, the computation of embeddings is generally treated separately from building an index from those embeddings.

2. Building a searchable embedding 'index'.

Code: [./examples/search_python_documentation.py](./examples/search_python_documentation.py)

Assuming the Python documentation embeddings were created via example 1, this loads them into a searchable index.  This script runs a 'prompt / response' flow where the user's input is searched against the Python documentation.


## Next steps
 - testing at larger volumes of data, particularly when all computed embeddings will not fit in RAM
 - testing on GPU
