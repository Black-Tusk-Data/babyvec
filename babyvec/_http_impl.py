from contextlib import asynccontextmanager
import logging
from types import SimpleNamespace

from fastapi import FastAPI
from pydantic import BaseModel

from babyvec.common import setup_logging
from babyvec._packaged_providers import CachedParallelJinaEmbedder


class Services(SimpleNamespace):
    embedder: CachedParallelJinaEmbedder


setup_logging()

services = Services()


class GetEmbeddingsInput(BaseModel):
    texts: list[str]


def init_app():
    logging.info("starting babyvec server...")
    services.embedder = CachedParallelJinaEmbedder(
        persist_dir=".babyvec",
        n_computers=1,
        device="cpu",
    )
    logging.info("initialized successfully!")
    return

@asynccontextmanager
async def lifespan(app):
    init_app()
    yield
    logging.info("shutting down babyvec server...")
    services.embedder.shutdown()
    return

app = FastAPI(lifespan=lifespan)

@app.post("/embeddings")
def get_embeddings(body: GetEmbeddingsInput):
    embeddings = services.embedder.get_embeddings(body.texts)
    return [
        embed.tolist()
        for embed in embeddings
    ]
