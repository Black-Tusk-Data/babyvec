from contextlib import asynccontextmanager
import logging
from types import SimpleNamespace

from fastapi import FastAPI
from pydantic import BaseModel

from babyvec.common import setup_logging
from babyvec.interface import BabyVecLocalEmbedder


class Services(SimpleNamespace):
    embedder: BabyVecLocalEmbedder


setup_logging()

services = Services()


class GetEmbeddingsInput(BaseModel):
    texts: list[str]


def init_app():
    logging.info("starting babyvec server...")
    services.embedder = BabyVecLocalEmbedder(
        persist_path="./persist/embeds/dat",
        embedding_size=768,
        model="jinaai/jina-embeddings-v2-base-en",
        device="mps",
    )
    logging.info("initialized successfully!")
    return

@asynccontextmanager
async def lifespan(app):
    init_app()
    yield
    logging.info("shutting down babyvec server...")
    services.embedder.close()
    return

app = FastAPI(lifespan=lifespan)

@app.post("/embeddings")
def get_embeddings(body: GetEmbeddingsInput):
    embeddings = services.embedder.get_embeddings(body.texts)
    return [
        embed.tolist()
        for embed in embeddings
    ]


#     return app


# def main():
#     return
