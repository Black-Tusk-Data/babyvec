from contextlib import asynccontextmanager
import logging
from types import SimpleNamespace

from babyvec.store.abstract_embedding_store import EmbeddingPersistenceOptions
from babyvec.store.embedding_store_numpy import EmbeddingStoreNumpy
from fastapi import FastAPI
from pydantic import BaseModel

from babyvec.common import BaseArgs


DEFAULT_PORT = 9999
DEFAULT_HOST = "127.0.0.1"


class HttpServerArgs(BaseArgs):
    port: int = DEFAULT_PORT
    host: str = DEFAULT_HOST
    n_computers: int = 1
    device: str = "cpu"
    persist_dir: str = "./.babyvec"


class Services(SimpleNamespace):
    # embedder: CachedParallelJinaEmbedder
    pass


services = Services()


class GetEmbeddingsInput(BaseModel):
    texts: list[str]


def build_app(args: HttpServerArgs):
    def init_app():
        logging.info("starting babyvec server...")
        # services.embedder = CachedParallelJinaEmbedder(
        #     n_computers=args.n_computers,
        #     persist_dir=args.persist_dir,
        #     device=args.device,
        # )
        logging.info("initialized successfully!")
        return

    @asynccontextmanager
    async def lifespan(app):
        init_app()
        yield
        logging.info("shutting down babyvec server...")
        # services.embedder.shutdown()
        return

    app = FastAPI(lifespan=lifespan)

    @app.post("/embeddings")
    def get_embeddings(body: GetEmbeddingsInput):
        return []
        # embeddings = services.embedder.get_embeddings(body.texts)
        # return [embed.tolist() for embed in embeddings]

    return app
