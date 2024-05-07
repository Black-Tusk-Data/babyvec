import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI
from pydantic import BaseModel

from babyvec.common import BaseArgs
from babyvec.computer.embedding_computer_jina_bert import EmbeddingComputerJinaBert
from babyvec.embed_provider.parallelized_cached_embed_provider import (
    ParallelizedCachedEmbedProvider,
)
from babyvec.models import EmbedComputeOptions, PersistenceOptions
from babyvec.store.embedding_store_numpy import EmbeddingStoreNumpy
from babyvec.store.metadata_store_sqlite import MetadataStoreSQLite

DEFAULT_PORT = 9999
DEFAULT_HOST = "127.0.0.1"


class HttpServerArgs(BaseArgs):
    port: int = DEFAULT_PORT
    host: str = DEFAULT_HOST
    n_computers: int = 1
    device: str = "cpu"
    persist_dir: str = "./.babyvec"


class Services(SimpleNamespace):
    embedding_provider: ParallelizedCachedEmbedProvider
    pass


services = Services()


class GetEmbeddingsInput(BaseModel):
    texts: list[str]


def build_app(args: HttpServerArgs):
    def init_app():
        logging.info("starting babyvec server...")
        persist_options = PersistenceOptions(
            persist_dir=args.persist_dir,
        )
        services.embedding_provider = ParallelizedCachedEmbedProvider(
            n_computers=args.n_computers,
            compute_options=EmbedComputeOptions(
                device=args.device,
            ),
            computer_type=EmbeddingComputerJinaBert,
            store=EmbeddingStoreNumpy(
                persist_options=persist_options,
                metadata_store=MetadataStoreSQLite(persist_options),
            ),
        )
        logging.info("initialized successfully!")
        return

    @asynccontextmanager
    async def lifespan(app):
        init_app()
        yield
        logging.info("shutting down babyvec server...")
        services.embedding_provider.shutdown()
        return

    app = FastAPI(lifespan=lifespan)

    @app.post("/embeddings")
    def get_embeddings(body: GetEmbeddingsInput):
        embeddings = services.embedding_provider.get_embeddings(body.texts)
        return [embed.tolist() for embed in embeddings]

    return app
