import logging

from transformers import AutoModel  # type: ignore

from babyvec.computer.abstract_embedding_computer import AbstractEmbeddingComputer
from babyvec.models import EmbedComputeOptions, Embedding

DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"


class EmbeddingComputerJinaBert(AbstractEmbeddingComputer):
    def __init__(
        self,
        compute_options: EmbedComputeOptions,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        logging.info(
            "JinaBert embedding computer coming online, with options: %s",
            compute_options,
        )
        super().__init__(compute_options)
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        return

    def compute_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self.model.encode(
            texts,
            device=self.compute_options.device,
        )  # float32
