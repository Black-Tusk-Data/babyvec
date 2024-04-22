from transformers import AutoModel
from babyvec.computer.abstract_embedding_computer import AbstractEmbeddingComputer
from babyvec.models import Embedding


DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"


class EmbeddingComputerJinaBert(AbstractEmbeddingComputer):
    def __init__(
            self,
            model_name: str = DEFAULT_MODEL_NAME,
            device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        return
    
    def compute_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self.model.encode(
            texts,
            device=self.device,
        )
