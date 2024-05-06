from typing import Iterable

from nltk.tokenize import sent_tokenize  # type: ignore

from babyvec.chunker.abstract_chunker import AbstractChunker


class RollingWindowSentenceChunker(AbstractChunker):
    def __init__(self, *, window_size: int, overlap: int, delimiter: str = " "):
        super().__init__(delimiter)
        assert overlap < window_size, "overlap must be less than window_size"
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap
        return

    def chunkify_document(self, document: str) -> Iterable[str]:
        sentences = sent_tokenize(document)
        for i in range(0, len(sentences), self.step_size):
            chunk = sentences[i : i + self.window_size]
            yield self.delimiter.join(chunk)
        return
