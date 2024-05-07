from collections import deque
from typing import Iterable

from babyvec.chunker.abstract_chunker import AbstractChunker


class RollingWindowChunker(AbstractChunker):
    def __init__(self, *, window_size: int, overlap: int, delimiter: str = " "):
        super().__init__(delimiter)
        assert overlap < window_size, "overlap must be less than window_size"
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap
        return

    def chunkify_document(self, document: str) -> Iterable[str]:
        return self.chunkify_stream(document)

    def chunkify_stream(self, iterable: Iterable[str]) -> Iterable[str]:
        stream = iter(iterable)
        window: deque[str]
        try:
            window = deque([next(stream) for _ in range(self.window_size)])
        except StopIteration:
            yield self.delimiter.join(iterable)
            return

        yield self.delimiter.join(window)
        i = 0
        for item in stream:
            i += 1
            window.popleft()
            window.append(item)
            if i == self.step_size:
                yield self.delimiter.join(window)
                i = 0

        if i > 0:
            while i < self.step_size:
                window.popleft()
                i += 1
            yield self.delimiter.join(window)
        return
