from unittest import TestCase

from babyvec.chunker.rolling_window_chunker import RollingWindowChunker


def L():
    for c in ["a", "b", "c", "d", "e", "f", "g"]:
        yield c


def apply_chunk_params(**kwargs):
    return list(RollingWindowChunker(**kwargs).chunkify_stream(L()))


class RollingWindowChunker_Test(TestCase):
    def test_chunkify_stream(self):
        self.assertEqual(
            apply_chunk_params(
                window_size=1,
                overlap=0,
            ),
            ["a", "b", "c", "d", "e", "f", "g"],
        )
        self.assertEqual(
            apply_chunk_params(
                window_size=2,
                overlap=0,
            ),
            ["a b", "c d", "e f", "g"],
        )
        self.assertEqual(
            apply_chunk_params(
                window_size=3,
                overlap=1,
            ),
            ["a b c", "c d e", "e f g"],
        )
        self.assertEqual(
            apply_chunk_params(
                window_size=5,
                overlap=3,
            ),
            ["a b c d e", "c d e f g"],
        )
        self.assertEqual(
            apply_chunk_params(
                window_size=5,
                overlap=4,
            ),
            ["a b c d e", "b c d e f", "c d e f g"],
        )

        return
