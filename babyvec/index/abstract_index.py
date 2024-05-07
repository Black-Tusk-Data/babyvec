import abc

from babyvec.models import IndexSearchResult


class AbstractIndex(abc.ABC):
    """
    TODO: support 'adding' to the index on the fly.
    """

    @abc.abstractmethod
    def search(self, query: str, k_nearest: int) -> list[IndexSearchResult]:
        pass
