"""Encode and manage sequence artefacts."""

from collections import Counter
from collections.abc import Hashable, Iterable, Iterator


class Sequence:
    __slots__ = "counter", "elements", "size"

    def __init__(self, elements: Iterable[Hashable]) -> None:
        """Init a sequence with given elements.

        The sequence elements will be stored following
        the provided order and the frequency count of
        each element will be retained.
        """
        self.setup_elements(elements)

    def setup_elements(self, elements: Iterable[Hashable]) -> None:
        """Set sequence elements, cache size and count frequency."""
        self.elements = tuple(elements)
        self.size = len(self.elements)
        self.counter = Counter(self.elements)

    def get_indexed(self) -> Iterator[tuple[int, Hashable]]:
        """Iterate enumerated elements."""
        yield from enumerate(self.elements)

    def get_n_grams(self, n: int) -> Iterator[tuple[Hashable, ...]]:
        """Iterate n-grams over the ordered sequence."""
        start = 0
        end = n
        while end <= self.size:
            yield self.elements[start:end]
            start += 1
            end += 1
        if n > self.size:
            yield self.elements

    def get_n_grams_sorted(self, n: int) -> Iterator[tuple[Hashable, ...]]:
        """Iterate sorted n-grams over the ordered sequence."""
        for n_gram in self.get_n_grams(n):
            yield tuple(sorted(n_gram))

    def get_frequency(self) -> Iterator[tuple[Hashable, int]]:
        """Iterate each unique element and its frequency."""
        yield from self.counter.items()

    def get_frequency_sorted(self) -> list[tuple[Hashable, int]]:
        """Return sorted list of unique elements and frequency.

        Sorts by frequency first (descending)
        and then by name (ascending).
        """
        return sorted(self.get_frequency(), key=lambda x: (-x[1], x[0]))
