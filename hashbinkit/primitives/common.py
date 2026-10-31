import struct
from collections.abc import Iterable
from typing import Callable, ClassVar, Self, TypeAlias, Union

from hexio import HexInt, HexStr, HexValue

HashablePrimitive: TypeAlias = Union[str, int, float, bytes, bool, None]
HashableElement: TypeAlias = Union[
    HashablePrimitive, tuple["HashableElement", ...]
]


def hashable_to_bytes(hashable: HashableElement) -> bytes:
    # Handle HashablePrimitive types
    if isinstance(hashable, bytes):
        return hashable
    elif isinstance(hashable, str):
        return hashable.encode("utf-8")
    elif isinstance(hashable, int):
        return bytes(HexInt(hashable))
    elif isinstance(hashable, bool):
        return int(hashable).to_bytes(1, "big")
    elif isinstance(hashable, float):
        return struct.pack("d", hashable)
    elif hashable is None:
        return b""
    # Handle nested HashableElement by concatenating them
    # Use delimiter `,` and separators `(` and `)`
    # to distinguish tuples from lone primitives
    elif isinstance(hashable, tuple):
        flat_hashable = b"("
        for element in hashable:
            byte_element = hashable_to_bytes(element)
            flat_hashable += byte_element + b","
        return flat_hashable[:-1] + b")"
    else:
        raise NotImplementedError(
            f"The type {type(hashable)} is not supported as a HashableElement"
        )


# region SIMILARITY / DISTANCE
def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def nand_distance(a: int, b: int) -> int:
    """Distance based on NAND ones population."""
    return (~(a & b)).bit_count()


def zero_neutral_similarity(a: int, b: int) -> float:
    """Hamming similarity based on matching ones only."""
    diff = hamming_distance(a, b)
    union = (a | b).bit_count()
    return 1 - diff / union


def jaccard_index(a: int, b: int) -> float:
    """Bitwise jaccard similarity.

    Treats two zeroes (empty sets) as equal.
    """
    union = a | b
    union_pop = union.bit_count()

    # Handle empty sets
    if union_pop == 0:
        return 1.0

    intersection = a & b
    intersection_pop = intersection.bit_count()
    return intersection_pop / union_pop


def jaccard_no_empty(a: int, b: int) -> float:
    """Like jaccard_index but two empty sets are dissimilar."""
    union = a | b
    union_pop = union.bit_count()

    # Handle empty sets
    if union_pop == 0:
        return 0.0

    intersection = a & b
    intersection_pop = intersection.bit_count()
    return intersection_pop / union_pop


# endregion SIMILARITY / DISTANCE

DISTANCE_FUNCS: dict[str, Callable] = {
    # mode -> distance_function(a: int, b: int) -> int
    "hamming": hamming_distance,
    "nand": nand_distance,
}
SIMILARITY_FUNCS: dict[str, Callable] = {
    # mode -> similarity_function(a: int, b: int) -> float
    "jaccard": jaccard_index,
    "jaccard_ne": jaccard_no_empty,
    "zero_neutral": zero_neutral_similarity,
}


class BaseFuzzyDigest(HexInt):
    """An abstract similarity hash.

    When compared with another hash of the same type it
    returns a distance score between [0, BIT_SIZE].

    A similarity score carries the opposite meaning and it's a
    floating value between [0, 1] such that:
    similarity = 1 - distance / BIT_SIZE
    """

    BIT_SIZE: ClassVar[int] = 0

    def __new__(cls, value: HexValue = 0) -> Self:
        cls._ensure_bit_size()
        obj = super().__new__(cls, value)
        numeric_value = int(obj)

        if numeric_value >= 1 << cls.BIT_SIZE:
            raise ValueError(
                f"Value does not fit in a {cls.BIT_SIZE}-bit digest"
            )

        return obj

    def __len__(self):
        return self.BIT_SIZE

    def __str__(self) -> str:
        return (HexStr(self) | self.upper_bound())[1:]

    def __invert__(self) -> Self:
        return self.__class__(self.mask() & ~int(self))

    @classmethod
    def _ensure_bit_size(cls) -> None:
        """Ensure class was extended and BIT_SIZE overridden."""
        if cls.BIT_SIZE == 0:
            raise NotImplementedError(
                "You must extend the BaseFuzzyDigest class"
                " and declare a BIT_SIZE > 0."
            )

    @classmethod
    def from_features(cls, features: Iterable[HashableElement]) -> Self:
        raise NotImplementedError(
            "You must extend the BaseFuzzyDigest class and define this method"
        )

    @classmethod
    def upper_bound(cls) -> HexInt:
        return HexInt(1 << cls.BIT_SIZE)

    @classmethod
    def mask(cls) -> Self:
        return cls(cls.upper_bound() - 1)

    @property
    def hexstr(self) -> HexStr:
        return HexStr(str(self))

    @property
    def binstr(self) -> str:
        bounded_value = self.upper_bound() | self
        return bin(bounded_value)[3:]

    @classmethod
    def bit_size(cls) -> int:
        cls._ensure_bit_size()
        return cls.BIT_SIZE

    @classmethod
    def byte_size(cls) -> int:
        try:
            return cls.BYTE_SIZE
        except AttributeError:
            bit_size = cls.bit_size()
            byte_size = bit_size // 8
            if bit_size % 8 != 0:
                byte_size = byte_size + 1
            cls.BYTE_SIZE = byte_size
        return byte_size

    def popcount(self) -> int:
        """Alias for int.bit_count."""
        return self.bit_count()

    def distance(self, other: Self, mode: str = "hamming") -> int:
        try:
            distance_func = DISTANCE_FUNCS[mode]
            return distance_func(self, other)
        except KeyError:
            raise NotImplementedError(
                f"The distance mode '{mode}' is not implemented."
            )

    def similarity(self, other: Self, mode: str = "hamming") -> float:
        if mode in SIMILARITY_FUNCS:
            similarity_func = SIMILARITY_FUNCS[mode]
            return similarity_func(self, other)
        elif mode in DISTANCE_FUNCS:
            # Reverse distance
            distance = self.distance(other, mode)
            return 1 - distance / self.BIT_SIZE
        else:
            raise NotImplementedError(
                f"The similarity mode '{mode}' is not implemented."
            )
