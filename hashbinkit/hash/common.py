import struct
from collections.abc import Callable, Hashable, Iterable
from typing import ClassVar, Self, TypeAlias

import numpy as np
from hexio import HexInt, HexStr, HexValue, hexvalue_to_bytes

HashablePrimitive: TypeAlias = str | int | float | bytes | bool | None
HashableElement: TypeAlias = HashablePrimitive | tuple["HashableElement", ...]


def hashable_to_bytes(hashable: Hashable) -> bytes:
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
    # Handle nested Hashable elements by concatenating them
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
def hamming_distance(a: "BaseFuzzyDigest", b: "BaseFuzzyDigest") -> np.uint64:
    return (a ^ b).popcount()


def nand_distance(a: "BaseFuzzyDigest", b: "BaseFuzzyDigest") -> np.uint64:
    """Distance based on NAND ones population."""
    return (~(a & b)).popcount()


def zero_neutral_similarity(
    a: "BaseFuzzyDigest", b: "BaseFuzzyDigest"
) -> float:
    """Hamming similarity based on matching ones only."""
    diff = hamming_distance(a, b)
    union = (a | b).popcount()
    return 1 - diff / union


def jaccard_index(a: "BaseFuzzyDigest", b: "BaseFuzzyDigest") -> float:
    """Bitwise jaccard similarity.

    Treats two zeroes (empty sets) as equal.
    """
    union = a | b
    union_pop = union.popcount()

    # Handle empty sets
    if union_pop == 0:
        return 1.0

    intersection = a & b
    intersection_pop = intersection.popcount()
    return intersection_pop / union_pop


def jaccard_no_empty(a: "BaseFuzzyDigest", b: "BaseFuzzyDigest") -> float:
    """Like jaccard_index but two empty sets are dissimilar."""
    union = a | b
    union_pop = union.popcount()

    # Handle empty sets
    if union_pop == 0:
        return 0.0

    intersection = a & b
    intersection_pop = intersection.popcount()
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

_BIT_MASKS = np.array([1 << i for i in range(8)], dtype=np.uint8)


class BaseFuzzyDigest:
    """An abstract similarity hash.

    When compared with another hash of the same type it
    returns a distance score between [0, BIT_SIZE].

    A similarity score carries the opposite meaning and it's a
    floating value between [0, 1] such that:
    similarity = 1 - distance / BIT_SIZE
    """

    BIT_SIZE: ClassVar[int] = 0

    @classmethod
    def _digest_to_array(
        cls, digest: HexValue | np.ndarray | Self
    ) -> np.ndarray:
        if isinstance(digest, BaseFuzzyDigest):
            return digest.bytes_array.copy()

        if isinstance(digest, np.ndarray):
            np_bytes = np.array(digest, dtype=np.uint8)
        else:
            digest_bytes = hexvalue_to_bytes(digest)  # type: ignore
            np_bytes = np.frombuffer(digest_bytes, dtype=np.uint8)

        expected_size = cls.byte_size()
        if np_bytes.size == expected_size:
            return np_bytes
        elif np_bytes.size < expected_size:
            offset = expected_size - np_bytes.size
            return np.pad(np_bytes, (offset, 0), mode="constant")
        else:
            raise ValueError(
                f"Value '{digest!r}' does not fit in a {cls.__name__}"
                f" digest ({np_bytes.size}B > {expected_size}B)"
            )

    @classmethod
    def _ensure_bit_size(cls) -> None:
        """Ensure class was extended and BIT_SIZE overridden."""
        if cls.BIT_SIZE == 0:
            raise NotImplementedError(
                "You must extend the BaseFuzzyDigest class"
                " and declare a BIT_SIZE > 0."
            )
        elif cls.BIT_SIZE % 8 != 0:
            raise ValueError(
                "You must extend the BaseFuzzyDigest class"
                " with a BIT_SIZE divisible by 8."
            )

    def __init__(self, digest: HexValue | np.ndarray | Self = 0) -> None:
        self._ensure_bit_size()
        self.bytes_array = self._digest_to_array(digest)

    def __len__(self):
        return self.bit_size()

    def __bytes__(self) -> bytes:
        return bytes(self.bytes_array)

    def __str__(self) -> str:
        return HexStr(bytes(self))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self!s}")'

    def __invert__(self) -> Self:
        return self.__class__(~self.bytes_array)

    def __eq__(self, value: object) -> bool:
        try:
            return np.array_equal(
                self.bytes_array,
                self._digest_to_array(value),  # type: ignore
            )
        except ValueError:
            return False

    def __or__(self, value: HexValue | np.ndarray | Self) -> Self:
        np_digest = self.bytes_array | self._digest_to_array(value)
        return self.__class__(np_digest)

    def __xor__(self, value: HexValue | np.ndarray | Self) -> Self:
        np_digest = self.bytes_array ^ self._digest_to_array(value)
        return self.__class__(np_digest)

    def __and__(self, value: HexValue | np.ndarray | Self) -> Self:
        np_digest = self.bytes_array & self._digest_to_array(value)
        return self.__class__(np_digest)

    @classmethod
    def from_features(cls, features: Iterable[Hashable]) -> Self:
        raise NotImplementedError(
            "You must extend the BaseFuzzyDigest class and define this method"
        )

    @classmethod
    def mask(cls) -> Self:
        try:
            return cls._MASK
        except AttributeError:
            cls._MASK = cls(np.full(cls.byte_size(), 255, dtype=np.uint8))
            return cls._MASK

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
            byte_size = bit_size >> 3  # bit_size // 8
            bit_modulo = bit_size & 0b111  # bit_size % 8
            if bit_modulo != 0:
                byte_size = byte_size + 1
            cls.BYTE_SIZE = byte_size
        return byte_size

    def popcount(self) -> np.uint64:
        """Count the number of 1-bits in the digest."""
        return np.bitwise_count(self.bytes_array).sum()

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

    @staticmethod
    def get_bit_coordinates(bit_pos: int) -> tuple[int, int]:
        """Return (byte, bit) coordinates of bit position."""
        return (
            bit_pos >> 3,  # bit_pos // 8
            bit_pos & 0b111,  # bit_pos % 8
        )
