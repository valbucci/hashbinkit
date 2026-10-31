"""Utilities to build SimHash digests.

Extend BaseSimHash and define custom sub-class variable BIT_SIZE
to implement a SimHash of chosen size.

See SimHash64 class for usage example.
"""

from collections.abc import Iterable
from hashlib import shake_128
from typing import Self

from hexio import HexInt

from .common import BaseFuzzyDigest, HashableElement, hashable_to_bytes


class SimHashAccumulator:
    """Accumulator utilities for SimHash generation."""

    def __init__(self, size: int) -> None:
        """Create empty accumulator.

        Args:
            size (int): The size of the accumulator vector
                and the bit size of the final SimHash digest.

        """
        self.size = size
        self.vector = [0] * size

    def add(self, value: int) -> None:
        """Insert value bits to the accumulator.

        From left to right, the value bits are
        included in the accumulator vector such that:
        if bits[i] == 1 --> vector[i] += 1
        if bits[i] == 0 --> vector[i] -= 1
        """
        # range reversed to work from left to right
        # i.e. '0b10' is processed as 1 then 0 for vector of size 2
        # otherwise it will stream 0, 0, 1, 0 for a vector of size 4
        for i in reversed(range(self.size)):
            bit_at_loc = (value >> i) & 1
            if bit_at_loc == 1:
                self.vector[i] += 1
            else:
                self.vector[i] -= 1

    def collapse(self) -> int:
        """Collapse accumulator vector.

        Positive and negative slots in the accumulator vector
        are represented as ones and zeroes, respectively.
        """
        final_bits = "".join(
            ["1" if value >= 0 else "0" for value in self.vector]
        )
        return int(final_bits, 2)


class BaseSimHash(BaseFuzzyDigest):
    """Abstract SimHash class of undefined size."""

    @classmethod
    def hash_feature(cls, feature: HashableElement) -> int:
        """Hash feature into digest of given bit size."""
        feature_bytes = hashable_to_bytes(feature)
        feature_hash = shake_128(feature_bytes, usedforsecurity=False).digest(
            length=cls.byte_size()
        )
        # NOTE: cls.mask caps the digest for fuzzy hashes
        #       with bit size not divisible by 8 (e.g. 37)
        # This may be unnecessary if accumulator discards
        # extra bits already
        return HexInt.from_bytes(feature_hash) & cls.mask()

    @classmethod
    def from_features(cls, features: Iterable[HashableElement]) -> Self:
        """Instantiate a SimHash digest from hashable features."""
        accumulator = SimHashAccumulator(cls.bit_size())

        for feature in features:
            feature_hash = cls.hash_feature(feature)
            accumulator.add(feature_hash)

        simhash = accumulator.collapse()
        return cls(simhash)
