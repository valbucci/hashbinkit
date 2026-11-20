"""Utilities to build SimHash digests.

Extend BaseSimHash and define custom sub-class variable BIT_SIZE
to implement a SimHash of chosen size.

See SimHash64 class for usage example.
"""

from collections.abc import Hashable, Iterable
from hashlib import shake_128
from typing import Self

import numpy as np
from numba import float64, uint8, uint32
from numba.experimental.jitclass.decorators import jitclass

from .common import BaseFuzzyDigest, hashable_to_bytes

accumulator_spec = [
    ("size", uint32),  # for the bit size (allowing huge SimHash)
    ("vector", float64[:]),  # for the accumulator vector
    ("collapsed_bytes", uint8[:]),  # for the collapsed bits as bytes
]


@jitclass(accumulator_spec)
class SimHashAccumulator:
    """Accumulator utilities for SimHash generation."""

    def __init__(self, size: int) -> None:
        """Create empty accumulator.

        Args:
            size (int): The size of the accumulator vector
                and the bit size of the final SimHash digest.

        """
        self.size = size
        self.vector = np.zeros(size, dtype=np.float64)
        byte_size = size // 8
        if size % 8 > 0:
            byte_size += 1
        self.collapsed_bytes = np.zeros(byte_size, dtype=np.uint8)

    def add(self, feature_hash: bytes, weight: float = 1.0) -> None:
        """Insert value bits to the accumulator.

        From left to right, the value bits are
        included in the accumulator vector such that:
        if bits[i] == 1 --> vector[i] += 1
        if bits[i] == 0 --> vector[i] -= 1
        """
        # range reversed to work from left to right
        # i.e. 0b1000 is processed as 1, 0, 0, 0
        bit_index = 0
        bit_offset = 7

        for byte_value in feature_hash:
            for _ in range(8):
                bit = (byte_value >> bit_offset) & 1

                if bit:
                    self.vector[bit_index] += weight
                else:
                    self.vector[bit_index] -= weight

                bit_index += 1
                bit_offset -= 1

            bit_offset = 7

    def collapse(self) -> np.ndarray:
        """Collapse accumulator vector.

        Positive and negative slots in the accumulator vector
        are represented as ones and zeroes, respectively.
        """
        # Basically do the same as np.packbits
        # using big endian.
        # Better than np.packbits because fixes
        # issues where values are not divisible by 8
        self.collapsed_bytes[:] = 0

        padding = len(self.collapsed_bytes) * 8 - self.size

        byte_index = padding // 8
        bit_offset = 7 - (padding % 8)

        for bit_index in range(self.size):
            if self.vector[bit_index] >= 0:
                self.collapsed_bytes[byte_index] |= 1 << bit_offset

            bit_offset -= 1

            if bit_offset < 0:
                bit_offset = 7
                byte_index += 1

        return self.collapsed_bytes


class BaseSimHash(BaseFuzzyDigest):
    """Abstract SimHash class of undefined size."""

    @classmethod
    def hash_feature(cls, feature: Hashable) -> bytes:
        """Hash feature into digest of given bit size."""
        feature_bytes = hashable_to_bytes(feature)
        return shake_128(feature_bytes, usedforsecurity=False).digest(
            length=cls.byte_size()
        )

    @classmethod
    def from_features(cls, features: Iterable[Hashable]) -> Self:
        """Instantiate a SimHash digest from hashable features."""
        accumulator = SimHashAccumulator(cls.bit_size())

        for feature in features:
            feature_hash = cls.hash_feature(feature)
            accumulator.add(feature_hash)

        return cls(accumulator.collapse())
