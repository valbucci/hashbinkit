from collections.abc import Iterable
from hashlib import shake_128
from typing import Self

from hexio import HexInt, HexValue, hexvalue_to_int

from .common import (
    BaseFuzzyDigest,
    HashableElement,
    hashable_to_bytes,
    jaccard_index,
)

HASH_BYTES = 8
BLOOM_MAX_BITSIZE = 8 * HASH_BYTES  # maximum supported value for bloom filter


class BaseBloomFilter(BaseFuzzyDigest):
    @classmethod
    def _coerce_filter(cls, value: HexValue) -> int:
        other_int = hexvalue_to_int(value)
        if other_int.bit_length() > cls.BIT_SIZE:
            raise ValueError(
                f"Size mismatch between value '{value}'"
                f" of size {other_int.bit_length()}"
                f" and bloom filter of size {cls.BIT_SIZE}"
            )

        return other_int

    @classmethod
    def hash_feature(cls, feature: HashableElement) -> int:
        """Hash feature into bloom filter position.

        Returned position is integer between [0, BIT_SIZE - 1].
        """
        feature_bytes = hashable_to_bytes(feature)
        feature_hash = shake_128(feature_bytes, usedforsecurity=False).digest(
            length=cls.byte_size()
        )
        feature_hash_int = HexInt.from_bytes(feature_hash)
        # Now compress into random bloom filter position
        # with modulo operation
        return feature_hash_int % cls.bit_size()

    @classmethod
    def from_features(cls, features: Iterable[HashableElement]) -> Self:
        new_bloom = cls(0)
        for feature in features:
            feature_hash_position = cls.hash_feature(feature)
            new_bloom |= 1 << feature_hash_position
        return new_bloom

    def intersection(self, other: HexValue) -> Self:
        other_coerced = self._coerce_filter(other)
        return self & other_coerced

    def union(self, other: HexValue) -> Self:
        other_coerced = self._coerce_filter(other)
        return self | other_coerced

    def jaccard_index(self, other: HexValue) -> float:
        other_coerced = self._coerce_filter(other)
        return jaccard_index(self, other_coerced)
