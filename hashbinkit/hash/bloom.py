from collections.abc import Hashable, Iterable
from hashlib import shake_128
from typing import Self

import numpy as np
from hexio import HexValue

from .common import (
    _BIT_MASKS,
    BaseFuzzyDigest,
    hashable_to_bytes,
    jaccard_index,
)

HASH_BYTES = 8
BLOOM_MAX_BITSIZE = 8 * HASH_BYTES  # maximum supported value for bloom filter


class BaseBloomFilter(BaseFuzzyDigest):
    @classmethod
    def _coerce_filter(cls, value: HexValue | np.ndarray | Self) -> Self:
        other_digest = cls(value)
        if other_digest.BIT_SIZE > cls.BIT_SIZE:
            raise ValueError(
                f"Size mismatch between value '{value}'"
                f" of size {other_digest.BIT_SIZE}"
                f" and bloom filter of size {cls.BIT_SIZE}"
            )

        return other_digest

    @classmethod
    def hash_feature(cls, feature: Hashable, n_pos=1) -> np.ndarray:
        """Hash feature into bloom filter positions.

        Returned positions are np.uint64 between [0, BIT_SIZE - 1].
        """
        feature_bytes = hashable_to_bytes(feature)
        feature_hash = shake_128(feature_bytes, usedforsecurity=False).digest(
            length=n_pos * 8
        )
        positions = np.frombuffer(feature_hash, dtype=np.uint64, count=n_pos)
        return positions % cls.BIT_SIZE

    @classmethod
    def from_features(cls, features: Iterable[Hashable], n_pos=1) -> Self:
        new_bloom = cls(0)
        positions = np.concatenate(
            [cls.hash_feature(feature, n_pos) for feature in features]
        )
        byte_idxs = positions >> 3  # positions // 8
        bit_idxs = positions & 0b111  # positions % 8
        masks = _BIT_MASKS[bit_idxs]
        np.bitwise_or.at(new_bloom.bytes_array, byte_idxs, masks)
        return new_bloom

    def intersection(self, other: HexValue | np.ndarray | Self) -> Self:
        other_coerced = self._coerce_filter(other)
        return self & other_coerced

    def union(self, other: HexValue | np.ndarray | Self) -> Self:
        other_coerced = self._coerce_filter(other)
        return self | other_coerced

    def jaccard_index(self, other: HexValue | np.ndarray | Self) -> float:
        other_coerced = self._coerce_filter(other)
        return jaccard_index(self, other_coerced)
