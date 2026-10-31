import pytest

from hashbinkit.primitives import BaseBloomFilter


class BloomFilter64(BaseBloomFilter):
    BIT_SIZE = 64


class BloomFilter37(BaseBloomFilter):
    BIT_SIZE = 37


class TestBloomFilter:
    def test_from_features(self):
        test_features1 = (1, 2)
        test_features2 = (2, 3)
        bf1 = BloomFilter64.from_features(test_features1)
        bf2 = BloomFilter64.from_features(test_features2)
        assert bf1.intersection(bf2).popcount() == 1
        assert bf1.union(bf2).popcount() == 3
        assert bf1.jaccard_index(bf2) == 1 / 3
        assert bf1.jaccard_index(bf2) == bf2.jaccard_index(bf1)
        assert bf1.similarity(bf2, "jaccard") == 1 / 3

    def test_size_mismatch(self):
        test_features1 = (1, 2)
        test_features2 = (2, 3) + tuple(range(5, 10))
        bf1 = BloomFilter37.from_features(test_features1)
        bf2 = BloomFilter64.from_features(test_features2)

        assert bf2.jaccard_index(bf1) > 0
        with pytest.raises(ValueError):
            assert bf1.jaccard_index(bf2) > 0
