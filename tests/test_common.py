import pytest

from hashbinkit.hash import BaseFuzzyDigest


class FuzzyDigest40(BaseFuzzyDigest):
    BIT_SIZE = 40


class TestFuzzyDigest:
    def test_length(self):
        digest40 = FuzzyDigest40(1)
        assert len(digest40) == 40
        assert digest40.bytes_array.size == digest40.byte_size()

    def test_and(self):
        zero_digest40 = FuzzyDigest40()
        full_digest40 = FuzzyDigest40(int("1" * 40, 2))
        assert zero_digest40 & full_digest40 == zero_digest40

        four_digest40 = FuzzyDigest40(4)
        assert four_digest40 & zero_digest40 == zero_digest40
        assert four_digest40 & full_digest40 == four_digest40

    def test_or(self):
        zero_digest40 = FuzzyDigest40()
        full_digest40 = FuzzyDigest40(int("1" * 40, 2))
        assert zero_digest40 | full_digest40 == full_digest40

        four_digest40 = FuzzyDigest40(4)
        assert four_digest40 | zero_digest40 == four_digest40
        assert four_digest40 | full_digest40 == full_digest40

    def test_xor(self):
        zero_digest40 = FuzzyDigest40()
        full_digest40 = FuzzyDigest40(int("1" * 40, 2))
        assert zero_digest40 ^ full_digest40 == full_digest40

        four_digest40 = FuzzyDigest40(4)
        assert four_digest40 ^ zero_digest40 == four_digest40
        assert four_digest40 ^ full_digest40 == ~four_digest40

    def test_popcount(self):
        four_digest40 = FuzzyDigest40(4)
        full_digest40 = FuzzyDigest40().mask()
        assert four_digest40.popcount() == 1
        assert full_digest40.popcount() == 40

    def test_distance(self):
        zero_digest40 = FuzzyDigest40()
        one_digest40 = FuzzyDigest40(1)
        assert zero_digest40.distance(one_digest40, "hamming") == 1
        assert zero_digest40.distance(one_digest40) == 1

        with pytest.raises(NotImplementedError):
            assert zero_digest40.distance(one_digest40, "notimplemented") == 1
