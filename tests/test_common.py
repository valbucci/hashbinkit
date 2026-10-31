import pytest

from hashbinkit.primitives import BaseFuzzyDigest


class FuzzyDigest37(BaseFuzzyDigest):
    BIT_SIZE = 37


class TestFuzzyDigest:
    def test_print(self):
        digest37 = FuzzyDigest37(1)
        binexpected37 = "0" * 36 + "1"
        assert digest37.binstr == binexpected37
        hexexpected37 = "0" * 8 + "1"
        assert digest37.hexstr == hexexpected37

    def test_length(self):
        digest37 = FuzzyDigest37(1)
        assert len(digest37) == 37

    def test_and(self):
        zero_digest37 = FuzzyDigest37()
        full_digest37 = FuzzyDigest37(int("1" * 37, 2))
        assert zero_digest37 & full_digest37 == zero_digest37

        four_digest37 = FuzzyDigest37(4)
        assert four_digest37 & zero_digest37 == zero_digest37
        assert four_digest37 & full_digest37 == four_digest37

    def test_or(self):
        zero_digest37 = FuzzyDigest37()
        full_digest37 = FuzzyDigest37(int("1" * 37, 2))
        assert zero_digest37 | full_digest37 == full_digest37

        four_digest37 = FuzzyDigest37(4)
        assert four_digest37 | zero_digest37 == four_digest37
        assert four_digest37 | full_digest37 == full_digest37

    def test_xor(self):
        zero_digest37 = FuzzyDigest37()
        full_digest37 = FuzzyDigest37(int("1" * 37, 2))
        assert zero_digest37 ^ full_digest37 == full_digest37

        four_digest37 = FuzzyDigest37(4)
        assert four_digest37 ^ zero_digest37 == four_digest37
        assert four_digest37 ^ full_digest37 == ~four_digest37

    def test_popcount(self):
        four_digest37 = FuzzyDigest37(4)
        full_digest37 = FuzzyDigest37().mask()
        assert four_digest37.popcount() == 1
        assert full_digest37.popcount() == 37

    def test_distance(self):
        zero_digest37 = FuzzyDigest37()
        one_digest37 = FuzzyDigest37(1)
        assert zero_digest37.distance(one_digest37, "hamming") == 1
        assert zero_digest37.distance(one_digest37) == 1

        with pytest.raises(NotImplementedError):
            assert zero_digest37.distance(one_digest37, "notimplemented") == 1
