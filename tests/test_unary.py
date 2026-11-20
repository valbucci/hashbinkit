import pytest

from hashbinkit.hash import BaseUnaryCode


class UnaryCode64(BaseUnaryCode):
    BIT_SIZE = 64


class TestUnaryCode:
    negative = -1
    out_of_bound = UnaryCode64.bit_size() + 1
    valid = 12

    @property
    def expected_int(self) -> int:
        """Return expected thermometer from valid input."""
        bit_str = "1" * self.valid
        return int(bit_str, base=2)

    @property
    def thermometer(self) -> UnaryCode64:
        return UnaryCode64.encode(self.valid)

    def test_from_integer(self):
        """Test values can be generated from integer."""
        assert self.thermometer == self.expected_int
        with pytest.raises(ValueError):
            UnaryCode64.encode(self.negative)
        with pytest.raises(ValueError):
            UnaryCode64.encode(self.out_of_bound)

        assert UnaryCode64.encode(self.valid).popcount() == self.valid

    def test_similarity_importance(self):
        """Test similarity importance based on value.

        For three values A, B, and C where: A + 2 = B + 1 = C ;
        We expect similarity(A, B) > similarity(B, C)
        """
        therm_a = UnaryCode64.encode(self.valid)
        therm_b = UnaryCode64.encode(self.valid + 1)
        therm_c = UnaryCode64.encode(self.valid + 2)
        assert therm_a.similarity(therm_b, "length") > therm_b.similarity(
            therm_c, "length"
        )

    def test_distance_difference(self):
        """Test that hamming distance is like difference."""
        therm_a = UnaryCode64.encode(32)
        therm_b = UnaryCode64.encode(17)
        assert therm_a.distance(therm_b) == 32 - 17
