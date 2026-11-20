from typing import Self

from .common import BaseFuzzyDigest


class BaseUnaryCode(BaseFuzzyDigest):
    """Thermometer encoding for scalar features.

    This class is used to encode integer features
    following the standard unary coding convention.
    """

    @classmethod
    def length_similarity(cls, a: Self, b: Self) -> float:
        """Score higher similarity when unary codes are smaller."""
        a_int = (1 << int(a.popcount())) - 1
        b_int = (1 << int(b.popcount())) - 1
        return 1 / (1 + abs(a_int - b_int) * max(a_int, b_int))

    @classmethod
    def encode(cls, integer: int) -> Self:
        """Convert integer value into thermometer representation.

        Example:
            With cls.bit_size() == 10 and integer == 4
            thermometer encoding is '0b0000001111'
            thus integer is encoded as the value 15

        """
        if integer < 0 or integer > cls.bit_size():
            raise ValueError(
                f"The provided value must fall in range [0-{cls.bit_size()}]"
            )
        thermometer_integer = 2**integer - 1
        return cls(thermometer_integer)

    def similarity(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        other: Self,
        mode="hamming",
    ) -> float:
        if mode == "length":
            return self.length_similarity(self, other)

        return super().similarity(other, mode)
