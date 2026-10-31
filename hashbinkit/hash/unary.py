from typing import Self

from .common import BaseFuzzyDigest


class BaseUnaryCode(BaseFuzzyDigest):
    """Thermometer encoding for scalar features.

    This class is used to encode integer features
    following the standard unary coding convention.
    """

    @staticmethod
    def length_similarity(a: int, b: int) -> float:
        """Score higher similarity when unary codes are smaller."""
        return 1 / (1 + abs(a - b) * max(a, b))

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

    def similarity(self, other, mode="hamming") -> float:
        if mode == "length":
            return self.length_similarity(self, other)

        return super().similarity(other, mode)
