"""Fuzzy hash primitives."""

from .bloom import BaseBloomFilter as BaseBloomFilter
from .common import BaseFuzzyDigest as BaseFuzzyDigest
from .common import hashable_to_bytes as hashable_to_bytes
from .simhash import BaseSimHash as BaseSimHash
from .unary import BaseUnaryCode as BaseUnaryCode
