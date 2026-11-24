"""Serialization utilities for r2bagpipe dataclasses.

Generated with Claude Sonnet 4.6 (AI:w) and adjusted manually.

Derived from miniserial's design:
<https://github.com/SinclaM/miniserial/blob/main/miniserial/serialization.py>
"""

import types
from abc import abstractmethod
from collections.abc import Callable, Collection
from dataclasses import fields
from enum import Enum
from inspect import isclass
from struct import pack, unpack
from typing import (
    Any,
    Self,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np

# ----------------------------------------------------------------------
# Custom serde registry
# ----------------------------------------------------------------------

_T = TypeVar("_T")

_SerdeMap = dict[
    type,
    tuple[Callable[[Any], bytes], Callable[[bytes], tuple[Any, bytes]]],
]
_serde: _SerdeMap = {}


def register_serde(
    cls: type[_T],
    serializer: Callable[[_T], bytes],
    deserializer: Callable[[bytes], tuple[_T, bytes]],
) -> None:
    """Register a custom ``(serializer, deserializer)`` pair for *cls*.

    *serializer* maps an instance of *cls* to ``bytes``.

    *deserializer* maps ``bytes`` to ``(instance, remaining_bytes)``.
    Registered pairs are checked before all built-in handlers.
    """
    _serde[cls] = serializer, deserializer  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Optional detection
# ----------------------------------------------------------------------


def _optional_inner(type_: type) -> type | None:
    """Return the inner type if type_ is Optional[X], otherwise None.

    Handles both `X | None` and typing.Optional[X]`, and
    `typing.Union[X, None]`.
    """
    origin = get_origin(type_)
    if origin is Union or isinstance(type_, types.UnionType):
        args = get_args(type_)
        non_none = [a for a in args if a is not type(None)]
        if len(args) == 2 and len(non_none) == 1:
            return non_none[0]
    return None


# ----------------------------------------------------------------------
# Core serialize / deserialize
# ----------------------------------------------------------------------


def _serialize(type_: type, v: Any) -> bytes:
    # --- Optional ---
    inner = _optional_inner(type_)
    if inner is not None:
        if v is None:
            return pack("<?", False)
        return pack("<?", True) + _serialize(inner, v)

    # --- Custom registry ---
    for cls, (serializer, _) in _serde.items():
        if isinstance(v, cls):
            return serializer(v)

    # --- Primitives ---
    if isinstance(v, bool):
        return pack("<?", v)
    if isinstance(v, Enum):
        return pack("<Q", v.value)
    if isinstance(v, int):
        return pack("<Q", v)
    if isinstance(v, float):
        return pack("<f", v)
    if isinstance(v, str):
        return v.encode() + b"\x00"

    # --- numpy ndarray ---
    if isinstance(v, np.ndarray):
        dtype_b = v.dtype.str.encode() + b"\x00"
        shape_b = pack("<I", v.ndim) + b"".join(pack("<Q", d) for d in v.shape)
        raw = v.tobytes()
        return dtype_b + shape_b + pack("<Q", len(raw)) + raw

    # --- dict ---
    if isinstance(v, dict):
        key_t, val_t = get_args(type_)
        out = pack("<I", len(v))
        for k, val in v.items():
            out += _serialize(key_t, k)
            out += _serialize(val_t, val)
        return out

    # --- other Collections (list, tuple, set, ...) ---
    if isinstance(v, Collection):
        # tuple[X, ...]
        args = [a for a in get_args(type_) if a is not Ellipsis]
        if len(args) != 1:
            raise TypeError(
                f"Cannot serialize collection {type_!r}: "
                f"expected one element type, got {args}"
            )
        elem_t = args[0]
        out = pack("<I", len(v))
        for x in v:
            out += _serialize(elem_t, x)
        return out

    # --- nested Serializable ---
    if isinstance(v, Serializable):
        return v.serialize()

    raise TypeError(f"Cannot serialize value of type {type(v)!r}")


def _deserialize(type_: type, b: bytes) -> tuple[Any, bytes]:
    # --- Optional ---
    inner = _optional_inner(type_)
    if inner is not None:
        present = b[0]
        b = b[1:]
        if not present:
            return None, b
        return _deserialize(inner, b)

    # --- Custom registry ---
    for cls, (_, deserializer) in _serde.items():
        if cls is type_ or get_origin(cls) is type_:
            return deserializer(b)

    # --- Primitives ---
    if type_ is bool:
        return unpack("<?", b[:1])[0], b[1:]
    if isclass(type_) and issubclass(type_, Enum):
        (value,) = unpack("<Q", b[:8])
        return type_(value), b[8:]
    if type_ is int:
        return unpack("<Q", b[:8])[0], b[8:]
    if type_ is float:
        return unpack("<f", b[:4])[0], b[4:]
    if type_ is str:
        i = b.index(b"\x00")
        return b[:i].decode(), b[i + 1 :]

    # --- numpy ndarray ---
    if isclass(type_) and issubclass(type_, np.ndarray):
        i = b.index(b"\x00")
        dtype = np.dtype(b[:i].decode())
        b = b[i + 1 :]
        (ndim,) = unpack("<I", b[:4])
        b = b[4:]
        shape = tuple(
            unpack("<Q", b[j * 8 : j * 8 + 8])[0] for j in range(ndim)
        )
        b = b[ndim * 8 :]
        (data_len,) = unpack("<Q", b[:8])
        b = b[8:]
        arr = np.frombuffer(b[:data_len], dtype=dtype).reshape(shape).copy()
        return arr, b[data_len:]

    # --- dict ---
    if get_origin(type_) is dict:
        key_t, val_t = get_args(type_)
        (length,) = unpack("<I", b[:4])
        result: dict[Any, Any] = {}
        b = b[4:]
        for _ in range(length):
            k, b = _deserialize(key_t, b)
            val, b = _deserialize(val_t, b)
            result[k] = val
        return result, b

    # --- other Collections ---
    origin = get_origin(type_)
    if isclass(origin) and issubclass(origin, Collection):
        args = [a for a in get_args(type_) if a is not Ellipsis]
        if len(args) != 1:
            raise TypeError(
                f"Cannot deserialize collection {type_!r}: "
                f"expected one element type, got {args}"
            )
        elem_t = args[0]
        (length,) = unpack("<I", b[:4])
        items: list[Any] = []
        b = b[4:]
        for _ in range(length):
            v, b = _deserialize(elem_t, b)
            items.append(v)
        return origin(items), b  # type: ignore[call-arg]

    # --- nested Serializable ---
    if isclass(type_) and issubclass(type_, Serializable):
        return type_._partial_deserialize(b)

    raise TypeError(f"Cannot deserialize type {type_!r}")


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------


class Serializable:
    """Mixin that makes a dataclass binary-serializable."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def serialize(self) -> bytes:
        """Serialize self into bytes."""
        resolved = get_type_hints(type(self))
        return b"".join(
            _serialize(resolved[f.name], getattr(self, f.name))
            for f in fields(self)  # type: ignore[reportArgumentType]
        )

    @classmethod
    def _partial_deserialize(cls, b: bytes) -> tuple[Self, bytes]:
        """Deserialize an instance from bytes blob b."""
        resolved = get_type_hints(cls)
        params: dict[str, Any] = {}
        for f in fields(cls):  # type: ignore[reportArgumentType]
            params[f.name], b = _deserialize(resolved[f.name], b)
        return cls(**params), b

    @classmethod
    def deserialize(cls, b: bytes) -> Self:
        """Deserialize an instance from bytes blob b."""
        return cls._partial_deserialize(b)[0]
