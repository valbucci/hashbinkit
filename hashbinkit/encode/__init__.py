"""Encode artefacts into features for hashing."""

# Utilities
# - Including them here because a separate file seemed overkill
from collections.abc import Iterable

from .graph import EmptyGraphError as EmptyGraphError
from .graph import Graph as Graph
from .graph import NodeEdge as NodeEdge
from .graph import NodeIndex as NodeIndex
from .graph import NodeIndexMap as NodeIndexMap
from .graph import NodeInputs as NodeInputs
from .graph import NodeInputsMap as NodeInputsMap
from .graph import NodeKey as NodeKey
from .graph import NodeLabel as NodeLabel
from .graph import NodeLabelMap as NodeLabelMap
from .graph import NodeOutputs as NodeOutputs
from .graph import NodeOutputsMap as NodeOutputsMap
from .graph import NodePath as NodePath
from .sequence import Sequence as Sequence


def get_bucket(scalar: int, bucket_bounds: Iterable[int]) -> int:
    """Assign scalar value to bucket number.

    Compute value bucketing/binning based on the provided bounds.

    Example:
        >>> bucket_bounds = [5, 10, 15, 20]
        >>> get_bucket(3, bucket_bounds)
        0
        >>> get_bucket(5, bucket_bounds)
        1
        >>> get_bucket(50, bucket_bounds)
        4

    """
    bucket = 0
    for upper_bound in bucket_bounds:
        if scalar < upper_bound:
            return bucket
        bucket += 1
    return bucket
