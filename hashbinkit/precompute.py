"""Pre-compute and cache canonical label lookup tables.

Covers all k * k adjacency matrices for k in {3, 4, 5}.

Lookup table layout:
- labels[bitmask] -> uint32 canonical label
- perms[bitmask]  -> uint32 best permutation (4 bits per node)

Usage (once):
    >>> python -m hashbinkit.precompute
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

from .canonicalize import get_adjacency_label as brute_force_adjacency_label

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_cache_dir() -> Path:
    """Return the cache directory based on env var or default."""
    return Path(
        os.environ.get(
            "HASHBINKIT_CACHE_DIR",  # cache env var
            Path.home() / ".cache" / "hashbinkit",
        )
    )


def get_table_path(k: int, cache_dir: Path | None = None) -> Path:
    """Return the path to the compressed table file for given k."""
    return (cache_dir or get_cache_dir()) / f"adj_labels_k{k}.npz"


@njit(cache=True, parallel=True)
def _build_table(k: int):
    """Build labels and perms arrays over all input matrices."""
    n_entries = 1 << (k * k)
    labels = np.zeros(n_entries, dtype=np.uint32)
    perms = np.zeros((n_entries, k), dtype=np.uint8)

    for bitmask in prange(n_entries):
        matrix = np.zeros((k, k), dtype=np.int8)
        for pos in range(k * k):
            if (bitmask >> pos) & 1:
                matrix[pos // k, pos % k] = 1

        label, perm = brute_force_adjacency_label(matrix)

        labels[bitmask] = label
        for i in range(k):
            perms[bitmask, i] = perm[i]

    return labels, perms


def precompute(k: int, cache_dir: Path | None = None) -> Path:
    """Build the lookup table and save it compressed.

    Returns the path where the table was saved.
    """
    path = get_table_path(k, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_entries = 1 << (k * k)
    logger.info(
        f"Pre-computing k={k} table ({n_entries:,} entries, "
        f"input space 2^{k * k}) ..."
    )
    t0 = time.perf_counter()
    labels, perms = _build_table(k)
    elapsed = time.perf_counter() - t0
    logger.info(f"  Computed in {elapsed:.1f}s — saving to {path} ...")

    np.savez_compressed(path, labels=labels, perms=perms)
    size_mb = path.stat().st_size / 1e6
    logger.info(f"  Saved ({size_mb:.1f} MB compressed)")
    return path


def load_table(
    k: int, cache_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed labels and perms arrays for given k.

    If the table does not exist it is computed and saved automatically
    on first call (one-time computation cost). A warning is emitted
    so the delay is visible in logs.

    Returns:
        - labels: uint32 array of shape (2^(k * k),)
        - perms:  uint8  array of shape (2^(k * k), k), already decoded

    """
    path = get_table_path(k, cache_dir)
    if not path.exists():
        logger.warning(
            "Precomputed canonical label table for k=%d not found at %s. "
            "Computing now (one-time computation cost) ...",
            k,
            path,
        )
        precompute(k, cache_dir)
    data = np.load(path)
    return data["labels"], data["perms"]


@njit(cache=True)
def _encode_matrix(matrix: np.ndarray) -> np.uint32:
    """Encode a k * k binary matrix as a row-major bitmask integer."""
    k = matrix.shape[0]
    bitmask = np.uint32(0)
    pos = np.uint32(0)
    for row in range(k):
        for col in range(k):
            if matrix[row, col]:
                bitmask |= 1 << pos
            pos += 1
    return bitmask


# Lazily populated lookup tables: k -> (labels, perms)
_TABLES: dict[int, tuple[np.ndarray, np.ndarray]] = {}

# NOTE: k=3 and k=4 brute-force are faster than the lookup overhead.
#       Only k=5 (120 perms) benefits enough to justify the table path.
_SUPPORTED_KS = frozenset((5,))


def _get_table(k: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the cached lookup table for k.

    Loads the lookup table on first access.
    """
    if k not in _SUPPORTED_KS:
        return None
    if k not in _TABLES:
        _TABLES[k] = load_table(k)
    return _TABLES[k]


def get_adjacency_label(
    matrix: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Return canonical label and best permutation for a k * k matrix.

    Uses a precomputed lookup table for supported values of k.
    Falls back to brute-force for other values.
    """
    k = matrix.shape[0]
    table = _get_table(k)
    if table is not None:
        labels, perms = table
        bitmask = _encode_matrix(matrix)
        return labels[bitmask], perms[bitmask]
    return brute_force_adjacency_label(matrix)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Pre-compute canonical label lookup tables for k=3..5."
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        metavar="K",
        help="Values of k to precompute (default: 3 4 5)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Output directory "
            "(default: $HASHBINKIT_CACHE_DIR or ~/.cache/hashbinkit)"
        ),
    )
    args = parser.parse_args()

    for k in sorted(args.ks):
        precompute(k, args.cache_dir)

    print("All tables ready.")
