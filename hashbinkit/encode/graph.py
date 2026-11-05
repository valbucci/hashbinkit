"""Encode and manage graph-structured artefacts."""

import logging
from collections.abc import Hashable, Iterable, Iterator
from typing import TypeAlias

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


NodeKey: TypeAlias = Hashable
NodeIndex: TypeAlias = int
NodeLabel: TypeAlias = str
NodeEdge: TypeAlias = tuple[NodeKey, NodeKey]
NodeInputs: TypeAlias = tuple[NodeKey, tuple[NodeKey, ...]]
NodeOutputs: TypeAlias = tuple[NodeKey, tuple[NodeKey, ...]]
NodeInputsMap: TypeAlias = dict[NodeKey, tuple[NodeKey, ...]]
NodeOutputsMap: TypeAlias = dict[NodeKey, tuple[NodeKey, ...]]
NodeLabelMap: TypeAlias = dict[NodeKey, NodeLabel]
NodeIndexMap: TypeAlias = dict[NodeKey, NodeIndex]
NodePath: TypeAlias = tuple[NodeIndex, ...]


# region NUMBA GRAPH LABELLING

# ---------------------------------------------------------------------
# In this region there are a few optimised functions for
# brute-forcing the smallest graph canonical labelling.
# ---------------------------------------------------------------------


@njit
def label_permutation(matrix: np.ndarray, permutation: np.ndarray) -> int:
    """Label matrix permutation bits as integer."""
    label = 0
    n = matrix.shape[0]
    pos = 0
    for row in range(n):
        for col in range(n):
            if matrix[permutation[row], permutation[col]]:
                label |= 1 << pos
            pos += 1
    return label


@njit
def swap_array_values(a: int, b: int, array: np.ndarray) -> None:
    """Swap positions a <-> b in array."""
    tmp = array[a]
    array[a] = array[b]
    array[b] = tmp


@njit
def next_permutation(n: int, array: np.ndarray) -> bool:
    """Generate next in-place permutation of ordered array.

    Returns True if more permutations can be done, False otherwise.
    """
    i = n - 2
    while i >= 0 and array[i] >= array[i + 1]:
        i -= 1

    if i < 0:
        return False

    j = n - 1
    while array[j] <= array[i]:
        j -= 1

    swap_array_values(i, j, array)

    # Reverse suffix in place
    left = i + 1
    right = n - 1
    while left < right:
        swap_array_values(left, right, array)
        left += 1
        right -= 1

    return True


@njit
def copy_array(dst, src) -> None:
    """Copy src into dst."""
    for i in range(len(dst)):
        dst[i] = src[i]


@njit
def search_canonical_label(matrix) -> tuple[int, np.ndarray]:
    """Brute-force lowest int label for isomorphic adjacency matrix.

    Returns the canonical graph label and its
    adjacency matrix permutation.
    """
    n = matrix.shape[0]
    permutation = np.arange(n)
    best_permutation = permutation.copy()
    best_label = label_permutation(matrix, permutation)

    while next_permutation(n, permutation):
        current_label = label_permutation(matrix, permutation)
        if current_label < best_label:
            copy_array(best_permutation, permutation)
            best_label = current_label

    return best_label, best_permutation


# endregion NUMBA GRAPH LABELLING


class Graph:
    def __init__(
        self,
        nodes: Iterable[NodeKey] = [],
        edges: Iterable[NodeEdge] = [],
    ) -> None:
        """Init a graph with given nodes and edges.

        The nodes will be indexed following the same order as
        the provided `nodes` Iterable.

        If no nodes are provided an empty graph is created.
        """
        self.setup_nodes(nodes)
        try:
            self.setup_adjacency(edges)
        except RuntimeError:
            # Just an empty graph
            pass

    def setup_nodes(self, nodes: Iterable[NodeKey]) -> None:
        """Set the node index for internal use."""
        self.index = list(nodes)
        self.index_map: dict[NodeKey, NodeIndex] = {
            node_key: node_index
            for node_index, node_key in enumerate(self.index)
        }

    def setup_adjacency(self, edges: Iterable[NodeEdge]) -> None:
        """Set adjacency matrix based on provided edges.

        Overwrites adjacency matrix with the provided edges.
        If the node index is not set it will throw a RuntimeError.
        """
        if len(self.index) == 0:
            raise RuntimeError("You must provide the nodes first.")

        N = len(self.index)
        self.adj = np.zeros(shape=(N, N), dtype=np.int8)
        for input, output in edges:
            input_index = self.index_map[input]
            output_index = self.index_map[output]
            self.adj[input_index, output_index] = 1

        # Cache indegree / outdegree values
        self._indegree = np.count_nonzero(self.adj, axis=0)
        self._outdegree = np.count_nonzero(self.adj, axis=1)
        self._inputs_cache = {}
        self._outputs_cache = {}

    @property
    def size(self) -> int:
        """Amount of nodes in the graph."""
        return self.adj.shape[0]

    def is_node_sink(self, node: NodeIndex) -> bool:
        """Return True if node has outdegree of 0."""
        return not np.any(self.adj[node])

    def get_node_indegree(self, node: NodeIndex) -> int:
        """Return node's number of inputs."""
        return int(self._indegree[node])

    def get_node_inputs(self, node: NodeIndex) -> np.ndarray:
        """Return the index of all the inputs of given node."""
        try:
            return self._inputs_cache[node]
        except KeyError:
            pass
        inputs = np.flatnonzero(self.adj[:, node])
        self._inputs_cache[node] = inputs
        return inputs

    def get_node_outdegree(self, node: NodeIndex) -> int:
        """Return node's number of outputs."""
        return int(self._outdegree[node])

    def get_node_outputs(self, node: NodeIndex) -> np.ndarray:
        """Return the index of all the outputs from given node."""
        try:
            return self._outputs_cache[node]
        except KeyError:
            pass

        outputs = np.flatnonzero(self.adj[node])
        self._outputs_cache[node] = outputs
        return outputs

    def get_node_graphlet(self, node: NodeIndex) -> tuple[int, int]:
        """Return indegree/outdegree pair of given node."""
        return self.get_node_indegree(node), self.get_node_outdegree(node)

    def get_graphlets(self) -> Iterator[tuple[NodeKey, int, int]]:
        """Yield indegree/outdegree pairs of all nodes in the graph."""
        for node_index, node_key in enumerate(self.index):
            yield (node_key,) + self.get_node_graphlet(node_index)

    def dfs_k_paths(
        self,
        k: int,
        current_node: NodeIndex,
        _current_path: list[NodeIndex] | None = None,
    ) -> Iterator[list[NodeIndex]]:
        """DFS of all k-sized paths from current_node."""
        if _current_path is None:
            _current_path = [current_node]
        elif len(_current_path) >= k:
            # Yield current k-sized path
            yield _current_path[:k]
            return

        outputs = self.get_node_outputs(current_node)
        if outputs.size == 0:
            # Sink node; stop early
            yield _current_path
            return

        for hop in outputs:
            # Compute next hops
            _current_path.append(hop)
            yield from self.dfs_k_paths(k, hop, _current_path)
            _current_path.pop()

    def get_k_subgraphs(
        self,
        k: int,
        entry: NodeIndex = 0,
        _current_path: list[NodeIndex] | None = None,
        _seen_paths=None,
    ) -> Iterator[tuple[NodePath, np.ndarray, int]]:
        """DFS traversal for all k-subgraphs from the entry point.

        For each subgraph yields:
            - NodePath: a tuple of neighbouring nodes
            - np.ndarray: their adjacency matrix
            - int: the canonical label of the adjacency matrix
        """
        A = self.adj
        seen_paths = _seen_paths
        if seen_paths is None:
            seen_paths = set()

        for path in self.dfs_k_paths(k, entry, _current_path):
            path_nodes = tuple(frozenset(path))
            if path_nodes in seen_paths:
                continue
            seen_paths.add(path_nodes)
            sub_A = A[np.ix_(path_nodes, path_nodes)]
            label, _ = search_canonical_label(sub_A)
            yield tuple(path), sub_A, label

            if not self.is_node_sink(path[-1]):
                yield from self.get_k_subgraphs(
                    k, path[-1], path[1:], seen_paths
                )

    def get_machoke_words(
        self, labels: NodeLabelMap | None = None
    ) -> Iterator[str]:
        """Encode graph with Machoke algorithm.

        Iterates Machoke words for each graph node (e.g. "1c:2,3;").
        Follows order of nodes provided at setup time.

        Args:
            labels (Optional): dictionary with str-encoded values
                suffixed to the machoke word -- The original algorithm
                adds a 'c' suffix for CFG basic blocks with external
                function calls.

        """
        if labels is None:
            labels = {}
        for node_index in range(self.size):
            node_key = self.index[node_index]
            node_label = labels.get(node_key, "")
            word_left = f"{node_index}{node_label}:"

            output_indexes = self.get_node_outputs(node_index)
            word_right = ",".join(map(str, output_indexes)) + ";"

            yield word_left + word_right

    def get_machoke_line(self, labels: NodeLabelMap | None = None) -> str:
        """Encode graph to string with Machoke algorithm."""
        return "".join(self.get_machoke_words(labels))
