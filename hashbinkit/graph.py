"""Encode and manage graph-structured artefacts."""

import logging
from collections.abc import Iterable, Iterator
from typing import Literal, Self, TypeAlias

import numpy as np
from scipy.sparse import csc_array, csr_array
from scipy.sparse.csgraph import breadth_first_order, depth_first_order

from .canonicalize import get_adjacency_label

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


NodeKey: TypeAlias = str | int
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


class EmptyGraphError(RuntimeError):
    """Raised when edges are referenced in an empty graph."""


class Graph:
    """Generic manager to handle graph operations."""

    __slots__ = (
        "_adj_csc",
        "_adj_csr",
        "_indegree",
        "_inputs_cache",
        "_outdegree",
        "_outputs_cache",
        "index",
        "index_map",
    )

    def __init__(self) -> None:
        self.index: tuple[NodeKey, ...] = ()
        self.index_map: dict[NodeKey, NodeIndex] = {}
        self._adj_csr = None
        self._adj_csc = None
        self._indegree: np.ndarray | None = None
        self._outdegree: np.ndarray | None = None
        self._inputs_cache: dict[NodeIndex, np.ndarray] = {}
        self._outputs_cache: dict[NodeIndex, np.ndarray] = {}

    def setup(
        self,
        nodes: Iterable[NodeKey] | None = None,
        edges: Iterable[NodeEdge] | None = None,
        outputs: Iterable[NodeOutputs] | None = None,
    ) -> Self:
        """Init a graph with given nodes and edges.

        The nodes will be indexed following the same order as
        the provided `nodes` Iterable.

        If no nodes are provided an empty graph is created.
        """
        if nodes is not None:
            self.setup_nodes(nodes)

        if edges is not None:
            self.setup_edges(edges)
        elif outputs is not None:
            self.setup_outputs(outputs)

        return self.setup_cache()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Graph):
            return NotImplemented
        if self._adj_csr is None or other._adj_csr is None:
            raise EmptyGraphError(
                "Comparing two graphs while at least one is empty."
            )
        if self.index != other.index:
            return False

        return (self._adj_csr != other._adj_csr).nnz == 0  # type: ignore[reportAttributeAccessIssue]

    @property
    def adj(self) -> csr_array:
        """The adjacency matrix in row-wise sparse format."""
        if self._adj_csr is None:
            raise EmptyGraphError(
                "The adjacency matrix is not set because the graph is empty."
            )
        return self._adj_csr

    @property
    def indegree(self) -> np.ndarray:
        """Lookup a NodeIndex indegree (node_index -> indegree)."""
        if self._indegree is None:
            raise EmptyGraphError
        return self._indegree

    @property
    def outdegree(self) -> np.ndarray:
        """Lookup a NodeIndex outdegree (node_index -> outdegree)."""
        if self._outdegree is None:
            raise EmptyGraphError
        return self._outdegree

    @property
    def adj_csc(self) -> csc_array:
        """The adjacency matrix in column-wise sparse format."""
        if self._adj_csc is None:
            raise EmptyGraphError(
                "The adjacency matrix is not set because the graph is empty."
            )
        return self._adj_csc

    @classmethod
    def build(cls, adj: csr_array, index: tuple[NodeKey, ...]) -> Self:
        """Return new Graph from a CSR adjacency and node index."""
        graph = cls()
        graph.setup_nodes(index)
        graph._adj_csr = adj
        graph.setup_cache()
        return graph

    def extract_subgraph(
        self,
        start: NodeIndex,
        order: Literal["breadth", "depth"] = "breadth",
        directed: bool = True,
    ) -> "Graph":
        """Extract subgraph reachable from start node via BFS or DFS."""
        traverse = (
            breadth_first_order if order == "breadth" else depth_first_order
        )
        reachable = traverse(
            self.adj, start, directed=directed, return_predecessors=False
        )
        nodes = reachable[reachable >= 0]  # filter -1 sentinels
        sub_index = tuple(self.index[i] for i in nodes)
        sub_adj = self.adj[nodes, :][:, nodes]
        return Graph.build(sub_adj, sub_index)

    def setup_nodes(self, nodes: Iterable[NodeKey]) -> Self:
        """Set the node index for internal use."""
        self.index = tuple(nodes)
        self.index_map: dict[NodeKey, NodeIndex] = {
            node_key: node_index
            for node_index, node_key in enumerate(self.index)
        }
        return self

    def setup_edges(self, edges: Iterable[NodeEdge]) -> Self:
        """Set adjacency matrix based on provided edges.

        Overwrites adjacency matrix with the provided edges.
        If the node index is not set it will throw a RuntimeError.
        """
        N = len(self.index)
        if N == 0:
            raise EmptyGraphError("Cannot set edges on an empty graph.")

        try:
            src_keys, dst_keys = zip(*edges)
            rows = [self.index_map[src] for src in src_keys]
            cols = [self.index_map[dst] for dst in dst_keys]
        except ValueError:
            rows, cols = [], []

        data = np.ones(len(rows), dtype=np.int8)
        self._adj_csr = csr_array(
            (data, (rows, cols)), shape=(N, N), dtype=np.int8
        )
        return self

    def setup_outputs(self, node_outputs: Iterable[NodeOutputs]) -> Self:
        """Set adjacency matrix based on provided node outputs.

        Like `setup_edges` but with grouped outputs for each node.
        """

        def iterate_edges() -> Iterator[NodeEdge]:
            for node, outputs in node_outputs:
                for out in outputs:
                    yield node, out

        return self.setup_edges(iterate_edges())

    def setup_cache(self) -> Self:
        """Cache indegree / outdegree values."""
        self._adj_csc = self.adj.tocsc()
        self._indegree = np.asarray(self.adj.sum(axis=0)).flatten()
        self._outdegree = np.asarray(self.adj.sum(axis=1)).flatten()
        return self

    @property
    def size(self) -> int:
        """Amount of nodes in the graph."""
        return self.adj.shape[0]  # type: ignore[reportOptionalSubscript]

    def is_node_sink(self, node: NodeIndex) -> bool:
        """Return True if node has outdegree of 0."""
        return self.adj[[node]].nnz == 0

    def get_node_indegree(self, node: NodeIndex) -> int:
        """Return node's number of inputs."""
        return int(self.indegree[node])

    def get_node_inputs(self, node: NodeIndex) -> np.ndarray:
        """Return the index of all the inputs of given node."""
        try:
            return self._inputs_cache[node]
        except KeyError:
            pass
        inputs = self.adj_csc[:, node].indices
        self._inputs_cache[node] = inputs
        return inputs

    def get_node_outdegree(self, node: NodeIndex) -> int:
        """Return node's number of outputs."""
        return int(self.outdegree[node])

    def get_node_outputs(self, node: NodeIndex) -> np.ndarray:
        """Return the index of all the outputs from given node."""
        try:
            return self._outputs_cache[node]
        except KeyError:
            pass

        outputs = self.adj[[node]].indices
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

    def get_subgraph_label(
        self, nodes: tuple[NodeIndex, ...]
    ) -> tuple[int, np.ndarray]:
        """Return subgraph from adjacency and its canonical label."""
        sub_adj = self.adj[nodes, :][:, nodes].toarray()
        return get_adjacency_label(sub_adj)

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
        seen_paths = _seen_paths
        if seen_paths is None:
            seen_paths = set()

        for path in self.dfs_k_paths(k, entry, _current_path):
            path_nodes = tuple(set(path))
            if path_nodes in seen_paths:
                continue
            seen_paths.add(path_nodes)
            label, sub_adj = self.get_subgraph_label(path_nodes)
            yield tuple(path), sub_adj, label

            if not self.is_node_sink(path[-1]):
                yield from self.get_k_subgraphs(
                    k, path[-1], path[1:], seen_paths
                )

    def get_machoke_words(
        self, labels: NodeLabelMap | None = None
    ) -> Iterator[str]:
        """Encode graph with Machoke algorithm.

        Iterates Machoke words for each graph node (e.g. "1:2,3;").
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
            node_label = labels.get(node_key, node_index)
            word_left = f"{node_label}:"

            output_indexes = self.get_node_outputs(node_index)
            word_right = ",".join(map(str, output_indexes)) + ";"

            yield word_left + word_right

    def get_machoke_line(self, labels: NodeLabelMap | None = None) -> str:
        """Encode graph to string with Machoke algorithm."""
        return "".join(self.get_machoke_words(labels))
