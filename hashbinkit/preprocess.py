"""Preprocess raw artefacts before feature encoding."""

import logging
from collections.abc import Iterator

from r2bagpipe import BinaryDataManager, FunctionCodeData

from hashbinkit.encode.graph import NodeKey

from .encode import Graph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def process_cg(bin_data: BinaryDataManager) -> Graph:
    """Process global callgraph into generic Graph encoder."""

    def iter_cg_nodes() -> Iterator[NodeKey]:
        # Follow the order of the decompiled functions
        # and leave all the external imports at the end
        callers = bin_data.func_imports.keys()
        yield from callers

        terminal_callees = bin_data.name_symbols - callers
        yield from terminal_callees

    graph = Graph()
    graph.setup_nodes(iter_cg_nodes())
    graph.setup_outputs(bin_data.func_imports.items())
    return graph


def process_cfg(func_code: FunctionCodeData) -> Graph:
    """Parse function CFG into generic Graph encoder."""
    nodes = []
    edges = []
    for block in func_code.blocks:
        nodes.append(block.addr)
        if block.jump is not None:
            edges.append((block.addr, block.jump))
        if block.fail is not None:
            edges.append((block.addr, block.fail))

    graph = Graph()
    graph.setup_nodes(nodes)
    graph.setup_edges(edges)
    return graph
