"""Encode graph-structured artefacts."""

import logging
from collections.abc import Iterable
from typing import TypeAlias

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


NodeKey: TypeAlias = str
NodeLabel: TypeAlias = str
NodeInputs: TypeAlias = tuple[NodeKey, tuple[NodeKey, ...]]
NodeOutputs: TypeAlias = tuple[NodeKey, tuple[NodeKey, ...]]
GraphInputs: TypeAlias = dict[NodeKey, tuple[NodeKey, ...]]
GraphOutputs: TypeAlias = dict[NodeKey, tuple[NodeKey, ...]]
GraphLabels: TypeAlias = dict[NodeKey, NodeLabel]


def machoke_line(
    graph_edges: Iterable[NodeOutputs], graph_labels: GraphLabels
) -> str:
    """Encode graph labels and edges with Machoke algorithm.

    Follows order of the provided list `graph_edges`.
    The `graph_labels` dictionary uses any str-encoded
    node key (e.g. basic block address, function name);
    normally, this should be chosen based on provided artefacts.
    """
    machoke_line = ""
    for node_key, outputs in graph_edges:
        try:
            node_label = graph_labels[node_key]
        except KeyError:
            logger.debug(f"No label provided for node: {node_key}")
            continue

        out_node_labels = []
        for out_key in outputs:
            try:
                out_node_labels.append(graph_labels[out_key])
            except KeyError:
                logger.debug(f"No label provided for output node: {node_key}")

        machoke_line += f"{node_label}:{','.join(out_node_labels)};"
    return machoke_line
