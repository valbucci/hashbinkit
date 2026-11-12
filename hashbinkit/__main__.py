import csv
import logging
import sys
from argparse import ArgumentParser
from collections.abc import Collection, Hashable, Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

from r2bagpipe import get_data_manager
from r2bagpipe.parsers import DataParserError
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from hashbinkit.hash.simhash import SimHashAccumulator

from .encode import EmptyGraphError, Graph
from .hash import BaseSimHash
from .preprocess import process_cg

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("r2bagpipe").setLevel(logging.WARNING)
logger = logging.getLogger()


class SimHash512(BaseSimHash):
    BIT_SIZE = 512


parser = ArgumentParser(
    prog="HashBinKit",
    description="Hash various function features across multiple binaries and cached data files.",
)
parser.add_argument("binaries_path")
parser.add_argument("data_path")
parser.add_argument("output_path")
args = parser.parse_args()


BINS_PATH = Path(args.binaries_path)
DATA_PATH = Path(args.data_path)
OUT_PATH = Path(args.output_path)

assert BINS_PATH.is_dir() and DATA_PATH.is_dir()


def iter_graphlet_features(
    graphlets: Iterable[tuple[Hashable, int, int]],
) -> Iterator:
    for _, indegree, outdegree in graphlets:
        yield indegree, outdegree


def hash_subgraphs(graph: Graph, k: int):
    accum = SimHashAccumulator(512)
    seen_nodes = set()
    for node_index in range(graph.size):
        for node_path in graph.dfs_k_paths(k, node_index):
            nodes = tuple(set(node_path))
            if nodes in seen_nodes:
                continue
            n_nodes = len(nodes)
            weight = (n_nodes - 1) / (k - 1)
            subgraph_label, _ = graph.get_subgraph_label(nodes)
            accum.add(SimHash512.hash_feature(subgraph_label), weight)
            seen_nodes.add(nodes)
    return SimHash512(bytes(accum.collapse()))


class BinaryHashes(NamedTuple):
    bin_name: str
    graphlets: str
    subgraphs_3: str
    subgraphs_5: str


def get_hashes(binary_name: str) -> BinaryHashes:
    try:
        bin_data = get_data_manager(binary_name, DATA_PATH)
        graph = process_cg(bin_data)
    except EmptyGraphError:
        logger.error(f"{binary_name} has an empty callgraph.")
        return BinaryHashes(
            bin_name=str(binary_name),
            graphlets="",
            subgraphs_3="",
            subgraphs_5="",
        )
    except DataParserError as e:
        logger.error(f"{binary_name} could not be parsed.")
        logger.error(f"Cause: {e.__cause__!r}")
        return BinaryHashes(
            bin_name=str(binary_name),
            graphlets="",
            subgraphs_3="",
            subgraphs_5="",
        )
    except FileNotFoundError:
        logger.error(f"{binary_name} data files not found.")
        return BinaryHashes(
            bin_name=str(binary_name),
            graphlets="",
            subgraphs_3="",
            subgraphs_5="",
        )

    accum = SimHashAccumulator(512)
    for _, indegree, outdegree in graph.get_graphlets():
        accum.add(SimHash512.hash_feature((indegree, outdegree)))
    glt = SimHash512(bytes(accum.collapse()))

    subg_3 = hash_subgraphs(graph, 3)
    subg_5 = hash_subgraphs(graph, 5)
    return BinaryHashes(
        bin_name=str(binary_name),
        graphlets=str(glt),
        subgraphs_3=str(subg_3),
        subgraphs_5=str(subg_5),
    )


def iter_hashes(binary_names: Collection[str]) -> Iterator[BinaryHashes]:
    with logging_redirect_tqdm():
        for bin_name in tqdm(binary_names, desc=OUT_PATH.name):
            yield get_hashes(bin_name)


CSV_HEADERS = BinaryHashes._fields
OUT_EXISTS = OUT_PATH.is_file()

# List out paths that need analysed
bin_names = set()
if BINS_PATH.is_dir():
    bin_names.update(path.name for path in BINS_PATH.glob("*"))
elif BINS_PATH.is_file():
    # Treat as list of binary names
    with open(BINS_PATH, "r") as list_file:
        bin_names.update(list_file.readlines())

n_bins_todo = len(bin_names)

if OUT_EXISTS:
    logger.info("Checking existing file hashes ...")
    with open(OUT_PATH, newline="") as out_file:
        rows = csv.DictReader(out_file)
        for row in rows:
            done_bin_name = row["bin_name"]
            bin_names.discard(done_bin_name)
            logger.debug(f"Discarded {done_bin_name}: already done.")
    logger.info(f"Todo: {len(bin_names)} out of {n_bins_todo} binaries.")
else:
    logger.info(f"Todo: {n_bins_todo} binaries.")

# Process files to do
with open(OUT_PATH, "a", newline="") as out_file:
    writer = csv.writer(out_file)
    if not OUT_EXISTS:
        writer.writerow(CSV_HEADERS)
    for row in iter_hashes(bin_names):
        writer.writerow(row)
        out_file.flush()
