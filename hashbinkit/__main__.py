import csv
import logging
import sys
from argparse import ArgumentParser
from collections.abc import Collection, Iterable, Iterator
from pathlib import Path
from typing import NamedTuple, Self

from hexio import HexStr
from r2bagpipe import get_data_manager
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from hashbinkit.hash.common import HashableElement
from hashbinkit.hash.simhash import SimHashAccumulator
from hashbinkit.preprocess import (
    process_esil,
    process_opcodes,
    process_references,
    process_variables,
)

from .hash import BaseSimHash

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


class BinaryHashes(NamedTuple):
    bin_name: str
    op: str
    op_tri: str
    op_tri_sort: str
    op_canon: str
    op_canon_tri: str
    op_canon_tri_sort: str
    esil: str
    esil_tri: str
    esil_tri_sort: str
    esil_canon: str
    esil_canon_tri: str
    esil_canon_tri_sort: str
    ref: str
    ref_tri: str
    ref_tri_sort: str
    ref_canon: str
    ref_canon_tri: str
    ref_canon_tri_sort: str
    locals: str
    locals_canon: str
    args: str
    args_canon: str
    vars: str
    vars_canon: str

    @classmethod
    def empty(cls, binary_name: str) -> Self:
        return cls(
            bin_name=binary_name,
            op="",
            op_tri="",
            op_tri_sort="",
            op_canon="",
            op_canon_tri="",
            op_canon_tri_sort="",
            esil="",
            esil_tri="",
            esil_tri_sort="",
            esil_canon="",
            esil_canon_tri="",
            esil_canon_tri_sort="",
            ref="",
            ref_tri="",
            ref_tri_sort="",
            ref_canon="",
            ref_canon_tri="",
            ref_canon_tri_sort="",
            locals="",
            locals_canon="",
            args="",
            args_canon="",
            vars="",
            vars_canon="",
        )


def update_accumulator(
    accumulator: SimHashAccumulator, features: Iterable[HashableElement]
) -> None:
    for feat in features:
        accumulator.add(SimHash512.hash_feature(feat))


def update_accumulator_with_frequency(
    accumulator: SimHashAccumulator,
    features: Iterable[tuple[HashableElement, int]],
) -> None:
    for feat, freq in features:
        accumulator.add(SimHash512.hash_feature(feat), freq)


def accumulator_to_hexstr(accumulator: SimHashAccumulator) -> HexStr:
    return HexStr(bytes(accumulator.collapse()))


def get_hashes(binary_name: str) -> BinaryHashes:
    try:
        manager = get_data_manager(binary_name, DATA_PATH)
    except:
        logger.exception(f"Skipping {binary_name}.")
        return BinaryHashes.empty(binary_name)
    accum_op = SimHashAccumulator(512)
    accum_op_tri = SimHashAccumulator(512)
    accum_op_tri_sort = SimHashAccumulator(512)
    accum_op_canon = SimHashAccumulator(512)
    accum_op_canon_tri = SimHashAccumulator(512)
    accum_op_canon_tri_sort = SimHashAccumulator(512)
    accum_esil = SimHashAccumulator(512)
    accum_esil_tri = SimHashAccumulator(512)
    accum_esil_tri_sort = SimHashAccumulator(512)
    accum_esil_canon = SimHashAccumulator(512)
    accum_esil_canon_tri = SimHashAccumulator(512)
    accum_esil_canon_tri_sort = SimHashAccumulator(512)
    accum_ref = SimHashAccumulator(512)
    accum_ref_tri = SimHashAccumulator(512)
    accum_ref_tri_sort = SimHashAccumulator(512)
    accum_ref_canon = SimHashAccumulator(512)
    accum_ref_canon_tri = SimHashAccumulator(512)
    accum_ref_canon_tri_sort = SimHashAccumulator(512)
    accum_locals = SimHashAccumulator(512)
    accum_locals_canon = SimHashAccumulator(512)
    accum_args = SimHashAccumulator(512)
    accum_args_canon = SimHashAccumulator(512)
    accum_vars = SimHashAccumulator(512)
    accum_vars_canon = SimHashAccumulator(512)
    for func_addr in manager.func_index:
        func_code = manager.func_cfgs.get(func_addr)
        func_zig = manager.func_zigs.get(func_addr)

        if func_code is not None:
            opcode_sequence = process_opcodes(func_code)
            opcode_canon_sequence = process_opcodes(func_code, normalise="all")
            esil_sequence = process_esil(func_code)
            esil_canon_sequence = process_esil(func_code, normalise="all")
            ref_sequence = process_references(func_code)
            ref_canon_sequence = process_references(func_code, normalise="all")
            update_accumulator_with_frequency(
                accum_op, opcode_sequence.get_frequency()
            )
            update_accumulator(accum_op_tri, opcode_sequence.get_n_grams(3))
            update_accumulator(
                accum_op_tri_sort, opcode_sequence.get_n_grams_sorted(3)
            )
            update_accumulator_with_frequency(
                accum_op_canon, opcode_canon_sequence.get_frequency()
            )
            update_accumulator(
                accum_op_canon_tri, opcode_canon_sequence.get_n_grams(3)
            )
            update_accumulator(
                accum_op_canon_tri_sort,
                opcode_canon_sequence.get_n_grams_sorted(3),
            )
            update_accumulator_with_frequency(
                accum_esil, esil_sequence.get_frequency()
            )
            update_accumulator(accum_esil_tri, esil_sequence.get_n_grams(3))
            update_accumulator(
                accum_esil_tri_sort, esil_sequence.get_n_grams_sorted(3)
            )
            update_accumulator_with_frequency(
                accum_esil_canon, esil_canon_sequence.get_frequency()
            )
            update_accumulator(
                accum_esil_canon_tri, esil_canon_sequence.get_n_grams(3)
            )
            update_accumulator(
                accum_esil_canon_tri_sort,
                esil_canon_sequence.get_n_grams_sorted(3),
            )
            update_accumulator_with_frequency(
                accum_ref, ref_sequence.get_frequency()
            )
            update_accumulator(accum_ref_tri, ref_sequence.get_n_grams(3))
            update_accumulator(
                accum_ref_tri_sort, ref_sequence.get_n_grams_sorted(3)
            )
            update_accumulator_with_frequency(
                accum_ref_canon, ref_canon_sequence.get_frequency()
            )
            update_accumulator(
                accum_ref_canon_tri, ref_canon_sequence.get_n_grams(3)
            )
            update_accumulator(
                accum_ref_canon_tri_sort,
                ref_canon_sequence.get_n_grams_sorted(3),
            )

        if func_zig is not None:
            locals_sequence = process_variables(func_zig, mode="locals")
            locals_canon_sequence = process_variables(
                func_zig, mode="locals", keep_bits=False
            )
            args_sequence = process_variables(func_zig, mode="args")
            args_sequence_canon = process_variables(
                func_zig, mode="args", keep_bits=False
            )
            vars_sequence = process_variables(func_zig)
            vars_sequence_canon = process_variables(func_zig, keep_bits=False)
            update_accumulator_with_frequency(
                accum_locals, locals_sequence.get_frequency()
            )
            update_accumulator_with_frequency(
                accum_locals_canon, locals_canon_sequence.get_frequency()
            )
            update_accumulator_with_frequency(
                accum_args, args_sequence.get_frequency()
            )
            update_accumulator_with_frequency(
                accum_args_canon, args_sequence_canon.get_frequency()
            )
            update_accumulator_with_frequency(
                accum_vars, vars_sequence.get_frequency()
            )
            update_accumulator_with_frequency(
                accum_vars_canon, vars_sequence_canon.get_frequency()
            )

    return BinaryHashes(
        bin_name=binary_name,
        op=accumulator_to_hexstr(accum_op),
        op_tri=accumulator_to_hexstr(accum_op_tri),
        op_tri_sort=accumulator_to_hexstr(accum_op_tri_sort),
        op_canon=accumulator_to_hexstr(accum_op_canon),
        op_canon_tri=accumulator_to_hexstr(accum_op_canon_tri),
        op_canon_tri_sort=accumulator_to_hexstr(accum_op_canon_tri_sort),
        esil=accumulator_to_hexstr(accum_esil),
        esil_tri=accumulator_to_hexstr(accum_esil_tri),
        esil_tri_sort=accumulator_to_hexstr(accum_esil_tri_sort),
        esil_canon=accumulator_to_hexstr(accum_esil_canon),
        esil_canon_tri=accumulator_to_hexstr(accum_esil_canon_tri),
        esil_canon_tri_sort=accumulator_to_hexstr(accum_esil_canon_tri_sort),
        ref=accumulator_to_hexstr(accum_ref),
        ref_tri=accumulator_to_hexstr(accum_ref_tri),
        ref_tri_sort=accumulator_to_hexstr(accum_ref_tri_sort),
        ref_canon=accumulator_to_hexstr(accum_ref_canon),
        ref_canon_tri=accumulator_to_hexstr(accum_ref_canon_tri),
        ref_canon_tri_sort=accumulator_to_hexstr(accum_ref_canon_tri_sort),
        locals=accumulator_to_hexstr(accum_locals),
        locals_canon=accumulator_to_hexstr(accum_locals_canon),
        args=accumulator_to_hexstr(accum_args),
        args_canon=accumulator_to_hexstr(accum_args_canon),
        vars=accumulator_to_hexstr(accum_vars),
        vars_canon=accumulator_to_hexstr(accum_vars_canon),
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
