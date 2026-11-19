"""Process raw artefacts for feature encoding."""

import logging
import re
from collections.abc import Iterator
from typing import Literal

from r2bagpipe import (
    BinaryDataManager,
    EsilTokenType,
    FunctionCodeData,
    FunctionZignatureData,
    ReferenceType,
)

from hashbinkit.encode.graph import NodeKey

from .encode import Graph, Sequence

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# region CONSTANTS

VAR_BITS_REGEX = re.compile(
    r"\b(u?int|ptr|char|long|short|float|double)(\d+_?t?)\b",
    flags=re.IGNORECASE,
)

# NOTE: OPTYPE_CANONICAL was generated using Claude Sonnet 4.6
#       including source code file in context as reference (AI:w).
OPTYPE_CANONICAL = {
    # Unconditional jumps (direct, indirect, memory, register)
    "jmp": "JMP",
    "mjmp": "JMP",
    "ujmp": "JMP",
    "rjmp": "JMP",
    "ijmp": "JMP",
    "irjmp": "JMP",
    "switch": "JMP",
    "case": "JMP",
    # Conditional jumps
    "cjmp": "CJMP",
    "ucjmp": "CJMP",
    "rcjmp": "CJMP",
    "mcjmp": "CJMP",
    # Calls
    "call": "CALL",
    "ccall": "CALL",
    "ucall": "CALL",
    "rcall": "CALL",
    "icall": "CALL",
    "ircall": "CALL",
    "uccall": "CALL",
    # Returns
    "ret": "RET",
    "cret": "RET",
    # Stack
    "push": "PUSH",
    "upush": "PUSH",
    "rpush": "PUSH",
    "pop": "POP",
    "leave": "POP",  # frame teardown = pop frame pointer, x86-specific spelling
    # Memory access
    "load": "LOAD",
    "store": "STORE",
    # Data movement
    "mov": "MOV",
    "cmov": "MOV",
    "xchg": "MOV",
    # Arithmetic (including address arithmetic)
    "add": "MATH",
    "sub": "MATH",
    "mul": "MATH",
    "div": "MATH",
    "mod": "MATH",
    "abs": "MATH",
    "lea": "MATH",  # computes address via arithmetic, never touches memory
    "ulea": "MATH",
    # Bitwise logic
    "and": "LOGIC",
    "or": "LOGIC",
    "xor": "LOGIC",
    "not": "LOGIC",
    "cpl": "LOGIC",
    # Shifts and rotates
    "shl": "SHIFT",
    "shr": "SHIFT",
    "sal": "SHIFT",
    "sar": "SHIFT",
    "ror": "SHIFT",
    "rol": "SHIFT",
    # Compare
    "cmp": "CMP",
    "acmp": "CMP",
    # Syscall / software interrupt
    "swi": "SWI",
    "cswi": "SWI",
    # Privileged / hardware
    "trap": "TRAP",
    "io": "IO",
    "sync": "SYNC",
    "crypto": "CRYPTO",
    # No-op / padding
    "nop": "NOP",
    "null": "NOP",
    "rep": "NOP",  # repeat prefix surfaced as op type, not a true instruction class
    # Failed or unrecognised disassembly
    "ill": "UNK",
    "unk": "UNK",
    "new": "UNK",  # VM/CIL artefact, should not appear in native code
    "cast": "UNK",
    "length": "UNK",
}


# endregion CONSTANTS

# region GRAPHS


def process_cg(bin_data: BinaryDataManager) -> Graph:
    """Process global callgraph into Graph encoder."""

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
    """Parse function CFG into Graph encoder."""
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


# endregion GRAPHS

# region SEQUENCES


def process_opcodes(
    func_code: FunctionCodeData,
    normalise: list[str] | Literal["all"] | None = None,
) -> Sequence:
    """Parse function opcode type tokens into Sequence encoder."""

    def token_iter() -> Iterator[str]:
        for op_type in func_code.iter_op_types():
            if OPTYPE_CANONICAL[op_type] in ("NOP", "UNK"):
                continue
            if normalise is None or (
                normalise != "all" and op_type not in normalise
            ):
                yield op_type
            else:
                yield OPTYPE_CANONICAL[op_type]

    return Sequence(token_iter())


def process_esil(
    func_code: FunctionCodeData,
    normalise: list[EsilTokenType] | Literal["all"] | None = None,
) -> Sequence:
    """Parse function ESIL tokens into Sequence encoder."""

    def token_iter() -> Iterator[str]:
        if normalise is None:
            yield from func_code.iter_esil()
            return
        for token_type, token in func_code.iter_type_esil():
            if normalise == "all" or token_type in normalise:
                yield token_type.name
            else:
                yield token

    return Sequence(token_iter())


def process_references(
    func_code: FunctionCodeData,
    normalise: list[ReferenceType] | Literal["all"] | None = None,
) -> Sequence:
    """Parse function's opcode references into Sequence encoder."""

    def token_iter() -> Iterator[str]:
        for reference in func_code.iter_opcode_refs():
            if normalise is None or (
                normalise != "all" and reference.ref_type not in normalise
            ):
                yield reference.ref_type.name
            else:
                yield reference.ref_kind.name

    return Sequence(token_iter())


def process_variables(
    func_zig: FunctionZignatureData,
    mode: Literal["all", "args", "locals"] = "all",
    keep_bits=True,
) -> Sequence:
    """Parse function's variable types into Sequence encoder."""

    def strip_bits(variable_type: str) -> str:
        return VAR_BITS_REGEX.sub(
            lambda m: m.group(1).lower(), variable_type
        ).strip()

    def token_iter() -> Iterator[str]:
        if mode in ("args", "all"):
            if keep_bits:
                yield from func_zig.arg_types
            else:
                yield from map(strip_bits, func_zig.arg_types)
        if mode in ("locals", "all"):
            if keep_bits:
                yield from func_zig.local_types
            else:
                yield from map(strip_bits, func_zig.local_types)

    return Sequence(token_iter())


# endregion SEQUENCES
