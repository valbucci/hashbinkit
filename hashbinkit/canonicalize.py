"""Map radare2 features into canonical groups."""

import re
from enum import Enum

import numpy as np
from numba import njit


def get_canonical_symbol(func_name: str) -> str:
    """Strip disassembler prefixes from function name.

    Radare2 inconsistently prefixes function names
    as sym.<n> or imp.<n> depending on context.
    This function strips those prefixes.
    """
    for prefix in ("sym.imp.", "sym.", "imp."):
        if func_name.startswith(prefix):
            func_name = func_name[len(prefix) :]
            break
    return func_name


# region VARIABLES

VAR_BITS_REGEX = re.compile(
    r"\b(u?int|ptr|char|long|short|float|double)(\d+_?t?)\b",
    flags=re.IGNORECASE,
)


def strip_var_bits(variable_type: str) -> str:
    """Strip variable data type bits from string."""
    return VAR_BITS_REGEX.sub(
        lambda m: m.group(1).lower(), variable_type
    ).strip()


# endregion VARIABLES

# region OPCODES

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
    "leave": "POP",
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
    "lea": "MATH",
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
    "rep": "NOP",
    # Failed or unrecognised disassembly
    "ill": "UNK",
    "unk": "UNK",
    "new": "UNK",
    "cast": "UNK",
    "length": "UNK",
}


def get_canonical_optype(op_type: str) -> str:
    """Return canonical opcode type or the given one if unknown."""
    try:
        return OPTYPE_CANONICAL[op_type]
    except KeyError:
        return op_type


# endregion OPCODES

# region ESIL

# Classify atomic Evaluable String Intermediate Language (ESIL)
# operation types (aka ESIL tokens).
# Source: <https://github.com/radareorg/radare2/blob/master/libr/esil/esil_ops.c>

# NOTE: EsilTokenType and get_esil_type were generated
#       using Claude Sonnet 4.6 including the above
#       source file as reference (AI:w).


class EsilTokenType(Enum):
    """Types of radare2 ESIL atomic operations."""

    ARITHMETIC = 1
    BITWISE = 2
    COMPARISON = 3
    ASSIGNMENT = 4
    STACK = 5
    FLAG_QUERY = 6
    CONTROL_FLOW = 7
    SYSTEM = 8
    MEM_READ = 9
    MEM_WRITE = 10
    FLOAT = 11
    MEM_RMW = 12
    VALUE = 13
    IDENTIFIER = 14

    @property
    def description(self) -> str:
        """Provides definition for each ESIL token type."""
        descriptions = {
            "ARITHMETIC": "Integer arithmetic operations (add, subtract, multiply, divide, modulo, increments).",
            "BITWISE": "Bitwise operations, logical shifts, rotations, and bit-counting (AND, OR, XOR, NOT, LSL, ROR, CLZ).",
            "COMPARISON": "Relational and equality evaluations (==, <, >, <=, >=) that feed status flags.",
            "ASSIGNMENT": "Register write operations, from strong/weak assignments (=, :=) to compound assignments (+=, &=).",
            "STACK": "ESIL evaluation stack manipulation keywords (DUP, SWAP, ROT, POP, CLEAR).",
            "FLAG_QUERY": "Internal CPU status flag evaluations ($z, $c, $b, $p, $s, $o, $ds, $jt, $js) tracking zero, carry, parity, sign, overflow, and delay-slot state.",
            "CONTROL_FLOW": "Conditional blocks, branch targets, and execution flow modifiers (?{, }, GOTO, BREAK).",
            "SYSTEM": "System-level interrupts, traps, meta commands, and architecture configuration (TRAP, BITS, $).",
            "MEM_READ": "Memory peek operations reading fixed byte widths ([1], [4], [8]) from an address.",
            "MEM_WRITE": "Memory poke operations writing fixed byte widths (=[1], =[4]) to an address.",
            "FLOAT": "Floating-point math, type conversions, and IEEE-754 comparisons (I2D, F+, SQRT).",
            "MEM_RMW": "Compound memory operations combining a load, an arithmetic or bitwise operation, and a store back to the same address (+=[4], &=[8]).",
            "VALUE": "Literal numeric constants (decimal integers or hexadecimal values like 0xffffffff).",
            "IDENTIFIER": "Variables, CPU general-purpose registers (r0, rax), or internal flag registers (nf, zf).",
        }
        return descriptions.get(self.name, "Unknown ESIL token category.")


ARITHMETIC_OPS = {"+", "-", "*", "L*", "/", "~/", "%", "~%", "++", "--"}
BITWISE_OPS = {
    "&",
    "|",
    "^",
    "!",
    "~",
    "<<",
    "LSL",
    ">>",
    "LSR",
    "ASR",
    "ROR",
    "ROL",
    "CLZ",
    "POPCNT",
}
COMPARISON_OPS = {"==", "<", ">", "<=", ">="}
ASSIGNMENT_OPS = {
    "=",
    ":=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "!=",  # NOT-assign (DUP,!,SWAP,=)
    "~=",
    "<<=",
    ">>=",
    "++=",
    "--=",
}
STACK_OPS = {"DUP", "SWAP", "ROT", "POP", "CLEAR", "STACK", "NUM"}
FLAG_QUERIES = {"$z", "$c", "$b", "$p", "$s", "$o", "$ds", "$jt", "$js"}
CONTROL_FLOW = {"?{", "}", "}{", "GOTO", "BREAK"}
SYSTEM_META = {
    "$",
    "()",
    "#!",
    "TRAP",
    "TODO",
    "BITS",
    "SETJT",
    "SETJTS",
    "SETD",
    "r=",
}
MEMORY_READS = {"[1]", "[2]", "[3]", "[4]", "[8]", "[16]", "[*]"}
MEMORY_WRITES = {"=[1]", "=[2]", "=[3]", "=[4]", "=[8]", "=[16]", "=[*]"}
FLOAT_OPS = {
    "I2D",
    "U2D",
    "D2I",
    "D2F",
    "F2D",
    "NAN",
    "F==",
    "F!=",
    "F<",
    "F<=",
    "F+",
    "F-",
    "F*",
    "F*+",
    "F/",
    "-F",
    "CEIL",
    "FLOOR",
    "ROUND",
    "SQRT",
}

RMW_REGEX = re.compile(r"^(\+\+|--|<<|>>|[|^&+\-\%/*])=\[(\d+|\*)\]$")
VALUE_REGEX = re.compile(r"^-?(?:0x[0-9a-fA-F]+|\d+)$")


def get_esil_type(token: str) -> EsilTokenType:
    """Classify radare2 ESIL atomic operation type."""
    if VALUE_REGEX.match(token):
        return EsilTokenType.VALUE
    if token in ARITHMETIC_OPS:
        return EsilTokenType.ARITHMETIC
    if token in BITWISE_OPS:
        return EsilTokenType.BITWISE
    if token in COMPARISON_OPS:
        return EsilTokenType.COMPARISON
    if token in ASSIGNMENT_OPS:
        return EsilTokenType.ASSIGNMENT
    if token in STACK_OPS:
        return EsilTokenType.STACK
    if token in FLAG_QUERIES:
        return EsilTokenType.FLAG_QUERY
    if token in CONTROL_FLOW:
        return EsilTokenType.CONTROL_FLOW
    if token in SYSTEM_META:
        return EsilTokenType.SYSTEM
    if token in MEMORY_READS:
        return EsilTokenType.MEM_READ
    if token in MEMORY_WRITES:
        return EsilTokenType.MEM_WRITE
    if token in FLOAT_OPS:
        return EsilTokenType.FLOAT
    if RMW_REGEX.match(token):
        return EsilTokenType.MEM_RMW
    return EsilTokenType.IDENTIFIER


# endregion ESIL

# region REFERENCES


class ReferenceType(Enum):
    """Radare2 reference types."""

    # Not using auto() for consistency
    # in case I was to add/remove types in the future.
    UNKN = 0
    CALL = 1
    JUMP = 2
    CODE = 3
    ICOD = 4
    DATA = 5
    STRN = 6
    NULL = 7


def get_reference_kind(ref_type: ReferenceType) -> ReferenceType:
    """Normalise reference types to kind groups."""
    match ref_type:
        case ReferenceType.CALL:
            # An inter-procedural reference (function call)
            return ReferenceType.CALL
        case ReferenceType.JUMP | ReferenceType.CODE | ReferenceType.ICOD:
            # An intra-procedural reference (CFG edge)
            return ReferenceType.CODE
        case ReferenceType.DATA | ReferenceType.STRN:
            # A read (DATA/STRN) or write (STRN) reference
            return ReferenceType.DATA
        case _:
            return ReferenceType.UNKN


# endregion REFERENCES

# region GRAPHS


# ---------------------------------------------------------------------
# In this region there are a few optimised functions for
# brute-forcing the smallest graph canonical labelling.
# ---------------------------------------------------------------------


@njit(cache=True)
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


@njit(cache=True)
def swap_array_values(a: int, b: int, array: np.ndarray) -> None:
    """Swap positions a <-> b in array."""
    tmp = array[a]
    array[a] = array[b]
    array[b] = tmp


@njit(cache=True)
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


@njit(cache=True)
def copy_array(dst, src) -> None:
    """Copy src into dst."""
    for i in range(len(dst)):
        dst[i] = src[i]


@njit(cache=True)
def get_adjacency_label(matrix) -> tuple[int, np.ndarray]:
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


# endregion GRAPHS
