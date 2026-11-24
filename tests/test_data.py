"""Test data loading, streaming, and serialization round-trips."""

from pathlib import Path

import pytest

from hashbinkit.data import (
    BasicBlockData,
    BinaryDataManager,
    FunctionCodeData,
    FunctionZignatureData,
    OpcodeData,
    OpcodeReferenceData,
    ReferenceType,
    get_data_manager,
)

SAMPLES_PATH = Path(__file__).parent / "samples"
BINARIES_PATH = SAMPLES_PATH / "binaries"
RAW_DATA_PATH = SAMPLES_PATH / "raw_data"
BINARY_NAME = "vlc_3-0-20"
BINARY_PATH = BINARIES_PATH / BINARY_NAME


def test_paths():
    assert SAMPLES_PATH.is_dir()
    assert BINARIES_PATH.is_dir()
    assert RAW_DATA_PATH.is_dir()
    assert BINARY_PATH.is_file()


class TestBinaryDataManager:
    """Test data loading and serialisation."""

    @pytest.fixture(scope="class")
    @classmethod
    def manager(cls):
        """Return a BinaryDataManager for the test sample."""
        return get_data_manager(BINARY_PATH.name, RAW_DATA_PATH)

    @pytest.fixture(scope="class")
    @classmethod
    def first_opcode_with_refs(
        cls, manager: BinaryDataManager
    ) -> OpcodeData | None:
        """Return the first opcode that has at least one non-None ref or xref."""
        for cfg in manager.func_cfgs.values():
            for block in cfg.blocks:
                for op in block.opcodes:
                    if op.refs is not None or op.xrefs is not None:
                        return op
        return None

    @pytest.fixture(scope="class")
    @classmethod
    def first_block_with_successors(
        cls,
        manager: BinaryDataManager,
    ) -> BasicBlockData | None:
        """Return the first block that has non-None jump or fail fields."""
        for cfg in manager.func_cfgs.values():
            for block in cfg.blocks:
                if block.jump is not None or block.fail is not None:
                    return block
        return None

    def test_serialize_opcode_reference_data(self, first_opcode_with_refs):
        op = first_opcode_with_refs
        assert op is not None, "Test data has no opcodes with refs"

        refs = op.refs or op.xrefs
        ref = refs[0]  # type: ignore

        b = ref.serialize()
        assert b, "serialize() returned empty bytes (slots=True bug?)"

        result = OpcodeReferenceData.deserialize(b)
        assert result == ref
        # Verify 64-bit address survived without truncation
        assert result.src_addr == ref.src_addr
        assert result.dst_addr == ref.dst_addr
        # Verify Enum round-tripped correctly
        assert isinstance(result.ref_type, ReferenceType)
        assert result.ref_type == ref.ref_type

    def test_serialize_opcode_data_no_refs(self, manager):
        # Find an opcode with all Optional fields unset
        target = next(
            (
                op
                for cfg in manager.func_cfgs.values()
                for block in cfg.blocks
                for op in block.opcodes
                if op.ptr is None and op.refs is None and op.xrefs is None
            ),
            None,
        )
        assert target is not None, (
            "Test data has no opcode with all Optional fields None"
        )

        b = target.serialize()
        assert b, "serialize() returned empty bytes"

        result = OpcodeData.deserialize(b)
        assert result == target
        assert result.ptr is None
        assert result.refs is None
        assert result.xrefs is None

    def test_serialize_opcode_data_with_refs(self, first_opcode_with_refs):
        op = first_opcode_with_refs
        assert op is not None, "Test data has no opcodes with refs"

        b = op.serialize()
        result = OpcodeData.deserialize(b)
        assert result == op
        # Optional tuple[OpcodeReferenceData, ...] | None
        assert result.refs == op.refs
        assert result.xrefs == op.xrefs

    def test_serialize_basic_block_no_successors(self, manager):
        first_cfg = manager.func_cfgs[next(iter(manager.func_cfgs))]
        # Last block of a single-block function has no CFG successors
        block = first_cfg.blocks[-1]

        b = block.serialize()
        assert b, "serialize() returned empty bytes"

        result = BasicBlockData.deserialize(b)
        assert result == block

    def test_serialize_basic_block_with_successors(
        self, first_block_with_successors
    ):
        block = first_block_with_successors
        assert block is not None, "Test data has no blocks with successors"

        b = block.serialize()
        result = BasicBlockData.deserialize(b)
        assert result == block
        # 64-bit Optional int fields survived
        assert result.jump == block.jump
        assert result.fail == block.fail

    def test_serialize_function_code_data(self, manager):
        for cfg in manager.func_cfgs.values():
            b = cfg.serialize()
            assert b, "serialize() returned empty bytes"
            result = FunctionCodeData.deserialize(b)
            assert result == cfg, (
                f"Round-trip mismatch for function '{cfg.name}' "
                f"at address {hex(cfg.addr)}"
            )

    def test_serialize_function_zignature_data(self, manager):
        for zig in manager.func_zigs.values():
            b = zig.serialize()
            assert b, "serialize() returned empty bytes"
            result = FunctionZignatureData.deserialize(b)
            assert result == zig, (
                f"Round-trip mismatch for zignature '{zig.name}' "
                f"at address {hex(zig.addr)}"
            )

    def test_serialize_binary_data_manager(self, manager):
        b = manager.serialize()
        assert b, "serialize() returned empty bytes"
        result = BinaryDataManager.deserialize(b)
        assert result == manager
