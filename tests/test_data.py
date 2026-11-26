"""Test data loading, streaming, and serialization round-trips."""

from pathlib import Path

import pytest

from hashbinkit.data import (
    BasicBlockData,
    BinaryDataManager,
    OpcodeData,
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
