"""MCP server entry point for re-triton."""

from __future__ import annotations

import base64
import logging

from mcp.server.fastmcp import FastMCP

# The `re_triton.symbolic` import pulls in the Quarkslab Triton framework,
# which is a different package from the GPU-compiler `triton` (the two share
# the PyPI name and the Quarkslab build is a C++ source build — see
# servers/re-triton/README.md). When the wrong `triton` is installed, the
# `triton.arch` import inside `re_triton.symbolic` raises ImportError. We
# catch it at module top so the MCP server's @mcp.tool() decorators still
# run and the tools are *registered* (they return a structured error per
# call instead of failing the whole server).
try:
    from re_triton import symbolic  # type: ignore[import-not-found]
    _SYMBOLIC_IMPORT_ERROR: str | None = None
except ImportError as _exc:
    symbolic = None  # type: ignore[assignment]
    _SYMBOLIC_IMPORT_ERROR = (
        f"re-triton requires the Quarkslab Triton framework (a C++ source build); "
        f"the GPU-compiler `triton` is installed instead. Import error: {_exc}"
    )

logger = logging.getLogger("re_triton")
logger.setLevel(logging.INFO)

mcp = FastMCP("re-triton")


def _check_symbolic() -> dict | None:
    """Return a structured error if the Quarkslab Triton import failed."""
    if symbolic is None:
        return {
            "status": "ERROR",
            "error": "triton_unavailable",
            "message": _SYMBOLIC_IMPORT_ERROR or "Quarkslab triton not importable",
            "install_hint": (
                "pip install quarkslab-triton (or build from source per "
                "servers/re-triton/README.md)"
            ),
        }
    return None


@mcp.tool()
def check_triton() -> dict:
    """Return Triton import status and supported architectures."""
    err = _check_symbolic()
    if err is not None:
        return err
    return symbolic.check_triton()


@mcp.tool()
def emulate_function(
    code_b64: str,
    base_address: int = 0x400000,
    arch: str = "X86_64",
    concrete_inputs: dict[str, int] | None = None,
    steps: int = 1000,
) -> dict:
    """Concrete emulation of machine code for *steps* instructions.

    Args:
        code_b64: machine code bytes, base64-encoded
        base_address: where to map the code
        arch: X86 / X86_64 / AArch64 / ARM32 / RISCV32 / RISCV64
        concrete_inputs: register_name → value map
        steps: max instructions to execute
    """
    err = _check_symbolic()
    if err is not None:
        return err
    code = base64.b64decode(code_b64)
    return symbolic.emulate_function(
        code,
        base_address=base_address,
        arch=arch,
        concrete_inputs=concrete_inputs or {},
        steps=steps,
    )


@mcp.tool()
def symbolic_explore(
    code_b64: str,
    base_address: int = 0x400000,
    arch: str = "X86_64",
    symbolic_args: list[str] | None = None,
    max_paths: int = 16,
    timeout_s: int = 30,
) -> dict:
    """Run symbolic execution, return constraints at branch exits."""
    err = _check_symbolic()
    if err is not None:
        return err
    code = base64.b64decode(code_b64)
    return symbolic.symbolic_explore(
        code,
        base_address=base_address,
        arch=arch,
        symbolic_args=symbolic_args,
        max_paths=max_paths,
        timeout_s=timeout_s,
    )


@mcp.tool()
def solve_constraint(constraint_expr: str, vars: list[str]) -> dict:
    """Feed a constraint expression to Z3 and return a model."""
    err = _check_symbolic()
    if err is not None:
        return err
    return symbolic.solve_constraint(constraint_expr, vars)


@mcp.tool()
def taint_analysis(
    code_b64: str,
    base_address: int = 0x400000,
    arch: str = "X86_64",
    taint_sources: list[str] | None = None,
    steps: int = 1000,
) -> dict:
    """Taint tracking: which memory/registers are influenced by *taint_sources*."""
    err = _check_symbolic()
    if err is not None:
        return err
    code = base64.b64decode(code_b64)
    return symbolic.taint_analysis(
        code,
        base_address=base_address,
        arch=arch,
        taint_sources=taint_sources,
        steps=steps,
    )


@mcp.tool()
def find_magic_bytes(
    code_b64: str,
    target_bytes_b64: str,
    base_address: int = 0x400000,
    arch: str = "X86_64",
    length: int = 8,
    timeout_s: int = 30,
) -> dict:
    """Solve for an input that produces the target output bytes."""
    err = _check_symbolic()
    if err is not None:
        return err
    return symbolic.find_magic_bytes(
        base64.b64decode(code_b64),
        base_address=base_address,
        arch=arch,
        target_bytes_b64=target_bytes_b64,
        length=length,
        timeout_s=timeout_s,
    )


@mcp.tool()
def coverage_map(
    code_b64: str,
    base_address: int = 0x400000,
    arch: str = "X86_64",
    concrete_inputs: dict[str, int] | None = None,
    steps: int = 1000,
) -> dict:
    """Lift machine code to a coverage map: edges hit + blocks seen.

    Wraps :func:`emulate_function` + :func:`symbolic_explore` and
    summarises the result as a single coverage map. The map is the
    input the ``re-fuzz-replay`` skill uses to track new basic
    blocks reached by an input corpus.

    Args:
        code_b64: machine code bytes, base64-encoded
        base_address: where the code is mapped in virtual memory
        arch: X86 / X86_64 / AArch64 / ARM32 / RISCV32 / RISCV64
        concrete_inputs: optional register_name → value map
        steps: max instructions to execute

    Returns::

        {
          "arch": "X86_64",
          "base_address": 0x400000,
          "edges": [["0x...->0x...", ...], ...],
          "blocks": ["0x...", ...],
          "block_count": N,
          "edge_count": M,
          "signature": "..."
        }

    The ``signature`` is a hash of the sorted (block, edge) pairs;
    ``re-fuzz-replay.seed_replay`` uses it to detect new basic
    blocks reached by a corpus replay.
    """
    err = _check_symbolic()
    if err is not None:
        return err
    code = base64.b64decode(code_b64)
    # First do a concrete emulation to gather an edge list.
    emu = symbolic.emulate_function(
        code,
        base_address=base_address,
        arch=arch,
        concrete_inputs=concrete_inputs or {},
        steps=steps,
    )
    # Then do a (separate) symbolic exploration to gather a block
    # set; the two views cross-validate each other.
    sym = symbolic.symbolic_explore(
        code,
        base_address=base_address,
        arch=arch,
        symbolic_args=None,
        max_paths=8,
        timeout_s=10,
    )
    blocks = sym.get("blocks_visited", []) if isinstance(sym, dict) else []
    edges = emu.get("edges", []) if isinstance(emu, dict) else []
    import hashlib
    sig_src = "\n".join(sorted(blocks)) + "||" + "\n".join(
        sorted("->".join(e) if isinstance(e, (list, tuple)) else str(e) for e in edges)
    )
    signature = hashlib.sha256(sig_src.encode("utf-8")).hexdigest()[:16]
    return {
        "arch": arch,
        "base_address": base_address,
        "edges": edges,
        "blocks": blocks,
        "block_count": len(blocks),
        "edge_count": len(edges),
        "signature": signature,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
