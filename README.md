# re-triton

MCP server exposing the [Triton](https://github.com/JonathanSalwan/Triton) library for symbolic execution, taint analysis, and constraint solving.

## Tools

| Tool | What it does |
|---|---|
| `check_triton` | Confirm Triton is importable |
| `emulate_function` | Concrete emulation (no sym) |
| `symbolic_explore` | Symbolic execution through a function |
| `solve_constraint` | Z3-based constraint solver |
| `taint_analysis` | Taint tracking through a function |
| `find_magic_bytes` | Solve for input that matches target bytes |

## Install

```bash
pip install -e ./servers/re-triton
```

The `pyproject.toml` pins `triton @ git+https://github.com/JonathanSalwan/Triton`,
which pulls Quarkslab's binary-analysis framework directly from its source repo.
The PyPI `triton` package is the GPU compiler and is **not** what `re-triton`
needs. Building Triton from source requires CMake + a C++ toolchain; the install
can take a few minutes the first time.

**Best-effort on Windows** — if the source build doesn't install, the server
will return a clean "Triton not available" error from `check_triton`.

## Why Triton

Triton is the *easiest* of the symbolic-execution frameworks to embed in a Python tool. It pairs with the `capstone` disassembly (which we already use in `re-lief`). For binary-only symbolic exec on x86/x64/AArch64, this is the right choice. For more advanced use cases (deep program analysis, complex constraints), `angr` is the gold standard — that's a v2 candidate.

## Note on raw bytes

Triton operates on raw machine code, not files. The MCP tools accept `code_b64` (base64-encoded bytes) — the caller (Claude Code, via `re-rizin.disassemble_function` and friends) extracts the relevant bytes from the binary.
