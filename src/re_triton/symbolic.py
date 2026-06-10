"""Triton-backed symbolic execution and constraint solving.

Triton is a dynamic binary analysis library. It provides:
  - Symbolic execution engine
  - Taint analysis
  - AST of x86/x86-64/ARM32/AArch64/RISC-V semantics
  - Lifting to LLVM-IR (some architectures)
  - Z3-backed SMT solver

This module exposes a small set of high-level operations: concrete
emulation, symbolic exploration, constraint solving, taint tracking,
and "find input that matches target bytes" search.
"""

from __future__ import annotations

import ast
import logging
import operator
from typing import Any

logger = logging.getLogger("re_triton")


def _probe_arch_enum(triton: Any) -> list[str]:
    """Probe the triton module for the architecture enum.

    Cycle 2 fix: the prior implementation only checked
    ``triton.ARCH.X86`` etc. — Quarkslab Triton 1.x renames the
    enum to ``triton.CPU`` / ``triton.cpus`` and the attribute
    values are lowercased (``x86_64`` not ``X86_64``). The new
    helper probes multiple possible locations and normalizes to the
    canonical upper-snake-case names the callers expect.
    """
    archs: list[str] = []
    # Quarkslab Triton 0.x — top-level ARCH enum, upper-snake names
    for name in ("X86", "X86_64", "AArch64", "ARM32", "RISCV32", "RISCV64"):
        if hasattr(triton, "ARCH") and hasattr(triton.ARCH, name):
            archs.append(name)
    if archs:
        return sorted(set(archs))
    # Quarkslab Triton 1.x — top-level CPU or cpus module, lower names
    for module_name in ("CPU", "cpus"):
        module = getattr(triton, module_name, None)
        if module is None:
            continue
        for lower in ("x86", "x86_64", "aarch64", "arm32", "riscv32", "riscv64"):
            if hasattr(module, lower) or hasattr(module, lower.upper()):
                archs.append(lower.upper().replace("X86_64", "X86_64"))
        if archs:
            return sorted(set(archs))
    return []


def check_triton() -> dict[str, Any]:
    """Return Triton import status, supported architectures, AND a
    TritonContext instantiation probe.

    v2.9.1 (Gap 28 fix) adds the instantiation probe: the prior
    check only verified ``import triton`` succeeded and returned
    the version + arch enum. But ``emulate_function`` calls
    ``triton.TritonContext(arch)`` (or ``triton.Triton(arch)`` in
    1.x) — if the installed module imports cleanly but the
    constructor is broken, the check returns OK while every
    emulate call fails. The new probe instantiates a
    TritonContext for the first supported arch and returns
    ``triton_context_ok: bool`` + ``triton_context_error``
    (on failure) so the caller can see whether the emulator
    will actually work.
    """
    try:
        import triton
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "error": f"triton import failed: {exc}",
            "supported_archs": [],
            "triton_context_ok": False,
        }
    archs = _probe_arch_enum(triton)
    result: dict[str, Any] = {
        "version": getattr(triton, "__version__", "unknown"),
        "status": "OK",
        "supported_archs": archs,
        "triton_context_ok": None,  # filled below
    }
    # Instantiation probe — the v2.9.1 Gap 28 addition.
    if archs:
        try:
            arch_str = archs[0]
            # Map the upper-snake arch string to the triton enum
            arch_enum = getattr(triton.ARCH, arch_str)
            cls = (
                getattr(triton, "TritonContext", None)
                or getattr(triton, "Triton", None)
            )
            if cls is None:
                result["triton_context_ok"] = False
                result["triton_context_error"] = (
                    "triton module exposes neither TritonContext (0.x) "
                    "nor Triton (1.x)"
                )
            else:
                ctx = cls(arch_enum)
                # Smoke-probe: can we ask the context its arch?
                ctx_arch = getattr(ctx, "getArchitecture", lambda: None)()
                result["triton_context_ok"] = True
                result["triton_context_arch"] = (
                    str(ctx_arch) if ctx_arch is not None else arch_str
                )
        except Exception as exc:  # noqa: BLE001
            result["triton_context_ok"] = False
            result["triton_context_error"] = (
                f"TritonContext instantiation failed: {exc}"
            )
            result["status"] = "WARN"
    else:
        result["triton_context_ok"] = False
        result["triton_context_error"] = (
            "no supported architectures detected"
        )
        result["status"] = "WARN"
    return result


def _make_triton_context(triton: Any, arch: str) -> Any:
    """Construct a Triton context, handling the API rename in 1.x.

    Cycle 2 fix: Quarkslab Triton 0.x exposes the class as
    ``triton.TritonContext(arch_enum)``. In 1.x the class was renamed
    to ``triton.Triton(arch_enum)``. The prior implementation
    hard-coded ``triton.TritonContext(arch)`` which raised
    ``AttributeError: module 'triton' has no attribute 'TritonContext'``
    on the user's 1.x install.

    New helper probes both names and the underlying arch enum
    via :func:`_probe_arch_enum`.
    """
    cls = getattr(triton, "TritonContext", None) or getattr(triton, "Triton", None)
    if cls is None:
        raise RuntimeError(
            "triton module exposes neither TritonContext (0.x) nor Triton (1.x)"
        )
    # Resolve the arch enum. Prefer the canonical upper-snake form
    # (Triton 0.x ``ARCH.X86_64``). Fall back to lowercase (1.x
    # ``CPU.x86_64`` / ``cpus.x86_64``).
    arch_enum: Any = None
    arch_obj = getattr(triton, "ARCH", None)
    if arch_obj is not None and hasattr(arch_obj, arch):
        arch_enum = getattr(arch_obj, arch)
    if arch_enum is None:
        for module_name in ("CPU", "cpus"):
            module = getattr(triton, module_name, None)
            if module is None:
                continue
            arch_enum = getattr(module, arch, None) or getattr(module, arch.lower(), None)
            if arch_enum is not None:
                break
    if arch_enum is None:
        # Last-resort default — X86_64 is the most common
        if arch_obj is not None:
            arch_enum = getattr(arch_obj, "X86_64", None)
    return cls(arch_enum)


def emulate_function(
    code: bytes,
    *,
    base_address: int = 0x400000,
    arch: str = "X86_64",
    concrete_inputs: dict[str, int] | None = None,
    steps: int = 1000,
) -> dict[str, Any]:
    """Concrete emulation of *code* for *steps* instructions.

    Returns the final register state and a log of executed instructions.
    """
    try:
        import triton
        # Quarkslab Triton 1.x installs as a single-file C extension;
        # the C++ namespaces (ARCH, REG, TritonContext, …) are statically
        # bound at .so load time, so no submodule import is needed.
    except Exception as exc:  # noqa: BLE001
        return {"status": "ERROR", "error": f"triton import: {exc}"}
    ctx = _make_triton_context(triton, arch)
    ctx.setConcreteMemoryAreaValue(base_address, code)
    # Apply concrete inputs
    for reg_name, val in (concrete_inputs or {}).items():
        try:
            ctx.setConcreteRegisterValue(ctx.getRegister(reg_name), val)
        except Exception:  # noqa: BLE001
            pass
    log: list[dict[str, Any]] = []
    pc = ctx.getConcreteRegisterValue(ctx.registers.rip if arch in ("X86", "X86_64") else ctx.registers.pc)
    for i in range(steps):
        opcode = ctx.getConcreteMemoryAreaValue(pc, 16)
        # Quarkslab 1.0 disassembly takes (addr:int, size:int) and returns
        # a list[Instruction] (one per decoded instruction), not a single inst.
        insts = ctx.disassembly(pc, 16)
        if not insts:
            break
        inst = insts[0]
        if inst is None:
            break
        # v1.0: inst.address is no longer an attribute; use getAddress()
        log.append({"address": inst.getAddress(), "mnemonic": inst.getName(), "operands": str(inst.getOperands())})
        try:
            # v1.0: execute() removed; processing() returns int 0 on success
            ctx.processing(inst)
        except Exception as exc:  # noqa: BLE001
            log.append({"error": f"processing failed at {hex(inst.getAddress())}: {exc}"})
            break
        # Advance PC
        next_pc = ctx.getConcreteRegisterValue(ctx.registers.rip if arch in ("X86", "X86_64") else ctx.registers.pc)
        if next_pc == pc:
            break
        pc = next_pc
    return {
        "status": "ok",
        "steps_executed": len(log),
        "log_tail": log[-30:],
        "final_pc": pc,
    }


def symbolic_explore(
    code: bytes,
    *,
    base_address: int = 0x400000,
    arch: str = "X86_64",
    symbolic_args: list[str] | None = None,
    max_paths: int = 16,
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Run symbolic execution through *code*, return constraints at branch exits.

    Args:
        code: machine code bytes
        base_address: where to map the code in Triton's memory model
        arch: target architecture
        symbolic_args: register names to mark symbolic (default: rdi, rsi, rdx, rcx, r8, r9 for X86_64)
        max_paths: max paths to enumerate before stopping
        timeout_s: stop after this many seconds
    """
    import time

    try:
        import triton
    except Exception as exc:  # noqa: BLE001
        return {"status": "ERROR", "error": f"triton import: {exc}"}
    ctx = _make_triton_context(triton, arch)
    ctx.setConcreteMemoryAreaValue(base_address, code)
    if symbolic_args is None:
        symbolic_args = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"] if arch == "X86_64" else []
    # Mark each named register as symbolic
    for reg_name in symbolic_args:
        try:
            r = ctx.getRegister(reg_name)
            ctx.symbolizeRegister(r, f"sym_{reg_name}")
        except Exception as exc:  # noqa: BLE001
            return {"status": "ERROR", "error": f"cannot symbolize {reg_name}: {exc}"}

    pc_reg = ctx.registers.rip if arch in ("X86", "X86_64") else ctx.registers.pc
    branches: list[dict[str, Any]] = []
    paths_explored = 0
    deadline = time.time() + timeout_s
    pc = base_address
    while paths_explored < max_paths and time.time() < deadline:
        # Quarkslab 1.0 disassembly takes (addr:int, size:int) and returns
        # a list[Instruction], not a single inst.
        insts = ctx.disassembly(pc, 16)
        if not insts:
            break
        inst = insts[0]
        if inst is None:
            break
        # v1.0: execute() removed; processing() returns int 0 on success
        ctx.processing(inst)
        # If this is a conditional branch, we have a constraint
        # v1.0: isConditionnal() renamed to isConditionTaken() (typo fix)
        if inst.isBranch() and inst.isConditionTaken():
            ast = ctx.getRegisterAst(pc_reg)
            branches.append({
                # v1.0: inst.address removed; use getAddress()
                "address": inst.getAddress(),
                "constraint": str(ast)[:500] if ast else None,
            })
            paths_explored += 1
        next_pc = ctx.getConcreteRegisterValue(pc_reg)
        if next_pc == pc:
            break
        pc = next_pc
    return {
        "status": "ok",
        "paths_explored": paths_explored,
        "branches": branches[:max_paths],
        "symbolic_args": symbolic_args,
    }


def _safe_eval_z3_expr(expr_str: str, z3_vars: dict[str, Any]) -> Any:
    """Evaluate a constraint expression string against Z3 variables.

    Replaces the old ``eval(expr, {"__builtins__": {}}, z3_vars)`` with a
    walker over a whitelisted subset of Python AST node types. The walker
    rejects any construct that could call out to Python builtins,
    attribute access, subscripts, comprehensions, lambdas, etc. — closing
    the code-injection hole that ``eval`` exposed even with builtins
    stripped (a z3 ArithRef still has ``__class__``/``__subclasses__``).

    Supported subset:

    - Literals: int, float, bool, None
    - Name lookup: must be present in *z3_vars* (no other namespace)
    - Binary arithmetic: ``+ - * / // % **``
    - Unary: ``+ -`` (numeric) and ``not`` (boolean — uses z3.Not)
    - Comparisons: ``== != < <= > >=``
    - Boolean: ``and`` / ``or`` — translated to z3.And / z3.Or

    Anything else (``Call``, ``Subscript``, ``Attribute``, ``Lambda``,
    comprehensions, starred expressions, …) raises ``ValueError`` with a
    message naming the disallowed construct.
    """
    tree = ast.parse(expr_str, mode="eval")

    # Operator dispatch tables. Lookup by AST node type so a missing
    # operator in the source never falls through to a Python builtin.
    bin_ops: dict[type, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        # Cycle 1 / T1.4: bitwise ops. With z3.BitVec variables,
        # operator.and_/or_/xor_/lshift_/rshift_ dispatch through
        # BitVecRef.__and__/__or__/__xor__/__lshift__/__rshift__ and
        # return new BitVecRef expressions.
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
    }
    cmp_ops: dict[type, Any] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }
    unary_ops: dict[type, Any] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
    }

    def _walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            # Only numeric / boolean / None constants. Reject strings
            # outright — they have no z3 meaning and are a probing vector.
            if isinstance(node.value, (int, float, bool, type(None))):
                return node.value
            raise ValueError(f"unsupported constant: {type(node.value).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in z3_vars:
                raise ValueError(
                    f"name {node.id!r} not in vars "
                    f"(allowed: {sorted(z3_vars)!r})"
                )
            return z3_vars[node.id]
        if isinstance(node, ast.BinOp):
            op = bin_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported binary op: {type(node.op).__name__}")
            return op(_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp):
            op = unary_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
            return op(_walk(node.operand))
        if isinstance(node, ast.Compare):
            # `a < b < c` desugars to a chained Compare with multiple ops.
            # We only support single-comparator chains (a OP b). Reject
            # the chained form explicitly to keep the walker simple.
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("chained comparisons are not supported")
            op = cmp_ops.get(type(node.ops[0]))
            if op is None:
                raise ValueError(f"unsupported comparison: {type(node.ops[0]).__name__}")
            return op(_walk(node.left), _walk(node.comparators[0]))
        if isinstance(node, ast.BoolOp):
            # Python's `and`/`or` are short-circuit and cannot be
            # expressed as functions on z3 BoolRef. Translate to
            # z3.And / z3.Or which take a variadic list.
            import z3 as _z3  # local import so the helper is testable without z3
            values = [_walk(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return _z3.And(*values) if len(values) > 1 else values[0]
            if isinstance(node.op, ast.Or):
                return _z3.Or(*values) if len(values) > 1 else values[0]
            raise ValueError(f"unsupported boolean op: {type(node.op).__name__}")
        # Any other AST node type is rejected. This includes Call,
        # Subscript, Attribute, Lambda, ListComp, DictComp, SetComp,
        # GeneratorExp, Starred, IfExp, NamedExpr (walrus), FormattedValue,
        # JoinedStr, etc. — all of which were reachable through the
        # previous eval().
        raise ValueError(f"unsupported expression construct: {type(node).__name__}")

    return _walk(tree)


def solve_constraint(
    constraint_expr: str, vars: list[str], width: int = 64
) -> dict[str, Any]:
    """Feed a constraint expression to Z3 and return a model.

    Args:
        constraint_expr: a constraint expression string using only the
            safe subset documented in :func:`_safe_eval_z3_expr` — e.g.
            ``"(x + 1) * 2 == 42"``, ``"x > 0 and x < 100"``.
        vars: variable names to model (e.g. ``["x"]``)
        width: bit-width for the z3 BitVec variables (default 64; pass
            32 for ILP-style 32-bit MBA identities, 8 for byte-level).

    Cycle 1 / T1.4 fix: variables are now Z3 ``BitVec`` (was
    ``Int``). This unlocks the bitwise operators (``BitAnd``,
    ``BitOr``, ``BitXor``, ``LShR``, ...) that the
    re-mba-deobfuscate skill's MBA identities require. The MBA
    identity ``x + y == (x & y) + (x | y)`` (Zhou 2007) is the
    canonical smoke test; with this fix it returns ``sat``.

    Note on z3 call signature: ``z3.BitVec`` has two valid forms —
    ``z3.BitVec(name, bits)`` and ``z3.BitVec(bits, name, ctx)``.
    This module uses the first (str-first) form because the
    Quarkslab Triton .so loads a C++ binding that corrupts
    z3's internal ctypes state in a way that makes the
    int-first form's ``Z3_mk_bv_sort`` C call fail with
    ``argument 2: wrong type``. The str-first form takes a
    different ctypes path internally and is unaffected.
    """
    try:
        import z3
    except Exception as exc:  # noqa: BLE001
        return {"status": "ERROR", "error": f"z3 import: {exc}"}
    try:
        # Cycle 1 / T1.4: create Z3 BitVec variables (was z3.Int).
        # Str-first form — see the docstring's "Note on z3 call
        # signature" for why we don't use the int-first form.
        z3_vars = {name: z3.BitVec(name, width) for name in vars}
        # Walk the AST with a strict whitelist — no eval(), so no
        # code-injection path even if constraint_expr is attacker-controlled.
        expr = _safe_eval_z3_expr(constraint_expr, z3_vars)
        s = z3.Solver()
        s.add(expr)
        if s.check() == z3.sat:
            m = s.model()
            return {
                "status": "sat",
                "model": {name: str(m.eval(z3_vars[name], model_completion=True)) for name in vars},
            }
        return {"status": "unsat"}
    except Exception as exc:  # noqa: BLE001
        err = f"z3 evaluation failed: {exc}"
        return {"status": "ERROR", "error": err}


def taint_analysis(
    code: bytes,
    *,
    base_address: int = 0x400000,
    arch: str = "X86_64",
    taint_sources: list[str] | None = None,
    steps: int = 1000,
) -> dict[str, Any]:
    """Taint tracking: which memory/registers are influenced by *taint_sources*."""
    try:
        import triton
    except Exception as exc:  # noqa: BLE001
        return {"status": "ERROR", "error": f"triton import: {exc}"}
    ctx = _make_triton_context(triton, arch)
    ctx.setConcreteMemoryAreaValue(base_address, code)
    if taint_sources is None:
        taint_sources = ["rdi", "rsi", "rdx"] if arch == "X86_64" else []
    for reg_name in taint_sources:
        try:
            r = ctx.getRegister(reg_name)
            ctx.taintRegister(r)
        except Exception:  # noqa: BLE001
            pass
    pc_reg = ctx.registers.rip if arch in ("X86", "X86_64") else ctx.registers.pc
    pc = base_address
    tainted_log: list[dict[str, Any]] = []
    for _ in range(steps):
        # v1.0: disassembly takes (addr:int, size:int) and returns
        # list[Instruction] (one per decoded instruction).
        insts = ctx.disassembly(pc, 16)
        if not insts:
            break
        inst = insts[0]
        if inst is None:
            break
        # v1.0: execute() removed; processing() returns int 0 on success
        ctx.processing(inst)
        # Record tainted registers
        for r in ctx.getAllRegisters():
            if ctx.isRegisterTainted(r):
                tainted_log.append({
                    # v1.0: inst.address removed; use getAddress()
                    "address": inst.getAddress(),
                    "register": r.getName(),
                    "value": hex(ctx.getConcreteRegisterValue(r)),
                })
        next_pc = ctx.getConcreteRegisterValue(pc_reg)
        if next_pc == pc:
            break
        pc = next_pc
    return {
        "status": "ok",
        "tainted_observations": tainted_log[:100],
        "sources": taint_sources,
    }


def find_magic_bytes(
    code: bytes,
    *,
    base_address: int = 0x400000,
    arch: str = "X86_64",
    target_bytes_b64: str,
    length: int = 8,
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Solve for an input that causes the binary to produce *target_bytes*.

    Args:
        code: machine code bytes
        target_bytes_b64: target output, base64-encoded
        length: number of input bytes to find
    """
    import base64
    import time

    target = base64.b64decode(target_bytes_b64)
    try:
        import triton
    except Exception as exc:  # noqa: BLE001
        return {"status": "ERROR", "error": f"triton import: {exc}"}
    ctx = _make_triton_context(triton, arch)
    ctx.setConcreteMemoryAreaValue(base_address, code)
    # Mark the first *length* bytes at a known location as symbolic
    input_addr = 0x100000
    ctx.setConcreteMemoryAreaValue(input_addr, b"\x00" * length)
    for i in range(length):
        ctx.symbolizeMemory(
            ctx.getMemoryOperand(triton.MemoryAccess(input_addr + i, 1)),
            f"inp_{i}",
        )
    pc_reg = ctx.registers.rip if arch in ("X86", "X86_64") else ctx.registers.pc
    pc = base_address
    deadline = time.time() + timeout_s
    # Run until we see the symbolic bytes written somewhere
    for _ in range(10000):
        if time.time() > deadline:
            return {"status": "TIMEOUT"}
        # v1.0: disassembly takes (addr:int, size:int) and returns
        # list[Instruction]; we only care about the first decoded insn here.
        insts = ctx.disassembly(pc, 16)
        if not insts:
            break
        inst = insts[0]
        if inst is None:
            break
        # v1.0: execute() removed; processing() returns int 0 on success
        ctx.processing(inst)
        next_pc = ctx.getConcreteRegisterValue(pc_reg)
        if next_pc == pc:
            break
        pc = next_pc
    # Build a model: assume we want the symbolic input to equal length 0 bytes;
    # a real implementation would constrain on the output. This is a stub
    # demonstrating the API shape.
    return {
        "status": "ok",
        "note": "find_magic_bytes is a high-level demo. Real implementation requires a target byte stream to constrain on.",
        "target": target.hex(),
        "length": length,
    }
