# re-triton

MCP server exposing the Triton library for symbolic execution and constraint solving.

Version: 0.1.0 | License: MIT

## Structure

```
re-triton/
  pyproject.toml                    # build config (setuptools, mcp[cli] + deps)
  src/re_triton/
    __init__.py
    __main__.py                     # entry: from server import main; main()
    server.py                       # FastMCP app with @mcp.tool() functions
  README.md
  LICENSE
  SECURITY.md


```

## Build

```bash
pip install -e .                    # install with deps
re-triton                         # start MCP server on stdio
```



## Tools

This server exposes these MCP tools: `check_triton,emulate_function,symbolic_explore,solve_constraint,taint_analysis,find_magic_bytes,coverage_map`

## Usage (standalone)

Register this server in your `.mcp.json`:

```json
{
  "mcpServers": {
    "re-triton": {
      "command": "uv",
      "args": ["--directory", "/path/to/re-triton", "run", "re-triton"]
    }
  }
}
```

Or use via the [RE-AI agent-space](https://github.com/Heretek-RE/RE-AI): `./install.sh` clones all servers at pinned versions.
