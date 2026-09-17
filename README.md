🌐 **English** | [Italiano](README.it.md)

# local-llm-offload

A standard [MCP](https://modelcontextprotocol.io) server that bridges an
MCP client (Claude Code, GitHub Copilot in VS Code or via CLI, or any
other compatible client) to an [Ollama](https://ollama.com) server on
your LAN, to offload low-risk mechanical tasks (code explanations,
documentation, boilerplate refactors, summaries) to local models —
keeping the "main" model's tokens for work that actually needs
reasoning/architecture.

Since it's a standard MCP server (stdio transport), there's nothing
client-specific about it: the same process
(`uv run --directory ... local-llm-offload`) is registered differently
depending on the client, but the code never changes. See
[Registering with an MCP client](#registering-with-an-mcp-client).

## Exposed tools

- **`delegate_to_local_llm(prompt, model?, context_files?)`** — sends a
  prompt to the Ollama server (OpenAI-compatible
  `/v1/chat/completions` endpoint, streamed) and returns the full text
  response. `context_files` accepts a list of paths to read and attach
  as context. If the MCP client supports progress notifications, it
  receives updates while the response is being generated (useful with
  slower models like `qwen3-coder:30b`). Every call is recorded in the
  local usage log (see `get_usage_stats`).
- **`list_local_models()`** — lists the models installed on the Ollama
  server (via the native `/api/tags` endpoint).
- **`get_usage_stats(since_hours?)`** — summarizes, per model, how many
  calls were made and how many *real* tokens (as reported by Ollama, not
  estimated) were processed locally — the concrete measure of how much
  work was actually offloaded from the main model.

## Requirements

- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/) for dependency management/execution
- An Ollama server reachable on the network (e.g. `192.168.1.50:11434`)

## Setup

```bash
uv sync
```

## Configuration

No `.env` file is loaded automatically by the server: environment
variables must be passed by the MCP client at registration time (see
below). `.env.example` in the repo documents the available variables and
is handy for manual local testing.

| Variable                  | Default                    | Description                                                       |
|----------------------------|----------------------------|--------------------------------------------------------------------|
| `OLLAMA_HOST`              | `http://localhost:11434`   | Base URL of the Ollama server (no trailing slash)                  |
| `OLLAMA_DEFAULT_MODEL`     | `qwen2.5-coder:7b`         | Model used when `delegate_to_local_llm` doesn't specify `model`     |
| `OLLAMA_TIMEOUT_SECONDS`   | `120`                      | HTTP timeout towards Ollama                                         |
| `MAX_CONTEXT_FILE_CHARS`   | `20000`                    | Cap per single file passed via `context_files`                     |
| `MAX_TOTAL_CONTEXT_CHARS`  | `60000`                    | Total cap across all combined `context_files`                      |
| `USAGE_LOG_PATH`           | `~/.local-llm-offload/usage.jsonl` | JSONL call log, read by `get_usage_stats`                   |
| `PROGRESS_CHUNK_INTERVAL`  | `20`                       | How many streamed chunks between MCP progress notifications         |

## Registering with an MCP client

### One-time setup per machine

Needed before registering the server with any client:

1. Install `uv`:
   - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`
2. Clone the repo (private — you need an authorized GitHub account, via
   `gh auth login` or an SSH key configured on the machine) and run
   setup:
   ```bash
   git clone https://github.com/suppressio/local-llm-offload.git
   cd local-llm-offload
   ./install.sh
   ```
   `install.sh` runs `uv sync` and prints a ready-to-use `claude mcp add`
   command with the absolute path already resolved (just replace host
   and model). On Windows, or if you prefer doing it by hand: `uv sync`.
3. Check that the Ollama server on your LAN is reachable from this
   machine (replace with the real IP):
   ```bash
   curl http://192.168.1.50:11434/api/tags
   ```

### Claude Code

```bash
claude mcp add local-llm-offload -s user \
  --env OLLAMA_HOST=http://192.168.1.50:11434 \
  --env OLLAMA_DEFAULT_MODEL=qwen2.5-coder:7b \
  -- uv run --directory /absolute/path/local-llm-offload local-llm-offload
```

Windows (PowerShell): same command, with `` ` `` instead of `\` for line
continuation and a Windows path (e.g. `C:\Dev\local-llm-offload`).

`-s user` makes the server available in all your Claude Code sessions on
that machine, not just the current project.

Verify with:

```bash
claude mcp list
claude mcp get local-llm-offload
```

### GitHub Copilot Chat in VS Code (agent mode)

Create `.vscode/mcp.json` in the workspace (or, to make it available
across all projects, open the Command Palette → "MCP: Open User
Configuration"):

```json
{
  "servers": {
    "local-llm-offload": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/local-llm-offload", "local-llm-offload"],
      "env": {
        "OLLAMA_HOST": "http://192.168.1.50:11434",
        "OLLAMA_DEFAULT_MODEL": "qwen2.5-coder:7b"
      }
    }
  }
}
```

### GitHub Copilot CLI

In `~/.copilot/mcp-config.json` (note: the root key here is
`mcpServers`, not `servers`):

```json
{
  "mcpServers": {
    "local-llm-offload": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/local-llm-offload", "local-llm-offload"],
      "env": {
        "OLLAMA_HOST": "http://192.168.1.50:11434",
        "OLLAMA_DEFAULT_MODEL": "qwen2.5-coder:7b"
      }
    }
  }
}
```

### Note: Copilot coding agent (cloud) is not supported

GitHub Copilot's "coding agent" (the one you assign an issue to, which
works in a sandbox on GitHub) runs in GitHub's cloud, not on your LAN: it
can't reach `OLLAMA_HOST` unless you expose Ollama to the internet, which
is not recommended for security reasons. The configurations above apply
to clients that run locally on your machine (Claude Code, VS Code,
Copilot CLI).

### After registering

MCP tools are loaded when a session starts: if you register the server
while a session is already open, you need to open a new one for
`delegate_to_local_llm` and `list_local_models` to show up among the
available tools.

## Usage examples

During a session with an MCP client (Claude Code, Copilot Chat in agent
mode, etc.), call the tools explicitly, for example:

> "Use `list_local_models` to see what's available on Ollama."

> "Use `delegate_to_local_llm` with model `qwen2.5-coder:7b` to write
> docstrings for `src/foo.py`, passing it as `context_files`."

> "Delegate to `delegate_to_local_llm` with model `qwen3-coder:30b` a
> refactor of the `bar.py` module to extract the validation logic into a
> separate function."

> "Use `get_usage_stats` to see how many tokens we've offloaded to
> Ollama in the last 24 hours."

## Tests

`tests/test_smoke.py` makes real calls against the Ollama server
configured in `OLLAMA_HOST` (no mocking) and is automatically skipped if
the server isn't reachable. `tests/test_usage_log.py` covers the
logging/aggregation logic without any network access:

```bash
export OLLAMA_HOST=http://192.168.1.50:11434  # or: set -a && source .env && set +a
uv run pytest
```

## Roadmap

- [x] Streaming responses (progress reporting during generation)
- [x] Token/time savings metrics (`get_usage_stats`, real tokens from Ollama)
