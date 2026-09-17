#!/bin/bash
set -euo pipefail

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv could not be found on PATH. Install it with: curl -fsSL https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# Resolve the script's directory, regardless of the caller's cwd
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Install dependencies
uv sync --directory "$SCRIPT_DIR"

echo
echo "Setup complete. Register the server in Claude Code with:"
echo
echo "claude mcp add local-llm-offload -s user --env OLLAMA_HOST=YOUR_HOST --env OLLAMA_DEFAULT_MODEL=YOUR_MODEL -- uv run --directory \"$SCRIPT_DIR\" local-llm-offload"
echo
echo "Replace YOUR_HOST (e.g. http://192.168.1.50:11434) and YOUR_MODEL with your actual Ollama server and default model."
