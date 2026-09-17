"""MCP server exposing a local Ollama instance as explicit, callable tools.

This is deliberately *not* wired for automatic delegation: Claude Code (or
any MCP client) must be told explicitly to call `delegate_to_local_llm`.
The goal at this stage is to validate that the bridge works and that it
actually saves tokens, before building any automatic routing on top.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server import MCPServer

from . import config, ollama_client

mcp = MCPServer("local-llm-offload")


def _load_context_files(paths: list[str]) -> str:
    """Read and concatenate context files, applying size caps.

    Missing or unreadable files are reported inline instead of raising, so
    one bad path doesn't blow up an otherwise-fine request.
    """
    blocks: list[str] = []
    total_chars = 0

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        header = f"--- {raw_path} ---"

        if not path.is_file():
            blocks.append(f"{header}\n[file non trovato o non leggibile]")
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            blocks.append(f"{header}\n[errore di lettura: {exc}]")
            continue

        if len(text) > config.MAX_CONTEXT_FILE_CHARS:
            text = (
                text[: config.MAX_CONTEXT_FILE_CHARS]
                + f"\n[...troncato, file più lungo di {config.MAX_CONTEXT_FILE_CHARS} caratteri...]"
            )

        if total_chars + len(text) > config.MAX_TOTAL_CONTEXT_CHARS:
            remaining = max(config.MAX_TOTAL_CONTEXT_CHARS - total_chars, 0)
            text = text[:remaining] + "\n[...troncato, limite totale di contesto raggiunto...]"
            blocks.append(f"{header}\n{text}")
            total_chars += len(text)
            break

        blocks.append(f"{header}\n{text}")
        total_chars += len(text)

    return "\n\n".join(blocks)


@mcp.tool()
async def delegate_to_local_llm(
    prompt: str,
    model: str | None = None,
    context_files: list[str] | None = None,
) -> str:
    """Delega un task testuale a un LLM locale servito da Ollama.

    Utile per task meccanici a basso rischio (spiegazioni di codice,
    documentazione, refactor banali, riassunti) da eseguire su hardware
    locale invece di consumare token di Claude. Non viene invocato
    automaticamente: va richiamato esplicitamente.

    Args:
        prompt: L'istruzione/domanda da inviare al modello locale.
        model: Nome del modello Ollama da usare (es. "qwen2.5-coder:7b" per
            task rapidi, "qwen3-coder:30b" per task più corposi). Se omesso,
            usa OLLAMA_DEFAULT_MODEL.
        context_files: Percorsi di file da includere come contesto, letti e
            allegati al prompt (con troncamento se troppo grandi).

    Returns:
        Il testo generato dal modello locale, oppure un messaggio di errore
        chiaro se il server Ollama non è raggiungibile.
    """
    chosen_model = model or config.OLLAMA_DEFAULT_MODEL

    full_prompt = prompt
    if context_files:
        context_block = _load_context_files(context_files)
        if context_block:
            full_prompt = f"{context_block}\n\n--- richiesta ---\n{prompt}"

    try:
        return await ollama_client.chat_completion(full_prompt, model=chosen_model)
    except ollama_client.OllamaError as exc:
        return f"[local-llm-offload] Errore: {exc}"


@mcp.tool()
async def list_local_models() -> str:
    """Elenca i modelli disponibili sul server Ollama locale (via /api/tags).

    Utile per scoprire quali modelli sono installati prima di scegliere
    quale passare a `delegate_to_local_llm`.

    Returns:
        Un elenco testuale di modelli (nome, dimensione, data modifica),
        oppure un messaggio di errore chiaro se il server non è raggiungibile.
    """
    try:
        models = await ollama_client.list_models()
    except ollama_client.OllamaError as exc:
        return f"[local-llm-offload] Errore: {exc}"

    if not models:
        return f"Nessun modello trovato su {config.OLLAMA_HOST}."

    lines = [f"Modelli disponibili su {config.OLLAMA_HOST}:"]
    for entry in models:
        name = entry.get("name", "?")
        size_bytes = entry.get("size")
        size_str = f"{size_bytes / 1e9:.1f} GB" if isinstance(size_bytes, (int, float)) else "?"
        modified = entry.get("modified_at", "?")
        lines.append(f"- {name} ({size_str}, modificato: {modified})")

    return "\n".join(lines)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
