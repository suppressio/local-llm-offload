"""MCP server exposing a local Ollama instance as explicit, callable tools.

This is deliberately *not* wired for automatic delegation: Claude Code (or
any MCP client) must be told explicitly to call `delegate_to_local_llm`.
The goal at this stage is to validate that the bridge works and that it
actually saves tokens, before building any automatic routing on top.
"""

from __future__ import annotations

import time
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver import Context

from . import config, ollama_client, usage_log

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
    ctx: Context | None = None,
) -> str:
    """Delega un task testuale a un LLM locale servito da Ollama.

    Utile per task meccanici a basso rischio (spiegazioni di codice,
    documentazione, refactor banali, riassunti) da eseguire su hardware
    locale invece di consumare token di Claude. Non viene invocato
    automaticamente: va richiamato esplicitamente.

    La risposta viene generata in streaming: se il client lo supporta,
    riceve notifiche di progresso durante la generazione (utile per
    modelli lenti come qwen3-coder:30b), ma il valore ritornato è sempre
    il testo completo.

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

    total_chars = 0
    chunks_since_update = 0

    async def on_chunk(text: str) -> None:
        nonlocal total_chars, chunks_since_update
        total_chars += len(text)
        chunks_since_update += 1
        if ctx is not None and chunks_since_update >= config.PROGRESS_CHUNK_INTERVAL:
            chunks_since_update = 0
            await ctx.report_progress(total_chars, message=f"{chosen_model}: {total_chars} caratteri generati")

    try:
        result = await ollama_client.chat_completion_stream(full_prompt, model=chosen_model, on_chunk=on_chunk)
    except ollama_client.OllamaError as exc:
        usage_log.record(usage_log.UsageEntry(timestamp=time.time(), model=chosen_model, ok=False, elapsed_seconds=0.0))
        return f"[local-llm-offload] Errore: {exc}"

    usage_log.record(
        usage_log.UsageEntry(
            timestamp=time.time(),
            model=chosen_model,
            ok=True,
            elapsed_seconds=result.elapsed_seconds,
            prompt_tokens=result.usage.get("prompt_tokens", 0),
            completion_tokens=result.usage.get("completion_tokens", 0),
            total_tokens=result.usage.get("total_tokens", 0),
        )
    )

    return result.text


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


@mcp.tool()
async def get_usage_stats(since_hours: float | None = None) -> str:
    """Riassume l'uso reale di delegate_to_local_llm, per validare il risparmio di token.

    Legge il log locale delle chiamate (token reali riportati da Ollama,
    non stime) e lo aggrega per modello. Usalo per rispondere alla domanda
    "quanti token abbiamo davvero scaricato sul modello locale finora?".

    Args:
        since_hours: Se specificato, considera solo le chiamate delle
            ultime N ore. Se omesso, usa l'intera cronologia disponibile.

    Returns:
        Un riepilogo testuale per modello (chiamate, token, tempo), oppure
        un messaggio se non ci sono ancora dati.
    """
    summary = usage_log.summarize(since_hours=since_hours)

    if summary.calls == 0:
        period = f"nelle ultime {since_hours}h" if since_hours else "finora"
        return f"Nessuna chiamata a delegate_to_local_llm registrata {period}."

    period = f"ultime {since_hours}h" if since_hours else "intera cronologia"
    lines = [
        f"Uso di delegate_to_local_llm ({period}):",
        f"Totale: {summary.calls} chiamate, {summary.total_tokens} token, "
        f"{summary.elapsed_seconds:.1f}s di generazione locale.",
        "",
    ]
    for model, m in sorted(summary.by_model.items(), key=lambda kv: kv[1].total_tokens, reverse=True):
        failed = m.calls - m.ok_calls
        failed_note = f", {failed} falliti" if failed else ""
        avg_tok_s = m.completion_tokens / m.elapsed_seconds if m.elapsed_seconds > 0 else 0.0
        lines.append(
            f"- {model}: {m.calls} chiamate{failed_note}, "
            f"{m.prompt_tokens} token prompt + {m.completion_tokens} token risposta "
            f"= {m.total_tokens} totali, {m.elapsed_seconds:.1f}s (~{avg_tok_s:.1f} tok/s)"
        )

    return "\n".join(lines)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
