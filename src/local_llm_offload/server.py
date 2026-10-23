"""MCP server exposing a local Ollama instance as callable tools.

The server-level `instructions` below are injected by MCP clients (e.g.
Claude Code) into the model's system prompt, so the client model delegates
suitable tasks on its own instead of only when explicitly asked.
"""

from __future__ import annotations

import time
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver import Context

from . import config, ollama_client, usage_log

INSTRUCTIONS = """\
local-llm-offload ti permette di delegare lavoro a un LLM locale (Ollama), \
risparmiando i tuoi token. Usalo di tua iniziativa, senza aspettare che l'utente \
lo chieda, ogni volta che un sotto-task è autocontenuto, meccanico e facile da \
verificare, ad esempio:
- scrivere docstring, commenti o documentazione per codice esistente;
- boilerplate, scaffolding di test, conversioni di formato (JSON/YAML/CSV...);
- riassumere file lunghi, log o output voluminosi che altrimenti dovresti leggere per intero;
- bozze di messaggi di commit, changelog, traduzioni.

Passa i file tramite `context_files` invece di leggerli tu: è lì che sta il \
risparmio maggiore, perché il contenuto non entra nel tuo contesto.

Per i task meccanici passa `think="off"` o `think="low"`: i modelli che \
ragionano rispondono molto prima e il risultato non peggiora.

Non delegare: decisioni architetturali, debugging non banale, codice di \
sicurezza, modifiche coordinate su più file, o qualsiasi cosa in cui verificare \
il risultato costi più che farla da te. Rivedi sempre l'output prima di usarlo: \
il modello locale è meno affidabile di te.
"""

mcp = MCPServer("local-llm-offload", instructions=INSTRUCTIONS)


def _load_context_files(paths: list[str]) -> tuple[str, list[str]]:
    """Read and concatenate context files, applying size caps.

    Missing or unreadable files are reported inline instead of raising, so
    one bad path doesn't blow up an otherwise-fine request.

    Returns the concatenated block plus the paths that were cut or skipped
    by the caps: the markers inside the block only reach the local model,
    so the caller needs this list to tell its own client.
    """
    blocks: list[str] = []
    cut: list[str] = []
    total_chars = 0

    for i, raw_path in enumerate(paths):
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
            cut.append(raw_path)
            text = (
                text[: config.MAX_CONTEXT_FILE_CHARS]
                + f"\n[...troncato, file più lungo di {config.MAX_CONTEXT_FILE_CHARS} caratteri...]"
            )

        if total_chars + len(text) > config.MAX_TOTAL_CONTEXT_CHARS:
            remaining = max(config.MAX_TOTAL_CONTEXT_CHARS - total_chars, 0)
            text = text[:remaining] + "\n[...troncato, limite totale di contesto raggiunto...]"
            blocks.append(f"{header}\n{text}")
            total_chars += len(text)
            if raw_path not in cut:
                cut.append(raw_path)
            cut.extend(paths[i + 1 :])
            break

        blocks.append(f"{header}\n{text}")
        total_chars += len(text)

    return "\n\n".join(blocks), cut


def _truncation_warnings(cut_files: list[str], result: ollama_client.ChatResult) -> str:
    """Notes for the calling model when the local model didn't see the whole input."""
    notes: list[str] = []
    if cut_files:
        notes.append(
            f"questi file sono stati tagliati o esclusi dai limiti di caratteri "
            f"(MAX_CONTEXT_FILE_CHARS={config.MAX_CONTEXT_FILE_CHARS}, "
            f"MAX_TOTAL_CONTEXT_CHARS={config.MAX_TOTAL_CONTEXT_CHARS}): {', '.join(cut_files)}"
        )
    if result.truncated:
        notes.append(
            f"Ollama ha scartato parte del prompt: ha elaborato solo "
            f"{result.usage.get('prompt_tokens', 0)} token, troppo pochi per la lunghezza inviata. "
            f"Contesto usato: {result.context_window} token (OLLAMA_NUM_CTX={config.OLLAMA_NUM_CTX}, "
            f"limitato al massimo del modello). Passa meno contesto, alza OLLAMA_NUM_CTX "
            f"o usa un modello con contesto più ampio"
        )
    if not notes:
        return ""
    lines = "\n".join(f"- {n}." for n in notes)
    return (
        "[local-llm-offload] ATTENZIONE: il modello locale NON ha visto tutto l'input, "
        f"la risposta qui sotto può essere incompleta o sbagliata.\n{lines}\n\n"
    )


@mcp.tool()
async def delegate_to_local_llm(
    prompt: str,
    model: str | None = None,
    context_files: list[str] | None = None,
    think: str | None = None,
    ctx: Context | None = None,
) -> str:
    """Delega un task testuale a un LLM locale servito da Ollama.

    Usalo di tua iniziativa per task meccanici a basso rischio
    (documentazione, boilerplate, refactor banali, riassunti di file o log
    lunghi) da eseguire su hardware locale invece di consumare i tuoi
    token. Vedi le istruzioni del server per quando delegare e quando no.

    La risposta viene generata in streaming: se il client lo supporta,
    riceve notifiche di progresso durante la generazione (utile per
    modelli grandi e lenti), ma il valore ritornato è sempre il testo
    completo.

    Args:
        prompt: L'istruzione/domanda da inviare al modello locale.
        model: Nome del modello Ollama da usare, tra quelli installati
            (vedi `list_local_models`). Se omesso, usa OLLAMA_DEFAULT_MODEL:
            lascialo vuoto salvo motivi precisi.
        context_files: Percorsi di file da includere come contesto, letti e
            allegati al prompt (con troncamento se troppo grandi).
        think: Quanto far ragionare il modello prima di rispondere: "off",
            "low", "medium", "high" o "on". Usa "off" o "low" per task
            meccanici (riassunti, docstring, conversioni), che così sono
            molto più veloci; lascialo vuoto o "on" per codice non banale.
            Se omesso usa OLLAMA_THINK; stringa vuota = default del
            modello. Ignorato dai modelli che non ragionano.

    Returns:
        Il testo generato dal modello locale, oppure un messaggio di errore
        chiaro se il server Ollama non è raggiungibile. Se il modello non ha
        visto tutto l'input (file tagliati dai limiti, o prompt troncato da
        Ollama), la risposta inizia con un avviso "ATTENZIONE".
    """
    chosen_model = model or config.OLLAMA_DEFAULT_MODEL

    full_prompt = prompt
    cut_files: list[str] = []
    if context_files:
        context_block, cut_files = _load_context_files(context_files)
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
        result = await ollama_client.chat_completion_stream(
            full_prompt, model=chosen_model, on_chunk=on_chunk, think=think
        )
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

    return _truncation_warnings(cut_files, result) + result.text


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
