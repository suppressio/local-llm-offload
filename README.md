# local-llm-offload

Server [MCP](https://modelcontextprotocol.io) standard che fa da ponte tra
un client MCP (Claude Code, GitHub Copilot in VS Code o via CLI, o
qualunque altro client compatibile) e un server [Ollama](https://ollama.com)
in LAN, per delegare a modelli locali i task meccanici a basso rischio
(spiegazioni di codice, documentazione, refactor banali, riassunti) e
ridurre il consumo di token del modello "principale", tenendolo per il
lavoro che richiede reasoning/architettura reale.

Essendo un server MCP standard (transport stdio), non c'è nulla di
specifico per un client in particolare: lo stesso processo
(`uv run --directory ... local-llm-offload`) si registra in modo diverso a
seconda del client, ma il codice non cambia. Vedi
[Registrazione in un client MCP](#registrazione-in-un-client-mcp).

**Stato attuale**: nessuna delegazione automatica. I tool vanno richiamati
esplicitamente durante una sessione — l'obiettivo di questa fase è
validare che il ponte funzioni e che il risparmio di token sia reale,
prima di automatizzare oltre.

## Tool esposti

- **`delegate_to_local_llm(prompt, model?, context_files?)`** — invia un
  prompt al server Ollama (endpoint OpenAI-compatible
  `/v1/chat/completions`) e ritorna la risposta testuale. `context_files`
  accetta una lista di percorsi da leggere e allegare come contesto.
- **`list_local_models()`** — elenca i modelli installati sul server
  Ollama (via `/api/tags` nativo).

## Requisiti

- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/) per gestione dipendenze/esecuzione
- Un server Ollama raggiungibile in rete (es. `192.168.1.50:11434`)

## Setup

```bash
uv sync
```

## Configurazione

Nessun file `.env` viene caricato automaticamente dal server: le variabili
d'ambiente vanno passate dal client MCP al momento della registrazione
(vedi sotto). `.env.example` nel repo documenta le variabili disponibili
ed è utile per test manuali in locale.

| Variabile                 | Default                    | Descrizione                                                       |
|----------------------------|----------------------------|--------------------------------------------------------------------|
| `OLLAMA_HOST`              | `http://localhost:11434`   | URL base del server Ollama (senza slash finale)                    |
| `OLLAMA_DEFAULT_MODEL`     | `qwen2.5-coder:7b`         | Modello usato quando `delegate_to_local_llm` non specifica `model`  |
| `OLLAMA_TIMEOUT_SECONDS`   | `120`                      | Timeout HTTP verso Ollama                                           |
| `MAX_CONTEXT_FILE_CHARS`   | `20000`                    | Cap per singolo file passato in `context_files`                    |
| `MAX_TOTAL_CONTEXT_CHARS`  | `60000`                    | Cap totale su tutti i `context_files` combinati                    |

## Registrazione in un client MCP

### Setup una tantum per macchina

Serve prima di registrare il server in qualunque client:

1. Installa `uv`:
   - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`
2. Clona il repo (privato — serve un account GitHub autorizzato, via
   `gh auth login` o una chiave SSH configurata sulla macchina):
   ```bash
   git clone https://github.com/suppressio/local-llm-offload.git
   cd local-llm-offload
   uv sync
   ```
3. Verifica che il server Ollama in LAN sia raggiungibile dalla macchina
   (sostituisci con l'IP reale):
   ```bash
   curl http://192.168.1.50:11434/api/tags
   ```

### Claude Code

```bash
claude mcp add local-llm-offload -s user \
  --env OLLAMA_HOST=http://192.168.1.50:11434 \
  --env OLLAMA_DEFAULT_MODEL=qwen2.5-coder:7b \
  -- uv run --directory /percorso/assoluto/local-llm-offload local-llm-offload
```

Windows (PowerShell): stesso comando, con `` ` `` al posto di `\` per
andare a capo e un percorso Windows (es. `C:\Dev\local-llm-offload`).

`-s user` rende il server disponibile in tutte le sessioni Claude Code
sulla macchina, non solo nel progetto corrente.

Verifica con:

```bash
claude mcp list
claude mcp get local-llm-offload
```

### GitHub Copilot Chat in VS Code (agent mode)

Crea `.vscode/mcp.json` nel workspace (oppure, per renderlo disponibile in
tutti i progetti, apri la Command Palette → "MCP: Open User
Configuration"):

```json
{
  "servers": {
    "local-llm-offload": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/percorso/assoluto/local-llm-offload", "local-llm-offload"],
      "env": {
        "OLLAMA_HOST": "http://192.168.1.50:11434",
        "OLLAMA_DEFAULT_MODEL": "qwen2.5-coder:7b"
      }
    }
  }
}
```

### GitHub Copilot CLI

In `~/.copilot/mcp-config.json` (nota: qui la chiave radice è
`mcpServers`, non `servers`):

```json
{
  "mcpServers": {
    "local-llm-offload": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/percorso/assoluto/local-llm-offload", "local-llm-offload"],
      "env": {
        "OLLAMA_HOST": "http://192.168.1.50:11434",
        "OLLAMA_DEFAULT_MODEL": "qwen2.5-coder:7b"
      }
    }
  }
}
```

### Nota: Copilot coding agent (cloud) non è supportato

Il "coding agent" di GitHub Copilot (quello a cui assegni una issue e che
lavora in una sandbox su GitHub) gira nel cloud di GitHub, non sulla tua
LAN: non può raggiungere `OLLAMA_HOST` a meno di esporre Ollama su
internet, cosa sconsigliata per sicurezza. Le configurazioni sopra
valgono per client che girano localmente sulla tua macchina (Claude Code,
VS Code, Copilot CLI).

### Dopo la registrazione

I tool MCP vengono caricati all'avvio di una sessione: se la registrazione
avviene mentre una sessione è già aperta, serve aprirne una nuova perché
`delegate_to_local_llm` e `list_local_models` compaiano tra gli strumenti
disponibili.

## Esempi d'uso

Durante una sessione con un client MCP (Claude Code, Copilot Chat in
agent mode, ecc.), richiama esplicitamente i tool, ad esempio:

> "Usa `list_local_models` per vedere cosa c'è disponibile su Ollama."

> "Usa `delegate_to_local_llm` con model `qwen2.5-coder:7b` per scrivere
> le docstring di `src/foo.py`, passandolo come `context_files`."

> "Delega a `delegate_to_local_llm` con model `qwen3-coder:30b` un
> refactor del modulo `bar.py` per estrarre la logica di validazione in
> una funzione separata."

## Test

Uno smoke test esegue chiamate reali contro il server Ollama configurato
in `OLLAMA_HOST` (nessun mock) e viene automaticamente saltato se il
server non è raggiungibile:

```bash
export OLLAMA_HOST=http://192.168.1.50:11434  # o: set -a && source .env && set +a
uv run pytest
```

## Roadmap

- [ ] Delegazione automatica (il client decide quando instradare a Ollama)
- [ ] Streaming delle risposte
- [ ] Metriche di risparmio token/tempo per confrontare modello principale vs LLM locale
