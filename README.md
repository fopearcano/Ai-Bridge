# Ai-Bridge_Houdini

**A natural-language bridge for Houdini automation.**

You type "create a sphere" or "scale the selected nodes by 2"; the bridge
asks an LLM (OpenAI, Anthropic, or LM Studio) to produce a small piece of
Houdini Python; it gates the result through a safety analyzer; and — only
if you allow it — runs that code in Houdini and shows you what happened.

Ai-Bridge_Houdini is **not** an AI artist. It does not invent shots,
suggest looks, or make creative decisions. It is a translation layer
between your words and `hou.*` calls, with explicit confirmation steps.

```
your words ──▶ LLM ──▶ JSON plan ──▶ safety gate ──▶ hython / receiver ──▶ Houdini
              ▲                          │
              │                          ▼
        scene snapshot ◀──────── (optional) live receiver
```

---

## Install

Python **3.11+** required.

```bash
git clone <this-repo> Ai-Bridge
cd Ai-Bridge

pip install -e .          # CLI only
pip install -e .[gui]     # CLI + Qt UI (PySide6)
```

You can also run from a fresh checkout without installing — the launcher
scripts (`run_cli.sh` / `run_cli.bat` / `run_ui.sh` / `run_ui.bat`) put
`./src` on `PYTHONPATH` for you.

---

## Configure `.env`

Copy the template and fill in the bits you need:

```bash
cp .env.example .env
$EDITOR .env
```

Required per provider:

| Provider   | Required vars                                  |
| ---------- | ---------------------------------------------- |
| `openai`    | `OPENAI_API_KEY` (also optional `OPENAI_MODEL`) |
| `anthropic` | `ANTHROPIC_API_KEY` (also `ANTHROPIC_MODEL`)   |
| `lmstudio`  | `LMSTUDIO_MODEL` (and `LMSTUDIO_BASE_URL` if not on `localhost:1234`) |

Other knobs:

| Variable           | Purpose                                                              |
| ------------------ | -------------------------------------------------------------------- |
| `PROVIDERS`        | Comma list of providers the router is allowed to use (default: all) |
| `DEFAULT_PROVIDER` | Which provider is active at startup                                  |
| `MODE`             | `safe` (default), `dev`, or `direct` — see *Safety notes* below      |
| `HYTHON_PATH`      | Absolute path to Houdini's `hython` binary; required to **execute**  |
| `HOUDINI_HOST` / `HOUDINI_PORT` | Where the in-Houdini receiver listens (default `127.0.0.1:18861`) |
| `AIBRIDGE_LOG_LEVEL` / `AIBRIDGE_LOG_DIR` | Standard logging knobs                          |

Verify the loaded config with secrets masked:

```bash
./run_cli.sh --check-config
```

---

## Run the Houdini receiver (optional)

When the receiver is running inside an interactive Houdini session, the
bridge fetches your live scene state and forwards it to the LLM as
context — `/obj` children, current selection, cameras, lights, geo
containers, and the current frame. Without the receiver, the bridge
still works; it just sends prompts without scene context.

In Houdini's Python Source Editor, a shelf tool, or `123.py`:

```python
from aibridge_houdini.houdini.houdini_receiver import HoudiniReceiver
rx = HoudiniReceiver(host="127.0.0.1", port=18861)
rx.start_in_background()
```

Or headless via `hython`:

```bash
"$HYTHON_PATH" -c "from aibridge_houdini.houdini.houdini_receiver import main; main()"
```

The receiver speaks length-prefixed JSON over loopback only:

```
{"type": "exec",          "code": "...", "request_id": "..."}
{"type": "inspect_scene", "request_id": "..."}
```

It refuses to bind to anything outside `{127.0.0.1, localhost}` and
double-checks the client address on every connection. It runs
**untrusted code via `exec()`** in your Houdini process — see *Safety
notes* below before exposing it anywhere.

---

## CLI usage

```bash
./run_cli.sh           # macOS / Linux
run_cli.bat            # Windows
# or after pip install:
aibridge-houdini
```

Inside the REPL, type a prompt and press Enter. The bridge:

1. Fetches scene context (if the receiver is running).
2. Asks the active LLM for a plan.
3. Validates the JSON; auto-repairs once if it's malformed.
4. Runs the safety gate against the generated `houdini_python`.
5. Asks for confirmation in `safe` mode, blocks in any mode for hard
   findings.
6. Executes via `hython` (a separate process; no shared state with your
   interactive session) when allowed.

Slash commands:

| Command                                          | Effect                                          |
| ------------------------------------------------ | ----------------------------------------------- |
| `/provider`                                      | Show active and enabled providers              |
| `/provider openai\|anthropic\|lmstudio`          | Switch the active provider at runtime          |
| `/mode`                                          | Show the current safety mode                    |
| `/mode safe\|dev\|direct`                        | Change the safety mode                          |
| `/help` (or `/?`)                                | Print the command list                          |
| `/quit`, `/exit`, `exit`, `quit`, `:q`           | Leave                                           |

Other useful flags:

```bash
./run_cli.sh --version
./run_cli.sh --check-config
./run_cli.sh --env-file path/to/other.env
```

---

## UI usage

```bash
./run_ui.sh            # macOS / Linux
run_ui.bat             # Windows
# or:
aibridge-houdini --ui qt
```

The window has:

- **Provider** and **Mode** dropdowns (top bar).
- **Prompt** input (multi-line) with **Send** / **Clear**.
- **Response** panel — plan summary or clarification question.
- **Code** panel — the generated Houdini Python (read-only, monospace).
- **Log** panel — live log lines from the bridge.
- **Status bar** — provider, mode, transport, busy state.

In `safe` mode, a confirmation dialog appears before any code runs. Click
**No** to skip execution; the plan stays in the panels for review.

LLM calls and `hython` runs happen on background threads, so the window
never freezes during a turn.

---

## Safety notes

The bridge assumes the LLM may produce surprising or unsafe code. It
defends in three places:

1. **System prompt** (`prompts/houdini_operator_system.txt`): forbids
   `os.system`, `subprocess`, sockets, `urllib`, `eval`, `exec`,
   `compile`, `__import__`, scene-destruction calls
   (`hou.hipFile.clear()` etc.), and external file I/O outside
   `$HIP`/`$JOB`/`$TEMP`. Also requires JSON-only output and documents
   the clarification protocol.

2. **Safety gate** (`aibridge_houdini.execution.safety.evaluate`): static
   AST analysis on the returned `houdini_python`. Verdicts:

   | Verdict              | Meaning                                                  |
   | -------------------- | -------------------------------------------------------- |
   | `safe`               | No findings — run automatically in `direct` mode         |
   | `needs_confirmation` | Soft hits — prompt before executing (default in `safe`) |
   | `blocked`            | Hard hits — refuse to execute, in any mode               |

   Modes change leniency, but **critical findings are always blocked,
   even in `direct`**.

3. **Execution layer**: `hython` mode launches a separate process with
   the script written to a temp file that's deleted on exit
   (`HythonTransport`). The optional in-Houdini receiver
   (`HoudiniReceiver`) only accepts loopback connections, executes one
   request per fresh namespace, captures stdout/stderr, and never lets
   `SystemExit` or `BaseException` from the user's code crash the
   server thread.

Recommended posture for a new user:

- Start with `MODE=safe` and review every plan before confirming.
- Watch the Code panel for any `os` / `subprocess` / network calls — if
  one slips through, file it as a bug; the safety layer should catch it.
- Run the live receiver only when you're actively iterating; stop it
  when you're done. **Never expose its port off-host.**

For clarity on what's intended to be safe vs. dangerous, see
`tests/manual_houdini_prompts.md` (catalog of expected behaviors) and
the `execution/safety.py` rules.

---

## Project layout

```
src/aibridge_houdini/
├── app.py                  # CLI entrypoint + slash-command parser
├── config.py               # .env loading, validation, masked summary
├── types.py                # LLMPlan, Command, UserRequest, RiskLevel
├── providers/
│   ├── router.py           # provider selection, scene-context injection, auto-repair
│   ├── openai_provider.py  # OpenAI Responses API + strict JSON schema
│   ├── lmstudio_provider.py# LM Studio via the OpenAI-compatible chat API
│   └── placeholder.py      # stand-in until the Anthropic adapter lands
├── houdini/
│   ├── bridge.py           # HoudiniBridge: pick a transport, run code
│   ├── transport.py        # HythonTransport (subprocess), SocketTransport (placeholder)
│   ├── inspection.py       # gather scene state from `hou`
│   ├── scene_client.py     # host-side client to the receiver
│   └── houdini_receiver.py # the in-Houdini server (loopback only)
├── execution/safety.py     # AST-based static analyzer, MODE-aware
└── ui/
    ├── cli.py              # CLI rendering helpers
    └── qt_app.py           # PySide6 desktop UI (optional)
prompts/houdini_operator_system.txt
scripts/run_manual_test.py  # live end-to-end catalog runner
tests/                      # pytest suite (no network); plus manual catalog
```

---

## Optional: live prompt catalog

For end-to-end checks against a real LLM:

```bash
python scripts/run_manual_test.py --list                  # list ids
python scripts/run_manual_test.py --only sphere           # one prompt
python scripts/run_manual_test.py --execute --mode direct # full pipeline
python scripts/run_manual_test.py --output runs/x.json    # diffable transcript
```

See `tests/manual_houdini_prompts.md` for the catalog of eight prompts
and what to verify in Houdini for each.
