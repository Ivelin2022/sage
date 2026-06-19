# SAGE MVP — three working LLM + SAGE case studies

Three small, runnable demos that show **SAGE working as the memory layer of an LLM
agent**. They are built to be screen-recorded for a website / portfolio.

> **Honest framing (please keep it in the videos and the site).** SAGE here is a
> *working* gradient-free bounded associative memory — functionally a vector store
> plus merge / decay / evict. These demos show it **working** and integrated with a
> local LLM. They do **not** claim SAGE beats a vector database or a k-means
> baseline; the rigorous study showed it *ties* standard methods. What is on show is
> a real, LLM-independent, continually-updated memory — which is genuinely useful.

## What each demo shows

| Demo | File | Shows |
|---|---|---|
| 1. Personal memory assistant | `demo1_assistant.py` | Remembers facts across sessions; recalls them into the prompt; keeps answering when the LLM is switched **off** (answers from memory). |
| 2. Drone agent | `demo2_drone.py` | A 12-step mission; the LLM is **cut off at step 9** (connection lost) and the agent keeps deciding from SAGE memory — zero "give up" defaults, millisecond decisions. |
| 3. Continual learning | `demo3_continual.py` | Teach new facts live with **no retraining**; the LLM alone can't answer them, **LLM + SAGE** can — including the *first* fact taught (no forgetting). |

## Setup (one time)

1. Install/run Ollama: https://ollama.com — then `ollama serve`.
2. Pull the models:
   ```
   ollama pull nomic-embed-text
   ollama pull mistral        # or any chat model you have
   ```
3. Use the project venv (has numpy). To pick a different LLM:
   ```
   set SAGE_LLM=llama3.2      # Windows PowerShell:  $env:SAGE_LLM="llama3.2"
   ```

## Run

From `SPHERE/sage_sphere/`:
```
python demos\demo1_assistant.py     # interactive; commands: /off /on /mem /quit
python demos\demo2_drone.py         # scripted mission, ~1 min
python demos\demo3_continual.py     # scripted teach-then-ask, ~1 min
```

### Demo 1 — suggested recording script (neutral, no personal info)
```
you> the weekly deployment runs every Tuesday at 09:00 UTC
you> the staging API key rotates on the first of each month
you> when does the weekly deployment run?   # -> recalls "Tuesday 09:00 UTC"
you> /off                                    # connection lost
you> when does the weekly deployment run?    # -> still answers from SAGE, no LLM
you> /mem                                    # show what SAGE is holding
you> /quit                                   # saved to disk
# re-run the script and ask again -> remembered across sessions
```

### Demo 2 — just run it
The mission plays automatically; at step 9 the banner
`*** CONNECTION TO GROUND STATION LOST ***` appears and the agent keeps flying on
SAGE memory. The summary prints zero defaults + the LLM-vs-SAGE timing.

### Demo 3 — just run it
It teaches 8 invented facts, then asks 3 questions, each answered **LLM-only** (fails)
then **LLM + SAGE** (correct), including the earliest fact.

## Recording tips
- Use a dark terminal (Windows Terminal) maximised; the demos use colour + clear
  banners for legibility. Set `NO_COLOR=1` to disable colour if needed.
- Demos 2 and 3 are deterministic and ~1 minute each — ideal short clips.
- Demo 1 is interactive — follow the script above for a clean ~90s clip.

## Files
- `_sage_demo.py` — shared lib: Ollama embed/chat (stdlib) + the text-storing
  `SageMemory` (faithful to `core/agent_memory.SAGEMemory`) + terminal UI helpers.
- `demo1_assistant.py`, `demo2_drone.py`, `demo3_continual.py` — the three demos.
- `data/` — created on first run (demo 1 saves its memory here).

## For the website
A good honest headline: **"SAGE — a working, weight-free, LLM-independent memory
for AI agents."** Lead with the three clips. Frame the contribution as the
*engineering* (gradient-free, bounded, persistent, degrades gracefully) and the
*honest study* (tested against strong baselines, reported straight) — that
combination is more credible, and more memorable, than an overclaim.
