# Running SignOrder locally

Two processes: the Next.js terminal and the Python ML service. Start both.

## 1. ML service (sign recognition)

```bash
cd services/ml
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

Check it: <http://127.0.0.1:8000/health>

```json
{ "ok": true, "sign_model": "stub", "recommender": "none", "feature_length": 153 }
```

`"sign_model": "stub"` is expected until `ml/asl` exports a trained model. The
stub returns **fake** predictions cycling through a demo vocabulary, and the
service says so in its startup log. It is never a silent fallback.

## 2. Terminal

```bash
cd web
npm install
npm run setup      # vendors MediaPipe wasm + models, seeds the menu
cp .env.example .env.local
npm run dev
```

Open <http://localhost:3000> and allow camera access.

### The database needs no setup

Storage is [PGlite](https://pglite.dev) - real Postgres compiled to WASM,
running in-process and writing to `web/.pglite/`. No Docker, no
connection string. Migrations apply automatically on first query.

To start from a clean menu: delete `.pglite/` and run `npm run db:seed`.

## Choosing the LLM backend

| `LLM_BACKEND` | Model | Use for |
|---|---|---|
| `gateway` | hosted, via AI SDK gateway | the baseline, and any demo that must not fail |
| `ollama` | our QLoRA Qwen2.5-3B | the fine-tuned result, and offline demos |

Both must pass the same fixture suite. That comparison is the benchmark
chapter, so keep them interchangeable. Run it with
`npm run benchmark:agent` (see below).

## Checks

```bash
cd web && npm run check      # feature-spec drift, types, lint
cd web && npm test           # ordering fixture suite (no LLM needed)
cd services/ml && ./.venv/Scripts/python.exe -m pytest tests/ -q
```

## Agent comparison benchmark

Same ordering scenarios through three policies: **rules** (keyword floor),
**ollama** (local Qwen), **gateway** (hosted frontier). Scores cart match,
price integrity (no invented `$` amounts), and latency.

```bash
cd web
cp .env.example .env.local   # once; add AI_GATEWAY_API_KEY if using gateway
npm run benchmark:agent                              # all backends (skips missing ones)
npm run benchmark:agent -- --backends=rules          # floor only, always works
npm run benchmark:agent -- --backends=rules,ollama   # after: ollama pull qwen2.5:3b-instruct
```

This is the GenAI scoreboard: how far local/fine-tuned Qwen sits behind the
hosted baseline, and how far both sit above the non-learned rules.

`check:spec` compares `lib/mediapipe/featureSpec.ts` against
`ml/feature_spec.json`. Do not skip it: if those disagree, the model is fed a
different vector than it was trained on, nothing throws, and accuracy just
quietly drops.

## Troubleshooting

**"Sign recognition: offline"** - the ML service is not running. The terminal
retries with backoff and keeps working on speech and touch.

**Camera blocked** - Chrome only grants `getUserMedia` on `localhost` or
HTTPS. `http://192.168.x.x:3000` will not work.

**`column "..." does not exist` on every request** — you seeded or migrated
while `npm run dev` was running. PGlite is single-process: the dev server
opened the database with the old schema and cannot see the new one. Stop the
dev server, re-run the seed, restart. `npm run db:seed` now refuses to run
while port 3000 is listening, so this should not recur.

**No response from the assistant** - check `LLM_BACKEND` and its credentials.
The agent route logs `[agent] backend=... 412ms` for every turn.
