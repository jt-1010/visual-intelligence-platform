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
{ "ok": true, "sign_model": "onnx", "recommender": "none", "feature_length": 153 }
```

`sign_model` tells you what is actually recognising signs:

| value | meaning |
|---|---|
| `onnx` | the trained classifier in `ml/asl/artifacts/` is loaded. Expected on a fresh clone — the export is committed so the service works without training first. |
| `none` | no model found. The service recognises **nothing** and says so on startup. It is silent rather than fake, on purpose. |
| `stub` | `SIGN_STUB=1` is set. Emits **fabricated** signs for pipeline debugging only. Never use it in a demo: it puts words in the mouth of the user this system exists to serve, and it can add items to a real order. |

Set `SIGN_ARTIFACT_DIR` to serve a different export (for example an
`ml/asl_mamba` experiment) without overwriting the committed model.

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
| `ollama` | whatever `OLLAMA_MODEL` names | offline demos |

Both must pass the same fixture suite. That comparison is the benchmark
chapter, so keep them interchangeable. Run it with
`npm run benchmark:agent` (see below).

**The fine-tuned model does not exist yet.** `ollama` currently points at a
stock instruct model, not our QLoRA adapter — that is milestone M5. A stock
quantised model answers conversationally but calls tools unreliably, so it
will happily say "Added two cheeseburgers" without ever calling `addToCart`,
and the order panel stays empty. If you are demonstrating *ordering* rather
than *signing*, use the `gateway` backend. See Troubleshooting.

## Checks

```bash
cd web && npm run check      # feature-spec drift, types, lint
cd web && npm test           # ordering + transcript suites (no LLM needed)
cd services/ml && ./.venv/Scripts/python.exe -m pytest tests/ -q   # segmenter
python -m pytest ml/asl_mamba/tests -q                             # Mamba variant
```

The segmenter suite is the one to watch. It pins how a continuous stream is cut
into discrete signs, which is where a working classifier most easily looks
broken — see "Segmentation" in `docs/spec.md`.

## Agent comparison benchmark

Same ordering scenarios through three policies: **rules** (keyword floor),
**ollama** (local Qwen), **gateway** (hosted frontier). Scores cart match,
price integrity (no invented `$` amounts), and latency.

```bash
cd web
cp .env.example .env.local   # once; add AI_GATEWAY_API_KEY if using gateway
npm run benchmark:agent                              # all backends (skips missing ones)
npm run benchmark:agent -- --backends=rules          # floor only, always works
npm run benchmark:agent -- --backends=rules,ollama   # after: ollama pull "$OLLAMA_MODEL"
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

**The assistant says it added something, but the order stays empty** - it
replied without calling the tool. The order panel reads the database, so it is
telling the truth and the sentence is not. Expect this from a stock quantised
model on `ollama`; switch to `gateway`, or treat it as the gap the QLoRA
fine-tune (M5) is meant to close. It is not a bug in the cart.

**The hand skeleton does not appear in the self-view** - it only draws once
MediaPipe has a hand in frame, so it is absent when your hands are down or out
of shot. If it never appears at all, check the console for a model-loading
error from `npm run setup`. The pose skeleton and the dashed shoulder
measurement are separate: those are calibration instruments and only appear
under `?tune=1`.
