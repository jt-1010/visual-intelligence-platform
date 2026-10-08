# SignOrder — Multimodal Accessible Ordering Terminal

An ordering terminal that takes a complete food order in **American Sign Language**, speech, or touch.
It notices when someone walks up, greets them in every modality at once, and adapts to whichever
channel the person actually uses.

CMPE 295A · San José State University · Advisor: Prof. Vidhyacharan Bhaskar

> **Note:** the repository is named `visual-intelligence-platform` for historical reasons — it once
> carried a second, traffic-related workstream. That has been dropped. The project is ASL accessible
> ordering only.

## Why landmarks, not video

The browser runs MediaPipe locally and streams only **hand/pose/lip landmark coordinates** to the
server — a few KB per frame instead of megabytes of video. Three consequences:

1. **Latency.** Round trip stays under ~100ms, which is what makes the terminal feel conversational.
2. **Privacy.** No camera imagery is ever transmitted or stored. Only coordinates leave the device.
3. **No train/serve skew.** Our training data (Google ISLR) *is* MediaPipe landmarks, so the model
   sees the same representation in training and in production.

## Prices are never generated

The menu lives in Postgres and is reached through tool calls. The language model handles conversation
and ASL gloss reordering (`WANT / BURGER / TWO` → a structured cart action) but never invents an item
or a price. A terminal that quotes a wrong price is worse than no terminal.

## Architecture

```
Browser (Chrome)                  Next.js /api/agent          FastAPI services/ml
  MediaPipe → landmarks  ──WS──────────────────────────────►  ASL classifier (ONNX)
  Face detect → presence                                       recommender
  Web Speech → STT/TTS   ──SSE──►  tool-calling agent  ──────►
                                          │
                                          ▼
                                   Postgres: menu · cart · orders
```

## Layout

| Path | What |
|---|---|
| `web` | Next.js + TypeScript + Tailwind UI and agent route |
| `services/ml` | FastAPI: sign recognition WebSocket, recommender |
| `ml/asl` | ASL classifier: data prep, training, eval, ONNX export |
| `ml/asl_mamba` | Same task with a Mamba encoder — the architecture comparison |
| `ml/recsys` | Order recommender training |
| `ml/llm` | Synthetic dialogues, QLoRA fine-tune, GGUF export |
| `ml/collect` | Tool for recording our own ordering signs |
| `docs` | Spec, ADRs, figures |
| `meetings` | Advisor and team notes |

## Getting started

See [docs/development.md](docs/development.md).

## The three models we train

1. **ASL sign classifier** — Transformer over landmark sequences. The centerpiece. A Mamba variant in
   `ml/asl_mamba` shares the stem and the input/output shapes, so the two can be compared directly.
2. **Order recommender** — next-item model over cart state; drives specials and upsells.
3. **Conversational LLM** — Qwen2.5-3B QLoRA fine-tuned on synthetic ordering dialogues, served
   locally via Ollama, benchmarked against a hosted baseline.

Classification accuracy is not the whole system. Cutting a continuous stream into discrete signs is a
separate problem, and the one most likely to make a working model look broken — see **Segmentation** in
[docs/spec.md](docs/spec.md).

## Documentation

| Doc | What |
|---|---|
| [docs/development.md](docs/development.md) | Running it locally |
| [docs/research.md](docs/research.md) | **How to make the models smarter, with citations — and why our 88.6% is not comparable to published results** |
| [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | **What is built, the real numbers, and what is not done** |
| [docs/datasets.md](docs/datasets.md) | **Which sign language datasets exist and which we use** |
| [docs/training.md](docs/training.md) | **What we train, on what data, with what settings** |
| [data/menu/README.md](data/menu/README.md) | Where the menu came from, and which numbers are estimates |
| [docs/spec.md](docs/spec.md) | Full design |
