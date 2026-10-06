# SignOrder — Multimodal Accessible Ordering Terminal (CMPE 295A)

## Context

**The problem.** Self-service ordering terminals and drive-thrus are effectively closed to people who
communicate in ASL, and voice-only ordering systems are closed to people who are Deaf or
speech-impaired. Existing translation apps support a narrow set of input types and none of them
handle a transactional task like ordering, where the system must be *correct about prices*, not just
fluent. This is the "Assistive Tech (ASL Translation)" scenario the advisor proposed on June 3rd.

**What we're building.** A browser-based terminal that notices a person is present, greets them in every
modality at once, and takes a complete food order through ASL, speech, or touch — conversationally,
with sub-second response, adapting to whichever channel the person actually uses.

**Scope.** Accessible ASL ordering, end to end. Nothing else. The whole repo is this project.

**Repo.** `git@github.com:jt-1010/multimodal-accessibility-traffic-ai.git` — the name is a leftover and
should be renamed to match the project (e.g. `signorder`), which needs admin rights on Jeremy's
account. Not a blocker; use it as-is until then.

**Verified environment.** SSH to GitHub already authenticates as `rl4658`; `git ls-remote` on the repo
returns a `main` branch, so read access is confirmed. Push access is unverified — confirm Jeremy has
added you as a collaborator before Milestone 0. Node v24.14.0, Python 3.14, and Git 2.45.1 are installed.

---

## Design decisions (from brainstorming)

| Decision | Choice |
|---|---|
| ASL recognition | 250-sign landmark model (Google ISLR) + ~30 self-recorded ordering signs |
| Models we train | ASL classifier, order recommender, **and** a QLoRA fine-tuned local LLM |
| Frontend | Next.js (App Router) + TypeScript + Tailwind |
| ML backend | Python FastAPI |
| Demo target | Chrome on a laptop webcam |
| Training compute | Free Colab / Kaggle T4 |

---

## Architecture

```
┌─ Browser (Chrome terminal) ────────────────────────────────┐
│  MediaPipe Tasks (WASM)                                  │
│    • Face Detector    → presence trigger                 │
│    • Hand + Pose      → 543 landmarks @ 30fps            │
│  Web Speech API       → STT in / TTS out                 │
│  Raw video NEVER leaves the browser                      │
└──────┬───────────────────────────────┬───────────────────┘
       │ WebSocket (landmarks, ~4KB/f) │ HTTP/SSE
       ▼                               ▼
┌─ FastAPI  services/ml ──┐   ┌─ Next.js  web ──────┐
│  /ws/sign  → ASL model  │   │  /api/agent  (AI SDK v6)    │
│  /recommend → recsys    │   │  tools: search_menu,        │
│  (ONNX Runtime)         │◄──┤   add_to_cart, get_cart,    │
└─────────────────────────┘   │   recommend, confirm_order  │
                              └──────────┬──────────────────┘
                                         ▼
                        Postgres (menu · cart · orders)
                        LLM: Qwen2.5-3B QLoRA via Ollama
                             ‖ hosted API (fallback)
```

### Why landmarks instead of video

Streaming webcam frames to a server is the reason most sign-language demos feel sluggish. Sending
only MediaPipe landmarks — 543 (x,y,z) points, a few KB per frame — cuts bandwidth by ~1000× and
keeps the round trip under 100ms. It also gives the project a real privacy property worth stating in
the thesis: **no camera imagery is ever transmitted or stored.** And it means the training data
(Google ISLR ships as pre-extracted landmarks) is in exactly the same format as runtime input, so
there is no train/serve skew.

### Why the LLM is not trained on the menu

The menu lives in Postgres and is reached through tool calls. A language model that generates prices
will eventually invent one, and a terminal that quotes a wrong price is worse than no terminal. The LLM
handles conversation, disambiguation, and — importantly — **ASL gloss reordering**: recognized signs
arrive as `WANT / BURGER / TWO / NO / ONION`, which is not English word order. Mapping that to
`{action: add, item: burger, qty: 2, mods: [-onion]}` is exactly what an LLM is good at, and it is a
genuine, defensible use of generative AI rather than decoration.

---

## The three models we train

### 1. ASL sign classifier — the centerpiece

- **Data:** Google Isolated Sign Language Recognition (~100k landmark sequences, 250 signs, 21 Deaf
  signers) + ~30 self-recorded ordering signs (BURGER, FRIES, COMBO, LARGE, SMALL, COMBO, CARD, CASH…).
  Record with the *same* MediaPipe pipeline the terminal uses — 4 team members × ~20 takes × 30 signs
  ≈ 2,400 new sequences, which also gives the project an original dataset contribution.
- **Preprocessing:** keep both hands (42 pts), lips (~40 pts), upper-body pose (~10 pts) — drop the
  other ~450 face landmarks. Resample to T=32 frames. Normalize per frame: center on chest, scale by
  shoulder width.
- **Architecture:** depthwise-separable Conv1D stem → 4-layer Transformer encoder (d=256, 4 heads) →
  masked mean-pool → linear classifier over ~280 classes. ~5–8M params, trains in 2–4h on a free T4.
- **Augmentation:** horizontal flip, time warp, random affine, frame dropout.
- **Target:** ≥80% top-1 on held-out signers (signer-independent split — do not split randomly, or
  you will report an inflated number).
- **Export:** ONNX, served via ONNX Runtime in FastAPI.

### Segmentation

**The hard part is segmentation, not classification.** The classifier labels a clip; a real user signs
a continuous stream. The pipeline is a sliding window over the last 32 frames, a motion-energy gate on
landmark velocity to skip idle hands, a confidence threshold, and a rule for not emitting one sign
three times. Budget real time for this — it is the most likely thing to make a working model feel broken.

That prediction was borne out on 2026-10-05. The classifier read every sign in the demo correctly and
the screen still showed `want want want want want eat eat eat pay hamburger hamburger`, with the phrase
never sent at all. Two causes, both downstream of the model:

- `SignSegmenter.accept()` treated `last_label` as a **rate limit**. Emitting reset `frames_since_emit`,
  the counter the debounce compared against, so a sign held longer than `debounce_frames` was emitted
  again every `debounce_frames`. Hold a sign for four seconds, get five of it.
- `useGlossBuffer` restarted the 1800 ms phrase timer on every repeat, so while repeats kept arriving
  the gap was never reached and the phrase was never sent.

The rule now: **`last_label` is a latch on the sign in progress, not a rate limit.** A sign is emitted
once when it starts, and can only be emitted again after a boundary — a *different* sign, or sustained
rest. `push()` wants several consecutive quiet windows before calling it rest, because most signs have a
hold partway through where the wrists barely move, and that hold is not a boundary. On the client, a
repeat of the sign currently being held returns before the phrase timer is touched.

The lesson worth keeping: the test covering this asserted `accepted <= (120 // debounce) + 1`, which
permitted six emissions, and passed throughout. A bound loose enough to admit the bug is not coverage.
It now asserts exactly one.

### Ordering is resolved before the model sees it

The menu lives in Postgres and is reached through tool calls, so the model can never
invent a price. That protects the *database*. It does not protect the *sentence*, and on
2026-10-05 the difference became concrete. Asked for "two burgers", `qwen2.5:7b-instruct`
replied:

> "Added two Burgers, $9.98. Which burger would you like? We have Big Mac and Quarter Pounder."

It called no tool at all. Nothing was added, no such price exists, and it named two of the
thirteen burgers from memory. On a second run it called `search_menu`; on a third,
`add_to_cart`. Prompting changed which of these happened, not whether it happened.

So the ordinary path no longer asks. `lib/agent/intent.ts` resolves the message against the
menu first:

| Input | Resolution | Who answers |
|---|---|---|
| "two burgers" | 13 rows match equally | **Choice**, from the database |
| "Big Mac" | one row | **Add**, priced from that row |
| "lobster thermidor" | nothing | model, which offers alternatives |
| "remove the fries", "what do you have?" | not a plain add | model |

"Two burgers" is not a language problem; it is a lookup against a 71-row table. Doing it in
ordinary code makes the common case exact, instant and identical every time, and leaves the
model the part it is genuinely good at — conversation that is not a lookup. The bar for
taking over is deliberately high: anything resembling a question, a correction or a removal
goes to the model, because being wrong here is worse than being slow.

### Choosing is the interaction, not an error path

The classifier knows 64 signs. A signer can produce BURGER; they cannot produce "Double
Quarter Pounder with Cheese". Thirteen of the menu's burgers are unreachable by signing
alone, and the same holds for chicken (18 rows), fries and drinks (sizes). Disambiguation is
therefore not a repair for bad recognition — for the sign path it is the **normal shape of
every order**, and it is why the menu is now permanently on screen rather than behind a
toggle.

The options are rendered as buttons from the tool result (`components/ChoicePrompt.tsx`),
never from the reply text. If the model narrates badly, the choice on screen is still correct
and still works. Answering by pressing is also faster than signing, which matters when the
person has just been told their signing was ambiguous.

### Mamba variant

`ml/asl_mamba/` is a second classifier that swaps the Transformer encoder for a bidirectional Mamba
(pure PyTorch; `mamba-ssm` does not build on Windows). It reuses the Transformer's `ConvStem` and keeps
the same input and output shapes, so the comparison changes one thing. Serve it without disturbing the
committed model by pointing `SIGN_ARTIFACT_DIR` at its export.

### 2. Order recommender — produces the "specials" and upsells

- **Model:** item-to-item co-occurrence baseline, then a GRU4Rec-style next-item model over cart state.
  Metrics: Recall@k and MRR against the baseline.
- **Data honesty (read this):** there is no large public *fast-food transaction* dataset. The plan is to
  validate the modeling approach on a real public dataset (Instacart / Retail Transactions), then train
  the deployed recommender on a synthetic fast-food order log generated from realistic co-occurrence
  priors over our actual menu. This is defensible **only if stated plainly** in the report — write it
  as a limitation, don't paper over it.
- **Menu seed:** Kaggle "Restaurant Menu Items" (~5k real DoorDash items with real prices) to seed a
  plausible ~60-item menu.

### 3. QLoRA fine-tuned conversational LLM

- **Base:** Qwen2.5-3B-Instruct (Apache 2.0 — cleanest license for a published thesis; Llama 3.2 3B is
  the alternative). 4-bit QLoRA, seq len 1024, batch 1 + grad accum fits comfortably in a 16GB T4.
- **Training data:** synthetic ordering dialogues we generate — gloss-sequence inputs paired with
  correct tool-call outputs, covering ordering, modification, removal, disambiguation, and confirmation.
- **Serving — the constraint that matters:** you **cannot** serve a model from free Colab during a live
  demo; sessions time out and you cannot bet a presentation on one. Instead: train on Colab → merge the
  adapter → export GGUF → run locally under **Ollama**. A 3B model at Q4_K_M is ~2GB and runs at usable
  speed on a laptop, so the demo is genuinely self-contained with no external API.
- **Design for it:** put the LLM behind a provider interface with two backends (local Ollama, hosted
  API). This keeps the demo safe if the local model underperforms, and gives the thesis a real
  comparison chapter: fine-tuned-3B vs. hosted-frontier on tool-call accuracy and latency.

---

## Accessibility design

**Principle: no "select your disability" screen.** When a person is detected, the terminal greets them in
*every* modality simultaneously — speaks the greeting, captions it in large type, and shows the menu —
then adapts to whichever channel the person actually uses. Making someone declare a disability to a
machine before it will serve them is slow, and it is the wrong thing to build.

| User | Input | Output |
|---|---|---|
| Deaf / hard of hearing | ASL, touch | Large captions, visual cart |
| Blind / low vision | Speech, keyboard | TTS, ARIA live regions, screen-reader flow |
| Speech-impaired | ASL, touch/text | TTS + text |
| Motor impairment | Speech | Speech + text, large targets |
| No disability | Speech or touch | Speech + text |

**Presence trigger:** MediaPipe Face Detector in-browser. Fire the greeting when a face is present for
>1.5s and the bounding box exceeds a size threshold (proxy for "at the counter, not walking past").
Reset after 5s of absence. Debounce so one person isn't greeted repeatedly.

**Hand landmarks are always drawn in the self-view.** Not as a debugging
instrument — as feedback. The hands are what the classifier reads, so joints
landing on your fingers is the one honest signal that the terminal is tracking
you, available *before* you commit to a phrase and discover it read nothing.
Signing into a camera with no indication it sees your hands is the visual
equivalent of speaking into a microphone with no level meter. Left and right
are coloured differently so a signer can see which hand dropped out of frame.

The pose skeleton and the dashed shoulder-width measurement are a different
thing: those are calibration instruments for whoever is setting the camera up,
and they stay behind `?tune=1`. A customer does not need to know the presence
gate is thresholding on shoulder width.

**Compliance target:** WCAG 2.1 AA — contrast, 44px touch targets, full keyboard nav, ARIA live regions
for every state change. Cite the DOJ self-service terminal accessibility rules for framing in the report.

---

## Repository layout

```
web/            Next.js + TS + Tailwind + AI SDK
  app/asl/             the ordering terminal (?tune=1 adds calibration)
  app/api/agent/       streaming agent route, tool definitions
  components/          camera, captions, cart, menu
  lib/interaction/     gloss buffering, speech, transcript
  lib/mediapipe/       landmark capture + presence detection
services/ml/           FastAPI: /ws/sign, /recommend, health
  app/segmenter.py     continuous stream -> discrete signs
ml/
  asl/                 data prep · model · train · eval · export_onnx
  asl_mamba/           same task, Mamba encoder — the architecture comparison
  recsys/              data prep · train · eval
  llm/                 dialogue synthesis · qlora_train · merge · export_gguf
  collect/             self-recording tool for the 30 custom signs
notebooks/             Colab training notebooks
docs/                  spec, ADRs, thesis figures
meetings/              advisor and team notes
data/                  gitignored; download scripts only
```

Work on a feature branch off `main`, PR in. Do not commit datasets or weights — download/export scripts only.

---

## Milestones

**M0 — Walking skeleton (do this first, before training anything).**
Browser captures landmarks → WebSocket → FastAPI returns a *hardcoded* sign → agent route → tool call →
TTS speaks it. Proves the full loop and its latency budget while the models are still stubs. If this
loop is slow, no amount of model accuracy will save the demo.

**M1 — Menu + agent.** Postgres schema, seed from the Kaggle menu dataset, tool-calling agent on a
hosted LLM, working touch + speech ordering. This is already a demoable product.

**M2 — ASL model.** Data prep, train on Google ISLR, signer-independent eval, ONNX export, wire into
`/ws/sign`. Then sliding-window segmentation and debouncing.

**M3 — Custom signs.** Build the recording tool, collect ~2,400 sequences across the team, retrain on
the union, re-evaluate.

**M4 — Recommender.** Train, evaluate against baseline, expose as the `get_recommendations` tool
driving the greeting's specials.

**M5 — QLoRA LLM.** Synthesize dialogues, fine-tune, merge, GGUF export, serve via Ollama, benchmark
against the hosted baseline on tool-call accuracy and latency.

**M6 — Accessibility audit + polish.** WCAG pass, keyboard nav, screen-reader run-through, latency
tuning, demo script.

---

## Verification

- **Latency budget, measured end to end** (instrument each hop): landmark capture → sign prediction
  <100ms; agent first token <500ms; full spoken response <1.5s. Log per-hop timings to the console
  from M0 onward so regressions are visible immediately.
- **ASL model:** signer-independent held-out split, top-1 and top-5 accuracy, confusion matrix. Confirm
  the split is by *signer*, not random — verify by asserting no signer ID appears in both splits.
- **Recommender:** Recall@5 and MRR vs. the co-occurrence baseline on a temporal split.
- **Agent correctness:** a fixture suite of ~40 ordering scenarios (add, modify, remove, ambiguous item,
  invalid item, mid-order change of mind) asserting the resulting cart JSON. Run against both LLM
  backends — this is also the fine-tuned-vs-hosted benchmark.
- **Price integrity test:** assert every price the agent utters matches the database. This is the one
  test that must never fail.
- **Accessibility:** axe-core in CI, plus a manual keyboard-only and screen-reader (NVDA) pass.
- **Live smoke test:** walk into frame, get greeted, order a combo in ASL, modify it, confirm — on a
  laptop webcam, end to end.

---

## Open items for the team

1. Confirm push access for `rl4658` on the repo.
2. Repo rename — needs Jeremy (admin).
3. **Role clarification:** the current presentation assigns people to two different workstreams. Confirm
   who owns which milestone here so two people aren't building the same thing.
4. Ask the advisor about SJSU HPC access — he raised GPU usage specifically, and a T4 caps the LoRA work.
5. The abstract and slides describe a second workstream. If the project is now ASL-only, those documents
   need updating before the next advisor review.
