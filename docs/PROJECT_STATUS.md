# Project Status — Context Briefing

**Repository:** `jt-1010/visual-intelligence-platform` (GitHub)
**Course:** CMPE 295A (project) · CMPE 294 (technical writing)
**Advisor:** Professor Charan Bhaskar
**Team:** Jeremy Tung, Raymond Li, Sophia Atendido, Jack Liang
**Status as of:** 2026-10-05

> This document exists to give an AI note-taker (or a reader) complete context
> on what has been built, what the numbers actually are, and what is not done
> yet. Every figure below was measured, not estimated.

---

## 1. One-paragraph summary

We are building **an accessible self-service food ordering terminal that takes
orders in American Sign Language**. A webcam detects that a customer has
approached, the browser converts their signing into skeletal coordinates,
a trained neural network classifies those coordinates into ASL signs, and a
language model turns the recognized signs into a verified order — where prices
come from a database and can never be invented by the model. The system also
accepts speech, typing and touch, so it serves Deaf, hard-of-hearing,
speech-impaired, blind and non-disabled customers through the same interface.

**Current headline result: 88.6% signer-independent accuracy across a
64-sign ordering vocabulary, deployed and running live at ~4 ms per
prediction.**

---

## 2. The most important number to state correctly

**Say this:** "We downloaded Microsoft Research's ASL Citizen corpus — about
47 GB, 83,399 video clips — and from it extracted and trained on 2,580 clips
covering our ordering vocabulary."

**Do not say:** "We trained a model on 50 GB of data." That would be wrong and
is the kind of thing an advisor will probe.

| | |
|---|---|
| Corpus downloaded | ASL Citizen, **~47 GB**, 83,399 clips, 2,731 signs, 52 signers |
| Clips actually trained on | **2,580** (3.1% of the corpus) |
| Size of the training data after processing | **51 MB** |

The 47 GB is video. We run MediaPipe over it once to extract skeletal
landmarks, then **discard the video** — a 2.4-second clip becomes a
`32 × 153` array of numbers. That reduction is what makes the project
trainable on a single consumer GPU instead of a cluster.

---

## 3. What the system does, step by step

```
1. PRESENCE   Webcam → MediaPipe pose → "someone is at the counter"
2. GREET      System greets in speech AND captions AND on-screen menu
3. INPUT      Signing → landmarks → WebSocket → classifier → ASL glosses
              Speech → Web Speech API → text
              Typing / touch → text
4. UNDERSTAND All channels become one tagged message: [SIGN] [SPEECH] [TEXT] [TOUCH]
5. ACT        Agent calls tools: search_menu → add_to_cart → get_cart
              Prices are read from Postgres, never generated
6. RESPOND    Reply is spoken sentence-by-sentence AND captioned; cart updates
```

**Design principle worth stating to the advisor:** there is no "select your
disability" screen. The terminal offers every modality at once and adapts to
whichever the customer uses. Requiring someone to declare a disability to a
machine before it will serve them is both slower and the wrong thing to build.

---

## 4. The trained sign-recognition model

### Result progression (all signer-independent)

| Vocabulary | Clips | Accuracy | |
|---|---|---|---|
| 12 classes | 369 | 0.863 | first working model |
| 84 classes | 2,580 | 0.835 | full ordering vocabulary |
| **64 classes** | **2,580** | **0.886** | **deployed** — variants merged |

16 of 49 signers were **held out entirely** from training.

### Why 84 classes became 64

ASL Citizen labels `eat1`/`eat2` and `want1`/`want2` as separate classes. They
are the same sign produced differently. **30% of the 84-class model's errors
were confusions between such variants** — answers that were correct for our
purposes. Merging them shrank the problem to 64 classes, doubled the data for
merged classes, and gained roughly 5 percentage points.

### Architecture

Depthwise-separable Conv1D stem → 4-layer pre-norm Transformer encoder →
mean-pool → linear classifier. **1.28M parameters** — deliberately small,
because with ~30 clips per sign model capacity is the enemy.

Input is `32 frames × 153 values`: both hands (21 points each), plus nine
upper-body pose points, each as x/y/z.

### Deployment

Exported to ONNX, **verified numerically against PyTorch across batch sizes
1/2/8/64**, served by the FastAPI service. Replaying real held-out clips
through the live WebSocket returns **10/10 correct at ~4 ms**.

---

## 5. Why the split matters (likely advisor question)

**We split the data by signer, never randomly.**

A random split puts the same person in both training and test data. The model
then scores well by recognizing *that person* — their proportions, their
camera, their habits — rather than the sign. Published sign-language results
are frequently reported this way and are correspondingly inflated.

Our 88.6% is measured on **signers the model has never seen**. It is a lower
number than a random split would produce, and it is the only one that says
anything about a stranger walking up to the terminal.

For comparison: Microsoft's own ASL Citizen paper reports **63% top-1 across
2,731 signs**; the PopSign baseline reports **82.1% across 250 signs**. Ours is
88.6% across 64 — an easier problem than either, so the comparison is about
methodology, not superiority.

---

## 6. Known weaknesses — measured, documented, not tuned away

These are deliberately recorded rather than hidden. Raising them proactively is
stronger than being asked.

1. **Per-signer accuracy spread is 0.33** (range 0.67 to 1.00). The model works
   much better for some signers than others. This is a *fairness finding* in a
   system built for accessibility, not statistical noise.
2. **The confidence gate barely works.** It blocks only a minority of wrong
   predictions. The top1−top2 margin separates right from wrong better
   (+0.335 vs +0.264) and should replace it.
3. **Weakest signs:** `spicy→napkin`, `fry→bag`, `fish→cup`.
4. **Training is noisy enough that early-stopping patience reads as a result.**
   An early merged-vocabulary run scored 0.806 and looked like evidence
   *against* merging; it had simply stopped at epoch 70 versus 112. With
   patience raised it reached 0.886. Runs that stopped at different epochs are
   not comparable.
5. **Accuracy is not usability — demonstrated, on 2026-10-05.** The first live
   signing test produced `want want want want want eat eat eat pay hamburger
   hamburger`, and the phrase was never submitted. The classifier had read every
   sign correctly. Both faults were in segmentation: `accept()` treated the
   last-label check as a rate limit rather than a latch, so a held sign
   re-emitted every `debounce_frames`; and the client restarted its phrase timer
   on every repeat, so the pause that submits a phrase never arrived. Both are
   fixed and pinned by tests.

   Worth raising proactively: the existing test asserted
   `accepted <= (120 // debounce) + 1` — it *allowed* six emissions and passed
   the whole time the bug was live. A bound loose enough to admit the bug is not
   coverage. This is the clearest evidence in the project that 88.6% top-1 says
   nothing on its own about whether a person can place an order.

---

## 7. What is built (and tested)

| Component | Location | State |
|---|---|---|
| Ordering web app | `web/` | Next.js 16, TypeScript, Tailwind |
| Sign recognition service | `services/ml/` | FastAPI, ONNX Runtime, WebSocket |
| Training pipeline | `ml/asl/` | prepare, parallel extract, train, evaluate, export |
| Mamba architecture variant | `ml/asl_mamba/` | bidirectional Mamba encoder, same stem and I/O as `ml/asl` so the comparison changes one thing; trains and exports to ONNX, not yet evaluated against the Transformer |
| Agent benchmark harness | `web/lib/agent/benchmark/` | 10 scenarios, 3 backends |
| Menu database | PGlite (Postgres in WASM) | 71 items, zero setup |

**41 automated tests, all passing:**
- 30 ordering fixtures (`web/tests/`) — add, modify, remove, totals, price integrity
- 5 train/serve skew guards (`ml/asl/tests/`)
- 6 sign segmenter tests (`services/ml/tests/`)

### Accessibility engineering

WCAG 2.1 AA targets: ARIA live regions for captions and cart changes, full
keyboard navigation, 44px touch targets, `prefers-reduced-motion` respected,
high-contrast dark theme.

### Privacy property worth mentioning

**No camera imagery ever leaves the browser.** MediaPipe runs locally; only
153 numbers per frame are transmitted. A camera pointed at people ordering
food is exactly the kind of thing that should not be streamed to a server.

---

## 8. Engineering decisions worth defending

**Prices are never generated.** The menu lives in Postgres and is reached
through tool calls. A language model that generates prices will eventually
invent one, and a terminal that quotes a wrong price is worse than no terminal.
There is a test asserting every price traces back to a menu row.

**Training and serving share one preprocessing module.** `ml/asl/prepare.py`
imports normalization directly from the live service's code rather than
reimplementing it. A model trained on subtly different preprocessing than it is
served with degrades silently, with nothing to trace.

**Video extraction uses the same MediaPipe model files the browser loads.** A
different pose model would place landmarks differently, and the classifier
would be trained on a distribution it never sees in production.

**93% of the vocabulary needed no self-recording.** We checked our 62 ordering
concepts against ASL Citizen and Google's GISLR: 58 were already covered by
signers who use ASL natively. Only CHICKEN, NUGGET, FIVE and TEN are missing.
This matters methodologically — a model for Deaf users trained mostly on four
hearing students' signing would be a weakness a committee would rightly probe.

---

## 9. What is NOT done

State these plainly; they are the roadmap.

| Gap | Impact |
|---|---|
| **The local model cannot be trusted to order** | `qwen2.5:7b-instruct` replies fluently and calls tools unreliably — for the same request it has called `add_to_cart`, called `search_menu`, and called nothing at all while claiming *"Added two Burgers, $9.98"* with the order empty and no such price on the menu. **Mitigated, not solved:** plain orders are now resolved against the database before the model is consulted (`lib/agent/intent.ts`), so the common path is exact and never reaches it. Conversational turns still do, and anything the model says about prices on those turns is unverified. This is the gap the QLoRA fine-tune (M5) exists to close. |
| **QLoRA fine-tune not started** | The plan is to fine-tune Qwen2.5-3B on synthetic ordering dialogues and benchmark it against a hosted model. Harness is written; no model yet. |
| **Benchmark never run** | 10 scenarios across rules / Qwen / gateway backends exist in code but have produced no results, because of the item above. |
| **Full-corpus pretraining not done** | Estimated 33 hours single-threaded. Would likely lift accuracy above 88.6%. |
| **Recommender not trained** | Upsells currently use a hand-written rules baseline. |
| **4 signs still need recording** | CHICKEN, NUGGET, FIVE, TEN. |
| **Never tested by a Deaf signer** | The most important gap. All results are from replayed dataset clips. |

---

## 10. Suggested talking points

1. **Lead with the measured result**: 88.6% signer-independent over 64 signs,
   deployed and running at 4 ms.
2. **Be precise about the data**: 47 GB corpus downloaded, 2,580 clips trained
   on, 51 MB after landmark extraction.
3. **Emphasise the split methodology** — it is the most defensible thing in the
   project and distinguishes it from papers reporting inflated numbers.
4. **Raise the per-signer spread yourself** (0.67–1.00). Presenting a fairness
   weakness before being asked is a strength.
5. **Ask about compute** for the 33-hour pretraining run — the advisor has
   previously raised GPU access, and SJSU HPC would change what is feasible.
6. **Ask about Deaf user testing** — whether the university can connect the
   team with the ASL program or a Deaf student association. No amount of
   dataset accuracy substitutes for it.

## 11. Open questions for the advisor

- Is signer-independent evaluation the standard he expects reported, or does he
  want a random-split number alongside it for comparison?
- Is a 64-sign vocabulary sufficient scope for 295A, or should the target be
  the full 2,731-sign corpus?
- Does the department have GPU resources for the full-corpus pretraining run?
- Can he facilitate contact with native ASL signers for evaluation?
- Should the fine-tuned-vs-hosted LLM comparison be a formal chapter, or is
  the tool-constrained architecture sufficient on its own?

---

## Appendix: reference documents in this repository

| File | Contents |
|---|---|
| `docs/datasets.md` | Survey of six sign-language datasets with sizes, licences, verified against the real data |
| `docs/training.md` | Every model we train: datasets, preprocessing, architectures, hyperparameters, evaluation splits |
| `docs/development.md` | How to run the system locally |
| `docs/spec.md` | Original design specification |
| `data/menu/README.md` | Menu data provenance — which numbers are real and which are estimates |
| `meetings/` | Advisor meeting notes |

**Important caveat recorded in `data/menu/README.md`:** all menu prices are
*estimated*, not observed. No open dataset of fast-food prices exists. Item
names, calories and protein come from a published dataset; prices are generated
by a documented heuristic and tagged `price_source=estimated` in the database.
This must be stated as a limitation in any written report.
