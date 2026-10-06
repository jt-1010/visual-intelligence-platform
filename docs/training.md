# What we train, and on what data

Four models are ours (one optional). Two things people assume we train, we do not. This
document is the answer to "what are you actually training?" — put the specifics
in the report from here.

---

## Summary

| # | Model | Trained by us | Data | Compute |
|---|---|---|---|---|
| 1 | ASL sign classifier | **Yes, from scratch** | Google ISLR (~100k sequences) + ~2.4k we record | Colab T4, 2–4h |
| 2 | Order recommender | **Yes, from scratch** | Instacart (method) + synthetic order log (deployed) | CPU, minutes |
| 3 | Ordering LLM | **Yes, QLoRA adapter** | ~8k synthetic dialogues we generate | Colab T4, 3–5h |
| 4 | Food image classifier | **Yes, fine-tuned** | Kaggle Fast Food Classification V2 (20k images) | Colab T4, ~1h |
| — | MediaPipe pose/hands | No — pretrained, used as-is | — | — |
| — | The menu | **Never trained.** It is a database table | — | — |

The last row is the one that surprises people. See "What we deliberately do
not train" at the bottom.

---

## 1. ASL sign classifier — the centerpiece

### Data

**Primary: Google Isolated Sign Language Recognition (GISLR), Kaggle 2023.**

- ~100,000 sequences of isolated signs
- 250-sign vocabulary
- 21 Deaf signers (this matters — see the split below)
- Already MediaPipe Holistic landmarks, not video

```bash
kaggle competitions download -c asl-signs -p data/asl
```

The raw parquet is tens of GB. Preprocess once to a compact `.npy` of only the
landmarks in `ml/feature_spec.json` and train from that — the reduced array is
a couple of GB and fits Colab's disk and RAM comfortably.

That the dataset ships as landmarks rather than video is the reason this
project is feasible on a free T4 at all. Training on raw video would mean a 3D
CNN, tens of GPU-hours, and a serving path that cannot hit our latency budget.

**What the 250 signs actually cover — verified, not assumed.**

The vocabulary comes from PopSign, a game teaching parents of Deaf toddlers,
so mealtime words are core to it. Checked against the real label file
(`data/asl/sign_to_prediction_index_map.json`), **31 of the 250 signs are
directly useful for ordering**:

```
drink, milk, water, food, hungry, thirsty, apple, orange, icecream, frenchfries, pizza, cereal, chocolate, snack, nuts, carrot, fish, taste, finish, please, thankyou, yes, no, give, have, like, hot, all, many, first, wait
```

`frenchfries`, `pizza`, `icecream`, `milk`, `water`, `drink`, `hungry` and
`thirsty` are all in there. A large part of an ordering vocabulary comes free.

Notably **absent**, and more surprising than the presences: `eat`, `want`,
`more`, `cheese`, `bread`, `cup`, `cold`, `help`.

**What we must record ourselves: the counter vocabulary.**

No toddler-directed dataset contains these, and they are unavoidable at a
till:

```
BURGER  NUGGET  COMBO  MEAL  LARGE  MEDIUM  SMALL  COFFEE  SODA
CHEESE  SAUCE   WANT   MORE  EAT    CARD    CASH   HERE    TO-GO
ONE ... TEN
```

≈28 signs. We record them with `ml/collect`, using the **same MediaPipe
pipeline the browser uses**, so the new samples are byte-identical in format
to GISLR.

4 people × ~20 takes × ~28 signs ≈ **2,200 sequences**. This is also an
original dataset contribution for the report.

### The class-imbalance trap

GISLR gives ~100,000 samples across 250 classes; we add ~2,200 across ~28.
That is a **40:1 imbalance**, and a naively trained model learns to almost
never predict the new classes — guessing a GISLR sign is right far more often.
You would report 85% overall accuracy and have a system that cannot recognise
BURGER.

Two fixes, and we do both:

1. **Two-stage training** — pretrain on GISLR, then fine-tune on the union with
   most of the backbone frozen.
2. **Class-balanced sampling** — each batch sees our signs as often as theirs.

Report **per-class recall**, never overall accuracy alone. The average hides
exactly the failure that matters.

> **Ethics note for the report:** none of us is a native signer. Signs we
> record ourselves will carry non-native production errors. State this as a
> limitation, and if at all possible have a Deaf signer or ASL instructor
> review the recorded vocabulary. A model trained only on learner signing will
> underperform for the people it is built for.

### Preprocessing

Defined once in `ml/feature_spec.json` and read by both the trainer and the
inference service, so training and production cannot silently diverge.

- **Keep** both hands (21 × 2), lips (~40), upper-body pose (9) → **153 floats/frame**
- **Drop** the other ~450 face landmarks — they cost 3× the input width and add little
- **Resample** every clip to 32 frames, so speed of signing is not a feature
- **Normalize** per frame: centre on the shoulder midpoint, divide by shoulder width

### Split — the one that must not be got wrong

**Split by signer (`participant_id`), never at random.**

A random split puts the same person's signing in both train and test, and the
model scores well by recognising *them*, not the sign. A signer-independent
split is the only number that means anything, and it will be lower. Report
that one.

### Architecture

```
(32, 153) → Conv1D depthwise-separable stem (temporal)
          → Transformer encoder ×4  (d_model=256, 4 heads, FFN 512)
          → masked mean-pool
          → Linear → ~280 classes
```

≈5–8M parameters.

### Training

| | |
|---|---|
| Optimizer | AdamW, weight decay 0.01 |
| LR | 1e-3, cosine decay, 5-epoch warmup |
| Batch | 64 |
| Epochs | ~60, early stop on signer-independent val |
| Loss | Cross-entropy, label smoothing 0.1 |
| Augment | horizontal flip, time warp ±20%, random affine, frame dropout |

### Evaluation

- Top-1 and top-5, signer-independent. **Target ≥80% top-1.**
- Confusion matrix — expect confusion between visually similar signs
- Per-signer accuracy — a model that works for 18 signers and fails for 3 is a
  fairness finding worth reporting, not an average to hide
- **Latency**: p50/p95 for one window on CPU. Must stay under ~100ms.

### Export

ONNX → `ml/asl/artifacts/sign_classifier.onnx` + `labels.json`. The service
picks it up automatically. Until then sign input is INACTIVE and says so -- it
does not fake predictions. Set `SIGN_STUB=1` only for pipeline debugging.

Set `SIGN_ARTIFACT_DIR` to serve a different export — an `ml/asl_mamba` run,
say — without overwriting the committed model.

---

## 2. Order recommender

### Data — read this honestly

**There is no public dataset of fast-food transactions.** So this is a
two-part approach, and the report must say so:

1. **Method validation on real data.** Instacart Online Grocery (3.2M orders,
   50k products) — real baskets, real sequences. We show the architecture and
   metrics work on real transaction data.
2. **Deployment on synthetic data.** A generated order log over our own menu,
   built from plausible co-occurrence priors (burger→fries→drink).

Do not present (2) as an empirical result. It makes the terminal behave
sensibly; it does not demonstrate anything about real customers.

### Models

- **Baseline**: item-to-item co-occurrence. Kept permanently — a baseline you
  can explain in one sentence is what makes an improvement mean something.
- **Ours**: GRU4Rec-style next-item model over cart-state sequences.

### Evaluation

Recall@5 and MRR against the baseline, on a **temporal** split (train on
earlier orders, test on later) — not random, or you leak the future.

---

## 3. Ordering LLM — QLoRA fine-tune

### Base model

**Qwen2.5-3B-Instruct** — Apache 2.0, which is the cleanest license for a
published thesis. Llama 3.2 3B is the fallback but carries a custom license.

We do **not** pretrain a language model. Pretraining costs millions of
GPU-hours; nobody at master's level does it, and claiming otherwise in a
report is a red flag. We fit a LoRA adapter (~20M new parameters) on our own
data, which is genuinely training a model — just adaptation, not pretraining.
Say it that way.

### Data — we generate it

~8,000 synthetic ordering dialogues, produced by `ml/llm/synthesize.py` from:

- the real menu in the database (so items and prices are always valid)
- scenario templates: order, modify, remove, ambiguous item, invalid item,
  change of mind, upsell accepted, upsell declined, confirm
- **ASL gloss inputs**, not just English — this is the whole point. The model
  must learn that `WANT I BURGER TWO NO ONION` means
  `add_to_cart(burger, 2, [no onion])`
- transcription noise injected into `[SPEECH]` turns, so it learns to cope
  with "aisle have a cheese burger"

Each example is `(tagged user input) → (correct tool call sequence)`. Held-out
dialogues become the eval set.

### QLoRA settings (fit a 16GB T4)

| | |
|---|---|
| Quantization | 4-bit NF4, double quant, bf16 compute |
| LoRA | r=16, α=32, dropout 0.05 |
| Target modules | q,k,v,o,gate,up,down projections |
| Sequence length | 1024 |
| Batch | 1 × grad accum 16 |
| LR | 2e-4, cosine |
| Epochs | 2–3 |

### Evaluation

The **same 30-scenario fixture suite** in `web/tests/ordering.test.ts`,
replayed through each backend:

| | Fine-tuned Qwen2.5-3B | Hosted baseline |
|---|---|---|
| Tool-call accuracy | | |
| Cart-state correctness | | |
| Price errors (must be 0) | | |
| p50 / p95 latency | | |

That table is the benchmark chapter. It only exists because the backends are
interchangeable via one env var.

### Serving

Free Colab **cannot** serve a live demo — sessions time out and you cannot bet
a presentation on one. So:

```
train on Colab → merge adapter → GGUF Q4_K_M (~2GB) → Ollama on the laptop
```

Genuinely self-contained, no API, no network.

---

## 4. Food image classifier — optional, and narrower than it looks

**Dataset:** [Fast Food Classification V2](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset)
— ~20,000 images, 10 classes: Baked Potato, Burger, Crispy Chicken, Donut,
Fries, Hot Dog, Pizza, Sandwich, Taco, Taquito.

```bash
kaggle datasets download -d utkarshsaxenadn/fast-food-classification-dataset   -p data/food-images --unzip
node scripts/import-food-images.mjs
npm --prefix web run db:seed -- --force
```

### Be precise about what it does

This model maps **image → category**. Three consequences worth stating before
anyone builds on it:

1. **It cannot classify a customer's request.** Requests arrive as ASL glosses
   or typed text. There is no image in that path, so an image classifier is
   not part of understanding what someone asked for. That job belongs to the
   sign classifier (#1) and the LLM (#3).
2. **It cannot tell a Big Mac from a Quarter Pounder.** Both are `Burger`. So
   photos attach per category, not per item.
3. **Its classes only partly overlap our menu.** Hot Dog, Pizza, Taco and
   Taquito are not on a McDonald's menu; our drinks, salads and shakes have no
   class here. Expect roughly half the menu to get a photo from this source
   and to photograph the rest yourselves.

### Where it genuinely earns its place

- **Menu photos.** A picture is the one description needing no shared
  language — it works for someone who is Deaf, reads little, or does not speak
  English. This is the real win, and it needs no training at all: just use the
  images.
- **Auto-tagging new photos.** Once trained, the classifier assigns a category
  to any new food photo the team adds, and flags mismatches (a photo filed
  under Fries that the model reads as Donut). That is a real, checkable job.
- **A second trained model for the report**, with a clean confusion matrix and
  an easy transfer-learning story.

### Recipe

Transfer learning, not from scratch: fine-tune EfficientNet-B0 or ResNet-50
pretrained on ImageNet. Freeze the backbone, train the head 5 epochs, then
unfreeze the top block at a 10× lower LR. Augment with flips, ±15° rotation
and colour jitter. Expect >90% top-1 — these classes are visually distinct,
which is also why this is a modest contribution rather than a headline one.

Report top-1, the confusion matrix, and per-class recall. Split by image, but
check the dataset for near-duplicate frames first: if the same burger appears
in train and test, the number is inflated.

### Suggesting the closest thing we do have

Worth separating clearly, because it sounds like the same feature and is not.
"We do not sell pizza — the closest we have is a Crispy Chicken Sandwich" is a
**text similarity** problem over menu names, not an image problem. It is
already implemented in `suggestAlternatives()` in `web/lib/agent/cart.ts`
using word overlap plus character trigrams, with four tests covering it, and
it needs no model. If it later proves too crude, the upgrade is sentence
embeddings over item names — still not images.

## Results so far (2026-09-17)

Trained on ASL Citizen landmarks, **signer-independent** split throughout —
16 of 49 signers held out entirely, never seen in training.

| Run | Classes | Clips | top-1 | top-5 |
|---|---|---|---|---|
| First model | 12 | 369 | 0.863 | — |
| Full vocabulary | 84 | 2,580 | 0.835 | 0.932 |
| **Variants merged** | **64** | **2,580** | **0.886** | **0.951** |

Random guess on 64 classes is 0.016, so 0.886 is ~55x chance.

For context: PopSign's LSTM baseline reports 0.821 over 250 signs, and ASL
Citizen's own paper 0.63 over 2,731. Ours is an easier problem than either —
64 classes — so do not read it as beating them.

### Merging sign variants was worth ~5 points

ASL Citizen labels regional and stylistic variants separately: `eat1`/`eat2`,
`want1`/`want2`. They are the same sign produced differently, and the terminal
does the same thing either way — so the 84-class model was being marked wrong
for answers that were, for our purposes, right. **30% of its errors were
variant confusions.**

Merging helps three ways at once: it removes a distinction we never wanted, it
doubles the data for merged classes (eat1 + eat2 = 63 clips, not two classes
of 31), and it shrinks the problem from 84 classes to 64.

> **A measurement trap worth recording.** The first merged run scored 0.806 —
> *worse* than the unmerged 0.835 — and looked like evidence against merging.
> It had early-stopped at epoch 70 against the unmerged run's 112. With
> patience raised from 15 to 30 it reached 0.886. This training is noisy
> enough that an aggressive patience reads as a result. Do not compare runs
> that stopped at different epochs.

### Known weaknesses

**Per-signer spread is 0.33** — from 0.67 (P28) to 1.00. The model works
materially better for some people than others. Report this as a fairness
finding; it is exactly the failure mode that matters for the users this is
built for, and averaging hides it.

**The confidence gate barely works.** Softmax separates right from wrong by
only +0.264, so the segmenter's 0.6 threshold passes most predictions and
blocks a minority of errors. The top1−top2 margin separates better (+0.335)
and should replace it.

**Weakest signs**: `spicy`→napkin (0.50), `fry`→bag (0.57), `fish`→cup (0.62).
23 of 64 classes are at 100% recall.

### Extraction cost, measured

| | |
|---|---|
| Single process | 0.41 clips/s |
| 14 workers | 0.70 clips/s — only **1.7x** |

MediaPipe runs TFLite with an XNNPACK delegate that spawns a thread pool per
process, so 14 processes on 16 cores mostly context-switch. Workers now pin
themselves to one thread before the native libraries load; that fix is **not
yet measured**. At 0.70 clips/s the full 2,731-sign corpus is ~33 hours, so
pretraining on all of it is only worth it if the thread fix lands.

## What we deliberately do not train

**MediaPipe pose and hand landmarkers.** Pretrained, used as-is. Re-training
them would be a project on its own and would be worse. This is the correct
call, not a shortcut.

**The menu.** It lives in Postgres and is reached by tool calls. A language
model that generates prices will eventually invent one, and a terminal that
quotes a wrong price is worse than no terminal. See `data/menu/README.md`.

**Speech recognition.** The browser's Web Speech API for now. If accuracy
proves inadequate, swap in `faster-whisper` in `services/ml` — still no
training, just a better pretrained model.

---

## Order of work

1. **ASL classifier on GISLR alone.** Biggest risk, longest lead time, and the
   thesis centerpiece. Start it first.
2. **Recording tool + our 30 signs.** Needs all four of us, so schedule early.
3. **Recommender.** Small and self-contained; good parallel task.
4. **LLM fine-tune.** Depends on the menu and the fixture suite, both of which
   already exist.

Sequencing note: (1) and (3) are independent and can run in parallel. (4)
should wait until (1) works, because debugging gloss→intent while the glosses
themselves are unreliable means debugging two unknowns at once.
