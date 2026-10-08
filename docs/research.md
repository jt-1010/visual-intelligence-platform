# Research notes — making the models smarter

Written 2026-10-08. Everything here is either measured in this repo or cited to a
paper. Where a number is ours and a number is theirs, both are labelled, because
the most useful finding below is that one of our numbers has been read wrongly.

---

## 1. Our 88.6% is not comparable to published ASL Citizen results

This matters before any training decision, because it changes what "good" means.

| | Vocabulary | Top-1 | Source |
|---|---|---|---|
| **Ours** | **64 signs** | **88.6%** | our signer-independent split |
| ASL Citizen baseline (I3D, appearance) | 2,731 signs | ~63% | [Desai et al., NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf) |
| ASL Citizen baseline (ST-GCN, pose) | 2,731 signs | a few points below I3D | same |
| Prior SOTA before that dataset | 2,000+ signs | ~30% | same |

Ours is the easier problem by a wide margin — 64 classes against 2,731 — so
**88.6% is not evidence that we beat a published baseline, and the report must
not imply it.** The honest framing is that we hit 88.6% on a deliberately small
ordering vocabulary chosen because it is what a customer needs, and that the
comparable published figure for the full dictionary task is ~63%.

Two further cautions from the literature:

- Many ASL Citizen results are **Recall@k for dictionary retrieval**, not plain
  classification accuracy. Check the metric before quoting any number.
- Vocabulary size, signer splits and input type (landmarks vs RGB) all differ
  between papers, so headline numbers are rarely directly comparable.

---

## 2. The highest-value training change we can make

**Pretrain on the full corpus, then fine-tune the 64-sign head.** The evidence
for this is consistent and the effect sizes are large:

- Pretraining an I3D on MS-ASL and fine-tuning on SIGNUM subsets raised accuracy
  **by up to 21%** over training from scratch
  ([Transfer Learning for ASL](https://homes.cs.washington.edu/~ali/papers/Transfer_Learning_ASL.pdf)).
- [Logos pre-training](https://arxiv.org/pdf/2505.10481) improves accuracy
  substantially over training from scratch on AUTSL and WLASL treated as
  low-resource, and reports a result that maps directly onto our situation:
  with the **encoder frozen** and only a classification head trained, AUTSL
  (small vocabulary) held **90.16% at 10 samples per class, 83.99% at 3, and
  82.44% at 1**, against 95.25% with the full dataset. WLASL (large vocabulary)
  collapsed under the same treatment — 61.12% / 54.10% / 37.07%. The authors
  attribute the gap to vocabulary size.

  **We are the AUTSL case, not the WLASL case.** 64 signs is a small vocabulary,
  which is precisely the regime where a frozen pretrained encoder holds up.
- Models trained without pretraining "all converged around the most frequent
  glosses", which is a plausible description of our own per-signer spread
  ([comprehensive study](https://arxiv.org/pdf/2007.12530)).

This is not a new idea for this project — `docs/PROJECT_STATUS.md` already lists
"full-corpus pretraining not done, estimated 33 hours" as a gap. The research
says that gap **is** the lever, and quantifies roughly what it is worth.

### What we actually have

| | |
|---|---|
| Clips downloaded | 83,399 (ASL Citizen) |
| Clips trained on | 2,580 — **3.1%** |
| Signs | 64 |
| Per-signer accuracy spread | 0.67 – 1.00 |

That spread is a **fairness finding in an accessibility system**, not noise: the
model works much better for some signers than others. More pretraining data is
the standard remedy, and it is measurable on the existing signer-independent
split.

### Order of work, cheapest first

1. **Pretrain the existing encoder on the full 83k corpus**, then fine-tune the
   64-class head. Measure against the current signer-independent split so the
   comparison is honest. Report per-signer accuracy, not just the mean.
2. **Try the frozen-encoder variant** as well as full fine-tuning. Logos found
   freezing competitive at small vocabularies and much cheaper to train.
3. **Then** consider self-supervised pretraining. [SHuBERT](https://arxiv.org/html/2411.16765v1)
   reports surpassing specialised models on ASL Citizen by ~5% using multi-stream
   self-supervised pretraining over unlabelled signing video.
4. Only after the above, consider architecture. `ml/asl_mamba/` exists for this
   comparison and should be evaluated against the Transformer **on the same
   split**, changing one variable.

Augmentation is worth testing but the evidence was thinner. A 2025 preprint
reports synthetic-data pretraining outperforming traditional augmentation and
being complementary to it; I found no solid comparison of specific classic
augmentations (time warp, flip, keypoint jitter) in the sources surfaced.

### What this will NOT fix

Segmentation. "Spicy four spicy" was a transition artefact and is handled in
`services/ml/app/segmenter.py` by requiring consecutive windows to agree — no
amount of training data addresses it, because the classifier was never wrong
about the sign, only about the hand travelling into it. Keep the two problems
separate when reporting.

---

## 3. Traffic: the data question, and a scope question first

**Scope.** This workstream was dropped. The README, the abstract and the
2026-10-01 advisor notes all describe an ASL-only project, and `docs/spec.md`
says the repo name is a leftover. Reviving it is a real decision with a real
cost — it is a second vision model, a second dataset problem and a second
evaluation story, in a semester that has not finished the first. **Decide that
before any of the below gets built.**

If it is revived, the data question has a clear answer and it is not Google Maps.

### Why not Google Maps

The Maps Platform licence does not permit using its traffic data to train a
model, and there is no endpoint that returns per-intersection vehicle counts.
It gives you a rendered picture of congestion, not the measurements underneath
it. It is a plausible *display* layer; it is not a training set.

### Three options that do work

| Option | What it gives | Cost |
|---|---|---|
| **SUMO** (Eclipse, open source) | A real microscopic traffic simulator: per-vehicle movement, per-intersection queues, configurable signal plans, and a Python API (TraCI) to drive it step by step | Learning curve, but it is the standard tool and it is free |
| **Public sensor datasets** (PeMS, UTD19) | Real loop-detector counts and speeds from real intersections | No video, no visual component |
| **Hand-built graph simulation** | Fastest to demo, full control | Not research — you would be training a model on your own assumptions |

Your instinct about a graph with red and yellow edges is the third one. It is
the fastest to show and the weakest to defend, because a model trained on a
simulation you wrote can only rediscover the rules you put in. **SUMO is the
same idea done defensibly**: it still gives you a graph you can colour by
congestion, but the congestion emerges from modelled car-following behaviour
rather than from a line you drew, and an advisor can ask where the numbers came
from and get an answer.

### The honest framing of the vision part

If the input is a simulation, there is no vision problem — you would read
vehicle counts out of SUMO directly. A vision model only earns its place if the
input is **video of real intersections**, which means a dataset of traffic
camera footage with vehicle annotations. That is a separate acquisition problem,
and it is the part that would make this a vision project rather than an
optimisation project. Worth being clear about which of the two you are
proposing, because they are different projects with different risks.

---

## Sources

- [ASL Citizen (Desai et al., NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf) — dataset, baselines, ~63% top-1 over 2,731 signs
- [Logos as a Well-Tempered Pre-train for Sign Language Recognition](https://arxiv.org/pdf/2505.10481) — pretraining and frozen-encoder few-shot results by vocabulary size
- [Transfer Learning for ASL](https://homes.cs.washington.edu/~ali/papers/Transfer_Learning_ASL.pdf) — up to 21% gain from MS-ASL pretraining
- [SHuBERT: Self-Supervised Sign Language Representation Learning](https://arxiv.org/html/2411.16765v1) — multi-stream self-supervision, ~5% over specialised models on ASL Citizen
- [A Comprehensive Study on Deep Learning-based Methods for SLR](https://arxiv.org/pdf/2007.12530) — why models without pretraining collapse onto frequent glosses
- [SignBart](https://arxiv.org/pdf/2506.21592) — skeleton-sequence model; headline 96.04% is **LSA-64**, not ASL Citizen, and is not comparable
