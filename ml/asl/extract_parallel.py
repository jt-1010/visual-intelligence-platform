"""Extract landmarks from ASL Citizen across every CPU core.

    python ml/asl/extract_parallel.py --out citizen_full
    python ml/asl/extract_parallel.py --out citizen_vocab --vocabulary
    python ml/asl/extract_parallel.py --out smoke --limit 64

WHY THIS EXISTS SEPARATELY FROM prepare.py

MediaPipe is the whole cost of this pipeline -- about 2.5s per clip, entirely
on the CPU, and the GPU sits idle throughout. Single-threaded, the full 83,399
clip corpus takes around 57 hours. Spread over 16 cores it is a single night.

The parallelism has to be processes, not threads: each worker builds its own
MediaPipe landmarkers, which hold native state that cannot be shared, and the
GIL would serialise the work anyway.

Workers write their own .npy shards rather than returning arrays. The full
corpus is ~1.6GB of float32, and pickling that back through a queue is both
slow and a memory spike in the parent for no benefit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import ASL_CITIZEN_ROOT, PREPARED, frames  # noqa: E402

SHARDS = PREPARED / "_shards"


def load_index() -> list[dict]:
    """Every clip in the corpus: gloss, signer, filename."""
    splits = ASL_CITIZEN_ROOT / "splits"
    if not splits.exists():
        sys.exit(f"No splits at {splits}. Is ASL_CITIZEN_ROOT set correctly?")

    rows: list[dict] = []
    for name in ("train.csv", "val.csv", "test.csv"):
        path = splits / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "file": (r.get("Video file") or "").strip(),
                        "gloss": (r.get("Gloss") or "").strip().lower(),
                        "signer": (r.get("Participant ID") or "").strip(),
                        "split": name.replace(".csv", ""),
                    }
                )
    return [r for r in rows if r["file"] and r["gloss"]]


def worker(task: tuple[int, list[dict]]) -> tuple[int, int, int]:
    """Process one chunk in its own process. Returns (shard, kept, skipped)."""
    shard_id, rows = task

    # Pin each worker to one thread BEFORE the native libraries load.
    #
    # MediaPipe runs TFLite with the XNNPACK delegate, which spawns its own
    # thread pool per process. Fourteen workers each spawning a pool onto
    # sixteen cores is heavy oversubscription: the machine spends its time
    # context-switching, and throughput scaled only 1.7x instead of ~10x.
    # One thread per worker, parallelism from the process count instead.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"

    # Imported inside the worker: MediaPipe initialises native state at import
    # time, and it must happen after the fork/spawn, once per process, and
    # after the thread limits above are in place.
    from prepare import make_extractor, video_to_array

    videos_dir = ASL_CITIZEN_ROOT / "videos"
    target = frames()

    extractor = make_extractor()
    X, y, signers, splits = [], [], [], []
    skipped = 0

    try:
        for r in rows:
            path = videos_dir / r["file"]
            if not path.exists():
                skipped += 1
                continue
            try:
                seq = video_to_array(path, extractor)
            except Exception:  # noqa: BLE001
                # One unreadable clip must not lose the other 5,000 in this
                # shard. Count it and carry on.
                skipped += 1
                continue
            if seq is None or len(seq) < 2:
                skipped += 1
                continue

            from app.features import resample

            X.append(resample(seq, target).astype(np.float32))
            y.append(r["gloss"])
            signers.append(r["signer"])
            splits.append(r["split"])
    finally:
        extractor.close()

    if X:
        np.savez_compressed(
            SHARDS / f"shard_{shard_id:04d}.npz",
            X=np.stack(X),
            y=np.array(y),
            signers=np.array(signers),
            splits=np.array(splits),
        )
    return shard_id, len(X), skipped


def merge(out_name: str) -> None:
    """Stitch the shards into the arrays train.py expects."""
    shards = sorted(SHARDS.glob("shard_*.npz"))
    if not shards:
        sys.exit("No shards produced.")

    Xs, ys, ss, sp = [], [], [], []
    for s in shards:
        # Context-managed: np.load on an .npz keeps the file open until the
        # handle is closed, and on Windows that makes the unlink() below fail
        # with "file is being used by another process" -- after the merge has
        # already succeeded, which is a confusing place to blow up.
        with np.load(s, allow_pickle=False) as d:
            Xs.append(d["X"])
            ys.append(d["y"])
            ss.append(d["signers"])
            sp.append(d["splits"])

    X = np.concatenate(Xs)
    y_raw = np.concatenate(ys)
    signers = np.concatenate(ss)
    splits = np.concatenate(sp)

    labels = sorted(set(y_raw.tolist()))
    idx = {s: i for i, s in enumerate(labels)}
    y = np.array([idx[s] for s in y_raw], dtype=np.int64)

    PREPARED.mkdir(parents=True, exist_ok=True)
    np.save(PREPARED / f"{out_name}_X.npy", X)
    np.save(PREPARED / f"{out_name}_y.npy", y)
    np.save(PREPARED / f"{out_name}_signers.npy", signers)
    np.save(PREPARED / f"{out_name}_splits.npy", splits)
    with (PREPARED / f"{out_name}_labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"\nWrote {PREPARED / (out_name + '_X.npy')}")
    print(f"  shape {X.shape}  ({X.nbytes / 1e9:.2f} GB)")
    print(f"  {len(labels)} classes, {len(set(signers.tolist()))} signers")

    counts = np.bincount(y, minlength=len(labels))
    print(f"  clips per class: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}")

    for s in shards:
        s.unlink()
    print(f"  cleaned up {len(shards)} shards")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output name, e.g. citizen_full")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, help="first N clips only (smoke test)")
    ap.add_argument("--vocabulary", action="store_true", help="ordering vocabulary only")
    ap.add_argument("--chunk", type=int, default=250, help="clips per shard")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep shards already on disk and extract only what is missing",
    )
    args = ap.parse_args()

    rows = load_index()
    print(f"ASL Citizen index: {len(rows)} clips")

    if args.vocabulary:
        from vocabulary import ORDERING_VOCABULARY, VARIANT_SUFFIX

        wanted = {g for gs in ORDERING_VOCABULARY.values() for g in gs}
        rows = [r for r in rows if VARIANT_SUFFIX.sub("", r["gloss"]) in wanted]
        print(f"  filtered to ordering vocabulary: {len(rows)} clips")

    if args.limit:
        rows = rows[: args.limit]
        print(f"  limited to {len(rows)} clips")

    if not rows:
        sys.exit("Nothing to extract.")

    SHARDS.mkdir(parents=True, exist_ok=True)

    chunks = [
        (i, rows[start : start + args.chunk])
        for i, start in enumerate(range(0, len(rows), args.chunk))
    ]

    # Extracting the full corpus takes hours, and the original behaviour was to
    # delete every shard on startup -- so a run that died at hour three threw
    # away three hours of work and began again. --resume keeps what is on disk
    # and processes only the shards that are missing.
    if args.resume:
        done = {int(p.stem.split("_")[1]) for p in SHARDS.glob("shard_*.npz")}
        before = len(chunks)
        chunks = [c for c in chunks if c[0] not in done]
        print(f"  resuming: {before - len(chunks)} shards already on disk")
    else:
        for old in SHARDS.glob("shard_*.npz"):
            old.unlink()

    if not chunks:
        print("  nothing left to extract")
        merge(args.out)
        return

    print(f"  {len(chunks)} shards of ~{args.chunk}, {args.workers} workers\n")

    t0 = time.time()
    done_clips = done_shards = skipped_total = 0

    # imap_unordered: shards vary in cost (clip length differs), so take
    # results as they land rather than waiting on a slow one in order.
    with Pool(processes=args.workers) as pool:
        for shard_id, kept, skipped in pool.imap_unordered(worker, chunks):
            done_shards += 1
            done_clips += kept
            skipped_total += skipped
            elapsed = time.time() - t0
            rate = done_clips / elapsed if elapsed else 0
            remaining = (len(rows) - done_clips - skipped_total) / rate if rate else 0
            print(
                f"  shard {done_shards}/{len(chunks)}  "
                f"{done_clips} clips  {elapsed/60:.1f} min elapsed  "
                f"~{remaining/60:.0f} min left  ({rate:.1f} clips/s)",
                flush=True,
            )

    print(f"\nExtracted {done_clips} clips ({skipped_total} skipped) in {(time.time()-t0)/60:.1f} min")
    merge(args.out)


if __name__ == "__main__":
    main()
