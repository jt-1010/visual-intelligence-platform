"""Train the sign classifier.

    python ml/asl/train.py --data citizen --epochs 60

THE SPLIT IS THE POINT

Data is split **by signer**, never at random. A random split puts the same
person in train and test, and the model scores well by recognising *them* --
their proportions, their camera, their habits -- rather than the sign. The
number that comes out is inflated and means nothing about a stranger walking
up to the terminal.

A signer-independent split gives a lower number that is actually about signs.
That is the one to report, and the one to optimise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from config import ARTIFACTS, PREPARED  # noqa: E402
from model import build  # noqa: E402


#: ASL Citizen labels regional/stylistic variants separately: eat1 and eat2,
#: want1 and want2. They are different ways of producing the SAME sign.
VARIANT_SUFFIX = re.compile(r"_?\d+$")


def merge_variants(y: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Collapse sign variants into one class each.

    Worth doing for three separate reasons, all pulling the same way:

    1. Distinguishing eat1 from eat2 is not a task we want. Both mean EAT, and
       the terminal does the same thing either way -- so the model was being
       penalised for answers that were, for our purposes, correct. Measured:
       30% of its errors were variant confusions.
    2. It doubles the data for merged classes. eat1 + eat2 is 63 clips rather
       than two classes of 31 -- and scarcity per class is this model's binding
       constraint.
    3. It shrinks the problem from 84 classes to 64.

    Anything that wants the finer distinction back can train without this.
    """
    bases = [VARIANT_SUFFIX.sub("", l) for l in labels]
    new_labels = sorted(set(bases))
    remap = {i: new_labels.index(bases[i]) for i in range(len(labels))}
    return np.array([remap[int(v)] for v in y], dtype=np.int64), new_labels


def load(name: str, merge: bool = False):
    X = np.load(PREPARED / f"{name}_X.npy")
    y = np.load(PREPARED / f"{name}_y.npy")
    signers = np.load(PREPARED / f"{name}_signers.npy")
    with (PREPARED / f"{name}_labels.json").open(encoding="utf-8") as f:
        labels = json.load(f)

    if merge:
        before = len(labels)
        y, labels = merge_variants(y, labels)
        print(f"  merged sign variants: {before} -> {len(labels)} classes")

    return X, y, signers, labels


def split_by_signer(
    signers: np.ndarray,
    y: np.ndarray,
    val_frac: float,
    seed: int,
    holdout: list[str] | None = None,
):
    """Hold out whole signers, not random clips.

    Signers are assigned greedily smallest-first so the validation set lands
    near the requested size without splitting anyone across both sides.

    `holdout` forces an exact set of signers into validation, which is what
    makes pretraining honest. Pretraining and fine-tuning run over different
    subsets of the corpus, so each picking its own validation signers put 13 of
    the fine-tune's 16 held-out signers inside the pretraining TRAINING set --
    the backbone had already watched those people sign, and the fine-tuned
    model scored 99.0% on them. That number was not signer-independent and
    reporting it as one would have been wrong. Pass the fine-tune's validation
    signers here when pretraining and the backbone never sees them.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(signers)

    if holdout:
        val_signers = [s for s in unique if s in set(holdout)]
        missing = set(holdout) - set(val_signers)
        if missing:
            print(f"  note: {len(missing)} held-out signer(s) absent from this data")
        val_mask = np.isin(signers, val_signers)
        train_idx = np.flatnonzero(~val_mask)
        val_idx = np.flatnonzero(val_mask)
        print(f"  train: {len(train_idx)} clips from {len(unique) - len(val_signers)} signers")
        print(f"  val  : {len(val_idx)} clips from {len(val_signers)} signers (forced holdout)")
        return train_idx, val_idx

    if len(unique) < 2:
        # Single-signer data (our own recordings, before we have several
        # people). Fall back to a random split and say so loudly -- the
        # resulting accuracy is not signer-independent and must not be
        # reported as if it were.
        print("  WARNING: only one signer in this data.")
        print("  Falling back to a RANDOM split. Accuracy will be optimistic")
        print("  and is NOT comparable to a signer-independent number.")
        idx = rng.permutation(len(y))
        cut = int(len(idx) * (1 - val_frac))
        return idx[:cut], idx[cut:]

    rng.shuffle(unique)
    target = int(len(y) * val_frac)
    val_signers, held = [], 0
    for s in unique:
        if held >= target and len(val_signers) > 0:
            break
        val_signers.append(s)
        held += int((signers == s).sum())

    val_mask = np.isin(signers, val_signers)
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)

    print(f"  train: {len(train_idx)} clips from {len(unique) - len(val_signers)} signers")
    print(f"  val  : {len(val_idx)} clips from {len(val_signers)} signers (held out entirely)")
    return train_idx, val_idx


def load_backbone(model, checkpoint: Path, device) -> "torch.nn.Module":
    """Load a pretrained model's weights, except the classification head.

    Pretraining happens on the full 2,731-sign corpus; fine-tuning happens on
    our ~58. The two models differ only in the final Linear layer, so every
    other tensor transfers. Dropping head.* by name rather than by
    strict=False is deliberate: a silent shape mismatch anywhere else would
    otherwise be swallowed, and a backbone that failed to load looks exactly
    like a backbone that loaded and did not help.
    """
    if not checkpoint.exists():
        sys.exit(f"No checkpoint at {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    src = ckpt["state_dict"]
    dst = model.state_dict()

    transferred, skipped = [], []
    for k, v in src.items():
        if k.startswith("head."):
            skipped.append(k)
        elif k in dst and dst[k].shape == v.shape:
            dst[k] = v
            transferred.append(k)
        else:
            skipped.append(k)

    if not transferred:
        sys.exit(f"Nothing transferred from {checkpoint}. Architecture mismatch?")

    model.load_state_dict(dst)
    print(f"  loaded backbone from {checkpoint.name}: "
          f"{len(transferred)} tensors transferred, {len(skipped)} skipped "
          f"({len(ckpt['labels'])} -> {model.head.out_features} classes)")
    return model


def set_backbone_trainable(model, trainable: bool) -> None:
    """Freeze everything but the head.

    With ~31 clips per class, letting the whole network move immediately would
    undo the pretraining before the randomly-initialised head has learned
    anything useful. Train the head first, then release the rest.
    """
    for name, param in model.named_parameters():
        if not name.startswith("head."):
            param.requires_grad = trainable


def augment(batch: torch.Tensor) -> torch.Tensor:
    """Cheap, label-preserving jitter.

    With ~31 clips per sign, augmentation is doing a lot of the work. All three
    of these mimic real variation -- where someone stands, how big they sign,
    and frames MediaPipe drops -- rather than adding abstract noise.
    """
    b = batch.shape[0]
    # Global translation: the person stands slightly off-centre.
    batch = batch + torch.randn(b, 1, 1, device=batch.device) * 0.02
    # Global scale: bigger or smaller signing.
    batch = batch * (1 + torch.randn(b, 1, 1, device=batch.device) * 0.05)
    # Frame dropout: tracking blinks. Zeroing matches what the browser sends
    # when a hand is not detected, so the model sees the real failure mode.
    mask = (torch.rand(b, batch.shape[1], 1, device=batch.device) > 0.05).float()
    return batch * mask


def run(args) -> None:
    X, y, signers, labels = load(args.data, merge=args.merge_variants)
    print(f"{args.data}: {X.shape[0]} clips, {len(labels)} classes, {len(np.unique(signers))} signers")

    counts = np.bincount(y, minlength=len(labels))
    print(f"  clips per class: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}")

    holdout = args.holdout_signers.split(',') if args.holdout_signers else None
    train_idx, val_idx = split_by_signer(signers, y, args.val_frac, args.seed, holdout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()

    train_ds = TensorDataset(Xt[train_idx], yt[train_idx])
    val_ds = TensorDataset(Xt[val_idx], yt[val_idx])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    model = build(len(labels), in_dim=X.shape[2], frames=X.shape[1]).to(device)

    if args.init_from:
        model = load_backbone(model, Path(args.init_from), device)
        if args.freeze_epochs > 0:
            set_backbone_trainable(model, False)
            print(f"  backbone frozen for the first {args.freeze_epochs} epochs")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(train_dl)), pct_start=0.2
    )
    # Label smoothing: with this little data the model would otherwise become
    # overconfident on the training signers.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_state, patience = 0.0, None, 0
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if args.init_from and args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, True)
            print(f"  epoch {epoch}: backbone unfrozen")

        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            xb = augment(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        correct = top5 = seen = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                k = min(5, logits.shape[1])
                top5 += (logits.topk(k, dim=1).indices == yb[:, None]).any(1).sum().item()
                seen += len(yb)

        acc = correct / max(1, seen)
        acc5 = top5 / max(1, seen)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3}  loss {total_loss/len(train_idx):.3f}  "
                  f"val top1 {acc:.3f}  top5 {acc5:.3f}")

        if acc > best:
            best, patience = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop at epoch {epoch} (no improvement for {args.patience})")
                break

    print(f"\nBEST signer-independent top-1: {best:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "labels": labels,
                "in_dim": X.shape[2], "frames": X.shape[1]},
               ARTIFACTS / f"{args.out}.pt")
    with (ARTIFACTS / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Wrote {ARTIFACTS / (args.out + '.pt')}")
    print("Next: python ml/asl/export.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="citizen", help="prepared dataset name")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge-variants", action="store_true",
                    help="treat eat1/eat2 as one sign (recommended)")
    ap.add_argument("--init-from", help="checkpoint to load a pretrained backbone from")
    ap.add_argument("--holdout-signers",
                    help="comma-separated signer ids forced into validation; use the "
                         "fine-tune's validation signers when pretraining so the "
                         "backbone never sees them")
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="train only the head for this many epochs first")
    ap.add_argument("--out", default="sign_classifier",
                    help="artifact basename, e.g. pretrained")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
