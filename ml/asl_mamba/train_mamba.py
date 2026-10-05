"""Train the Mamba classifier on the same data, split and augmentation as ml/asl/train.py.

    ml/.venv/Scripts/python.exe ml/asl_mamba/train_mamba.py --data citizen_vocab --merge-variants

Requires a CUDA GPU unless --allow-cpu is passed. Artifacts go to ml/asl_mamba/artifacts/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "asl"))  # config, train (shared pipeline)
sys.path.insert(0, str(HERE))                 # mamba_model
from config import PREPARED  # noqa: E402,F401
from train import augment, load, split_by_signer  # noqa: E402
from mamba_model import build  # noqa: E402

ARTIFACTS = HERE / "artifacts"


def pick_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  device: cuda ({torch.cuda.get_device_name(0)})")
        return dev
    if not allow_cpu:
        sys.exit(
            "No CUDA GPU visible to torch. Refusing to train on CPU.\n"
            "  Check: ml/.venv/Scripts/python.exe -c \"import torch; print(torch.version.cuda, torch.cuda.is_available())\"\n"
            "  A CPU-only torch build is the usual cause; reinstall from "
            "https://download.pytorch.org/whl/cu128\n"
            "  Pass --allow-cpu for a smoke test."
        )
    print("  device: cpu (--allow-cpu)")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dl, device) -> tuple[float, float]:
    model.eval()
    correct = top5 = seen = 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        k = min(5, logits.shape[1])
        top5 += (logits.topk(k, dim=1).indices == yb[:, None]).any(1).sum().item()
        seen += len(yb)
    return correct / max(1, seen), top5 / max(1, seen)


def run(args) -> None:
    X, y, signers, labels = load(args.data, merge=args.merge_variants)
    print(f"{args.data}: {X.shape[0]} clips, {len(labels)} classes, {len(np.unique(signers))} signers")

    train_idx, val_idx = split_by_signer(signers, y, args.val_frac, args.seed)
    device = pick_device(args.allow_cpu)

    Xt, yt = torch.from_numpy(X).float(), torch.from_numpy(y).long()
    train_dl = DataLoader(TensorDataset(Xt[train_idx], yt[train_idx]),
                          batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xt[val_idx], yt[val_idx]), batch_size=args.batch)

    model = build(
        len(labels), in_dim=X.shape[2], frames=X.shape[1],
        dim=args.dim, depth=args.depth, d_state=args.d_state,
        bidirectional=not args.causal,
    ).to(device)
    print(f"  mode: {'causal' if args.causal else 'bidirectional'}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(train_dl)), pct_start=0.2
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_state, patience = 0.0, None, 0
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = augment(xb.to(device)), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item() * len(xb)

        acc, acc5 = evaluate(model, val_dl, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3}  loss {total_loss / len(train_idx):.3f}  "
                  f"val top1 {acc:.3f}  top5 {acc5:.3f}  ({time.perf_counter() - t0:.0f}s)")

        if acc > best:
            best, patience = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop at epoch {epoch} (no improvement for {args.patience})")
                break

    print(f"\nBEST signer-independent top-1: {best:.3f}   "
          f"(trained {time.perf_counter() - t0:.0f}s, stopped at epoch {epoch})")

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {
        "state_dict": model.state_dict(), "labels": labels,
        "in_dim": X.shape[2], "frames": X.shape[1],
        "arch": {"dim": args.dim, "depth": args.depth, "d_state": args.d_state,
                 "bidirectional": not args.causal},
    }
    torch.save(ckpt, ARTIFACTS / f"{args.out}.pt")
    with (ARTIFACTS / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Wrote {ARTIFACTS / (args.out + '.pt')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="citizen_vocab", help="prepared dataset name")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    # 30, not ml/asl's 15: docs/training.md records an aggressive patience
    # reading as a result. Compare runs that stopped for the same reason.
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0, help="keep equal to the Transformer run")
    ap.add_argument("--merge-variants", action="store_true",
                    help="treat eat1/eat2 as one sign (match the Transformer run)")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--causal", action="store_true",
                    help="forward-only Mamba (the variant that can stream frame by frame)")
    ap.add_argument("--out", default="sign_mamba", help="artifact basename")
    ap.add_argument("--allow-cpu", action="store_true", help="smoke tests only")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
