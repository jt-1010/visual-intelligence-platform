"""Export the Mamba classifier to ONNX and check it against PyTorch.

Writes ml/asl_mamba/artifacts/sign_mamba.onnx, not the path the service loads by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mamba_model import SignMambaClassifier  # noqa: E402

ARTIFACTS = HERE / "artifacts"
CHECKPOINT = ARTIFACTS / "sign_mamba.pt"
ONNX_PATH = ARTIFACTS / "sign_mamba.onnx"


def export(model: SignMambaClassifier, frames: int, in_dim: int, path: Path) -> None:
    model.eval()
    torch.onnx.export(
        model,
        torch.randn(1, frames, in_dim),
        str(path),
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,  # legacy exporter; see ml/asl/export.py for why
    )


def verify(model: SignMambaClassifier, path: Path, frames: int, in_dim: int) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for batch in (1, 2, 8, 64):
        x = rng.standard_normal((batch, frames, in_dim)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        got = session.run(None, {"landmarks": x})[0]
        if got.shape != expected.shape:
            sys.exit(f"Shape mismatch at batch {batch}: {got.shape} vs {expected.shape}")
        diff = float(np.abs(got - expected).max())
        worst = max(worst, diff)
        print(f"  batch {batch:3}: max |torch - onnx| = {diff:.2e}")
    return worst


def main() -> None:
    if not CHECKPOINT.exists():
        sys.exit(f"No checkpoint at {CHECKPOINT}. Run train_mamba.py first.")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = SignMambaClassifier(
        len(ckpt["labels"]), in_dim=ckpt["in_dim"], frames=ckpt["frames"], **ckpt["arch"]
    )
    model.load_state_dict(ckpt["state_dict"])

    export(model, ckpt["frames"], ckpt["in_dim"], ONNX_PATH)
    print(f"Wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB, {len(ckpt['labels'])} classes)")

    if verify(model, ONNX_PATH, ckpt["frames"], ckpt["in_dim"]) > 1e-4:
        sys.exit("Export does not match PyTorch. Do not ship this model.")
    print("Export verified across batch sizes.")


if __name__ == "__main__":
    main()
