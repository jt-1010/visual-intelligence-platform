"""Data-independent checks on the Mamba classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mamba_model import MambaMixer, SignMambaClassifier  # noqa: E402

IN_DIM, FRAMES, CLASSES = 153, 32, 12


def make(**kw) -> SignMambaClassifier:
    torch.manual_seed(0)
    return SignMambaClassifier(CLASSES, in_dim=IN_DIM, frames=FRAMES, **kw).eval()


def test_output_shape_matches_transformer_contract():
    out = make()(torch.randn(5, FRAMES, IN_DIM))
    assert out.shape == (5, CLASSES)


def test_mixer_is_causal():
    """Changing later frames must not change earlier outputs.

    The forward-only version needs this to work live, frame by frame. If it
    failed, streaming would give different answers than running a whole window.
    """
    torch.manual_seed(0)
    mixer = MambaMixer(32).eval()
    a = torch.randn(1, FRAMES, 32)
    b = a.clone()
    b[:, 20:] = torch.randn(1, FRAMES - 20, 32)  # change only the future
    with torch.no_grad():
        ya, yb = mixer(a), mixer(b)
    assert torch.allclose(ya[:, :20], yb[:, :20], atol=1e-5)
    assert not torch.allclose(ya[:, 20:], yb[:, 20:], atol=1e-5)


def test_bidirectional_sees_the_future():
    x = torch.randn(1, FRAMES, IN_DIM)
    y = x.clone()
    y[:, -4:] += 1.0
    with torch.no_grad():
        assert not torch.allclose(make()(x), make()(y), atol=1e-6)


def test_batch_elements_are_independent():
    model = make()
    x = torch.randn(4, FRAMES, IN_DIM)
    with torch.no_grad():
        together = model(x)
        alone = torch.cat([model(x[i : i + 1]) for i in range(4)])
    assert torch.allclose(together, alone, atol=1e-5)


def test_gradients_reach_every_parameter():
    model = make()
    model.train()
    model(torch.randn(4, FRAMES, IN_DIM)).sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().any()]
    assert not dead, f"no gradient for: {dead}"


def test_decay_stays_negative():
    """The memory decay must always be negative. If it were positive the memory
    would grow every frame instead of fading, and training would blow up."""
    for m in make().modules():
        if isinstance(m, MambaMixer):
            assert (-torch.exp(m.A_log) < 0).all()


def test_dt_initialised_inside_the_documented_range():
    for m in make().modules():
        if isinstance(m, MambaMixer):
            dt = torch.nn.functional.softplus(m.dt_proj.bias)
            assert dt.min() >= 1e-4 - 1e-6 and dt.max() <= 1e-1 + 1e-3


def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from export_mamba import export  # noqa: E402

    model = make()
    path = tmp_path / "m.onnx"
    export(model, FRAMES, IN_DIM, path)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 8):
        x = torch.randn(batch, FRAMES, IN_DIM)
        with torch.no_grad():
            expected = model(x).numpy()
        got = session.run(None, {"landmarks": x.numpy()})[0]
        assert abs(got - expected).max() < 1e-4
