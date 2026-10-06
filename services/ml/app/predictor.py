"""Sign classification.

Two implementations behind one interface. `OnnxPredictor` is the real trained
model from ml/asl. `StubPredictor` stands in until that model exists.

The stub is not throwaway scaffolding - it is how M0 gets built. Wiring the
browser, the segmenter, the agent and speech together while the classifier is
a known quantity means that when the real model lands, any breakage is
unambiguously the model's. Debugging a novel pipeline and a novel model at the
same time is how weeks disappear.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np

# SIGN_ARTIFACT_DIR lets an experiment (e.g. ml/asl_mamba/serve) be served
# without overwriting the committed model. Unset, behaviour is unchanged.
ARTIFACT_DIR = Path(
    os.environ.get("SIGN_ARTIFACT_DIR")
    or Path(__file__).resolve().parents[3] / "ml" / "asl" / "artifacts"
)
MODEL_PATH = ARTIFACT_DIR / "sign_classifier.onnx"
LABELS_PATH = ARTIFACT_DIR / "labels.json"


class Predictor(Protocol):
    name: str
    #: False means this predictor cannot actually recognise signs.
    ready: bool

    def predict(self, window: np.ndarray) -> tuple[str, float] | None: ...


class NullPredictor:
    """Recognises nothing, because we have not trained anything yet.

    This is the default until ml/asl exports a model, and it is deliberately
    silent rather than fake. An earlier version cycled a demo vocabulary so the
    pipeline could be exercised end to end -- and the result was a screen that
    appeared to read signs from a person who had not signed anything. Fabricated
    input is worse than no input: it puts words in the mouth of the exact user
    this system exists to serve, and it can add items to a real order.

    Set SIGN_STUB=1 to re-enable the fake emitter for pipeline debugging only.
    """

    name = "none"
    ready = False

    def predict(self, window: np.ndarray) -> tuple[str, float] | None:
        return None


class StubPredictor:
    """Cycles a demo vocabulary. Debugging only -- never on by default."""

    name = "stub"
    ready = False

    DEMO_SEQUENCE = ["WANT", "BURGER", "TWO", "FRIES", "DRINK", "FINISH"]

    def __init__(self) -> None:
        self._i = 0

    def predict(self, window: np.ndarray) -> tuple[str, float]:
        label = self.DEMO_SEQUENCE[self._i % len(self.DEMO_SEQUENCE)]
        self._i += 1
        return label, 0.95


class OnnxPredictor:
    """The trained landmark Transformer, exported to ONNX."""

    name = "onnx"
    ready = True

    def __init__(self) -> None:
        import onnxruntime as ort  # imported lazily: optional dependency

        self.session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        with LABELS_PATH.open() as f:
            self.labels: list[str] = json.load(f)

    def predict(self, window: np.ndarray) -> tuple[str, float]:
        batch = window[None, :, :].astype(np.float32)
        logits = self.session.run(None, {self.input_name: batch})[0][0]

        # Softmax in float64: exp() on float32 logits overflows for confident
        # predictions, which shows up as nan confidence and silently dropped signs.
        z = logits.astype(np.float64)
        e = np.exp(z - z.max())
        probs = e / e.sum()

        idx = int(probs.argmax())
        return self.labels[idx], float(probs[idx])


def load_predictor() -> Predictor:
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        try:
            predictor = OnnxPredictor()
            print(f"[sign] loaded trained model: {MODEL_PATH.name}")
            return predictor
        except Exception as exc:  # noqa: BLE001
            # Never fall back silently. A stub quietly standing in for a model
            # you believe is loaded is a genuinely dangerous failure mode.
            print(f"[sign] FAILED to load {MODEL_PATH}: {exc}")

    if os.environ.get("SIGN_STUB") == "1":
        print("[sign] SIGN_STUB=1 - emitting FAKE signs. Debugging only.")
        return StubPredictor()

    print(f"[sign] no trained model at {MODEL_PATH}. Sign input is INACTIVE.")
    print("[sign] train one with ml/asl, or set SIGN_STUB=1 to fake it.")
    return NullPredictor()
