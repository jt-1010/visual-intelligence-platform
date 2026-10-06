"""Tests for the sign segmenter.

These cover the two failure modes that make a working classifier look broken:
classifying still hands, and emitting the same sign repeatedly because it spans
many overlapping windows.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import feature_length, spec  # noqa: E402
from app.segmenter import SignSegmenter  # noqa: E402

N_HAND = spec()["hands"]["count"] * spec()["hands"]["points_per_hand"]
POSE_NAMES = spec()["pose_index_names"]


def make_frame(t: int, moving: bool) -> list[float]:
    """A synthetic frame with anatomically plausible shoulders and optional wrist motion."""
    pts = np.zeros((N_HAND + len(POSE_NAMES), 3), dtype=np.float32)

    pts[N_HAND + POSE_NAMES.index("left_shoulder")] = [0.40, 0.40, 0.0]
    pts[N_HAND + POSE_NAMES.index("right_shoulder")] = [0.60, 0.40, 0.0]
    pts[N_HAND + POSE_NAMES.index("nose")] = [0.50, 0.30, 0.0]

    if moving:
        y = 0.55 + 0.05 * math.sin(t * 2 * math.pi / 20.0)
        x = 0.45 + 0.05 * math.cos(t * 2 * math.pi / 20.0)
    else:
        x, y = 0.45, 0.70

    pts[N_HAND + POSE_NAMES.index("left_wrist")] = [x, y, 0.0]
    pts[N_HAND + POSE_NAMES.index("right_wrist")] = [1.0 - x, y, 0.0]
    return pts.reshape(-1).tolist()


def test_frame_length_matches_spec():
    assert len(make_frame(0, True)) == feature_length()


def test_still_hands_never_produce_a_candidate():
    """The motion gate must hold shut for a person just standing there."""
    seg = SignSegmenter()
    candidates = [seg.push(make_frame(t, moving=False)) for t in range(120)]
    assert all(c is None for c in candidates)


def test_moving_hands_produce_candidates():
    seg = SignSegmenter()
    got = [c for t in range(120) if (c := seg.push(make_frame(t, moving=True))) is not None]
    assert got, "moving wrists should open the motion gate"
    assert got[0]["window"].shape == (spec()["frames"], feature_length())
    assert got[0]["motion"] > spec()["segmentation"]["motion_gate_threshold"]


def test_a_held_sign_is_emitted_exactly_once():
    """One sign spans many windows; it must be emitted once, not once per window.

    This assertion used to read `accepted <= (120 // debounce) + 1`, which
    allowed six. It passed while the demo was producing
    "want want want want want", because emitting reset the very counter the
    debounce compared against, making it a rate limit rather than a latch.
    The sign is held for the whole run here, so the only correct answer is one.
    """
    seg = SignSegmenter()
    accepted = 0
    for t in range(120):
        if seg.push(make_frame(t, moving=True)) is not None:
            if seg.accept("BURGER", 0.95):
                accepted += 1

    assert accepted == 1, f"emitted {accepted} times"


def test_a_different_sign_is_emitted_immediately():
    """Changing sign is a boundary in itself -- no pause should be required."""
    seg = SignSegmenter()
    emitted = []
    for t in range(120):
        if seg.push(make_frame(t, moving=True)) is not None:
            label = "BURGER" if t < 60 else "FRIES"
            if seg.accept(label, 0.95):
                emitted.append(label)

    assert emitted == ["BURGER", "FRIES"]


def test_low_confidence_is_rejected():
    seg = SignSegmenter()
    threshold = spec()["segmentation"]["confidence_threshold"]
    assert not seg.accept("BURGER", threshold - 0.01)
    assert seg.accept("BURGER", threshold + 0.01)


def test_returning_to_rest_clears_the_debounce():
    """Signing X, resting, then signing X again must emit twice, not once."""
    seg = SignSegmenter()

    def run(moving: bool, frames: int) -> int:
        n = 0
        for t in range(frames):
            if seg.push(make_frame(t, moving=moving)) is not None:
                if seg.accept("MORE", 0.95):
                    n += 1
        return n

    first = run(True, 40)
    run(False, 40)          # hands drop to the lap: a phrase boundary
    second = run(True, 40)

    assert first >= 1
    assert second >= 1, "a rest between two identical signs must reset the debounce"
