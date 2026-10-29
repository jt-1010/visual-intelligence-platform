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


def test_same_sign_can_repeat_without_rest_after_cooldown():
    """Tracking jitter must not latch a label forever, even with moving wrists."""
    seg = SignSegmenter()
    emitted_at = []
    for t in range(180):
        if seg.push(make_frame(t, moving=True)) is not None:
            if seg.accept("BURGER", 0.95):
                emitted_at.append(t)

    assert len(emitted_at) >= 2, "the same sign must recover without resting hands"
    assert all(
        later - earlier > seg.debounce_frames * 2
        for earlier, later in zip(emitted_at, emitted_at[1:])
    ), "overlapping windows must stay suppressed during the cooldown"


def test_cooldown_expires_only_after_twice_debounce_frames():
    seg = SignSegmenter()
    for _ in range(seg.stability_windows):
        seg.accept("BURGER", 0.95)
    assert seg.last_label == "BURGER"

    for t in range(seg.debounce_frames * 2):
        seg.push(make_frame(t, moving=True))
        assert seg.last_label == "BURGER"
        assert not seg.accept("BURGER", 0.95)

    seg.push(make_frame(seg.debounce_frames * 2, moving=True))
    assert seg.last_label is None
    assert seg.accept("BURGER", 0.95)


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
    """A confident reading still has to be repeated before it is believed.

    This used to assert that one confident call emits. It no longer does, and
    the change is the point: a single window is exactly what a hand travelling
    through a sign produces.
    """
    seg = SignSegmenter()
    threshold = spec()["segmentation"]["confidence_threshold"]

    for _ in range(seg.stability_windows + 2):
        assert not seg.accept("BURGER", threshold - 0.01), "unsure readings never emit"

    emitted = [seg.accept("BURGER", threshold + 0.01) for _ in range(seg.stability_windows)]
    assert emitted[-1] is True, "a reading held for stability_windows is believed"
    assert emitted[:-1] == [False] * (seg.stability_windows - 1), "and not before"


def test_a_sign_passed_through_on_the_way_in_is_not_emitted():
    """Signing FOUR read as "spicy four spicy" -- the hand on its way in and out.

    The flanking readings last a window or two; the sign itself is held. Only
    the held one may reach the person.
    """
    seg = SignSegmenter()
    emitted = []

    def feed(label: str, windows: int) -> None:
        for _ in range(windows):
            if seg.accept(label, 0.95):
                emitted.append(label)

    feed("SPICY", 2)   # travelling into the handshape
    feed("FOUR", 8)    # the sign, held
    feed("SPICY", 2)   # travelling back out

    assert emitted == ["FOUR"], f"got {emitted}"


def test_a_flicker_does_not_break_a_held_sign():
    """One stray window inside a held sign must not split it into two signs."""
    seg = SignSegmenter()
    emitted = []
    for label in ["FOUR"] * 5 + ["SPICY"] + ["FOUR"] * 5:
        if seg.accept(label, 0.95):
            emitted.append(label)

    assert emitted == ["FOUR"], f"got {emitted}"


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


def test_low_confidence_breaks_candidate_stability():
    seg = SignSegmenter()
    for _ in range(seg.stability_windows - 1):
        assert not seg.accept("BURGER", 0.95)
    assert not seg.accept("BURGER", seg.confidence_threshold - 0.01)
    for _ in range(seg.stability_windows - 1):
        assert not seg.accept("BURGER", 0.95)
    assert seg.accept("BURGER", 0.95)


def test_single_window_stability_setting():
    seg = SignSegmenter()
    seg.stability_windows = 1
    assert seg.accept("BURGER", 0.95)
    assert not seg.accept("BURGER", 0.95)


def test_reset_clears_all_state_and_allows_same_sign():
    seg = SignSegmenter()
    for t in range(40):
        if seg.push(make_frame(t, moving=True)) is not None:
            seg.accept("BURGER", 0.95)
    assert seg.last_label == "BURGER"
    seg.quiet_windows = 1

    seg.reset()

    assert not seg.buffer
    assert seg.frames_seen == 0
    assert seg.frames_since_emit == 10**9
    assert seg.last_label is None
    assert seg.candidate is None
    assert seg.candidate_windows == 0
    assert seg.quiet_windows == 0
    for _ in range(seg.stability_windows - 1):
        assert not seg.accept("BURGER", 0.95)
    assert seg.accept("BURGER", 0.95)
