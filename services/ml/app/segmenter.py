"""Turning a continuous landmark stream into discrete sign events.

This is the hard part, and it is worth being explicit about why.

The classifier answers "which sign is this 32-frame clip?". A person signing
at a kiosk does not produce clips - they produce an unbroken stream with no
markers for where one sign ends and the next begins. Everything below exists
to find those boundaries:

  motion gate   Hands resting in the lap still produce landmarks, and a
                classifier handed still hands will confidently return
                whichever sign happens to look most like stillness. We only
                classify when the wrists are actually moving.

  sliding window  We re-classify every `stride` frames rather than waiting for
                a clip to end, so the first prediction lands mid-sign instead
                of after it.

  debounce      One sign spans many overlapping windows, so a naive loop emits
                the same sign five times in a row. Debouncing is the
                difference between "MORE" and "MORE MORE MORE MORE".
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .features import normalize_frame, resample, spec


class SignSegmenter:
    def __init__(self) -> None:
        s = spec()["segmentation"]
        self.window_frames: int = s["window_frames"]
        self.stride: int = s["stride_frames"]
        self.motion_threshold: float = s["motion_gate_threshold"]
        self.confidence_threshold: float = s["confidence_threshold"]
        self.debounce_frames: int = s["debounce_frames"]
        # How many consecutive quiet windows count as a real phrase boundary.
        #
        # Kept small on purpose. `_wrist_motion` is already a mean over the
        # whole 32-frame buffer, so a brief hold partway through a sign barely
        # moves it; this is a little hysteresis on top of that, not the main
        # defence. Raising it delays the boundary by roughly stride/fps seconds
        # per step, which pushes back how soon a repeated sign can be signed.
        # Read with a default so an older feature_spec.json still loads.
        self.rest_windows: int = s.get("rest_windows_for_boundary", 2)

        # How many windows in a row must agree before a sign is believed.
        #
        # The window is 32 frames wide and slides every `stride`, so while a
        # hand travels INTO a sign the window is full of a handshape that is
        # neither the sign before nor the sign after -- and the classifier, which
        # must answer something, names whatever that transition most resembles.
        # Signing FOUR produced "spicy four spicy": the real sign in the middle,
        # flanked by two readings of the hand on its way in and out.
        #
        # A transition is brief and a held sign is not, so agreement over time
        # separates them. This costs stride * windows frames of latency before a
        # sign appears (~0.4s at the defaults), which is the honest price of not
        # emitting words the person never signed.
        self.stability_windows: int = s.get("stability_windows", 3)

        self.target_frames: int = spec()["frames"]
        self.pose_names: list[str] = spec()["pose_index_names"]
        self.n_hand: int = spec()["hands"]["count"] * spec()["hands"]["points_per_hand"]
        self.dims: int = spec()["dims"]

        self.buffer: deque[np.ndarray] = deque(maxlen=self.window_frames)
        self.frames_seen = 0
        self.frames_since_emit = 10**9
        self.last_label: str | None = None
        self.quiet_windows = 0
        self.candidate: str | None = None
        self.candidate_windows = 0

    def _wrist_motion(self) -> float:
        """Mean frame-to-frame wrist displacement, in shoulder-width units.

        Wrists rather than fingers: finger landmarks jitter constantly from
        tracking noise even when the hand is still, so they would keep the gate
        permanently open.
        """
        if len(self.buffer) < 2:
            return 0.0

        lw = self.pose_names.index("left_wrist")
        rw = self.pose_names.index("right_wrist")
        idx = [self.n_hand + lw, self.n_hand + rw]

        pts = np.stack([f.reshape(-1, self.dims)[idx] for f in self.buffer])
        deltas = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
        return float(deltas.mean())

    def push(self, raw_frame: list[float]) -> dict | None:
        """Feed one frame. Returns a sign event, or None."""
        vec = normalize_frame(np.asarray(raw_frame, dtype=np.float32))
        self.buffer.append(vec)
        self.frames_seen += 1
        self.frames_since_emit += 1

        if len(self.buffer) < self.window_frames:
            return None
        if self.frames_seen % self.stride != 0:
            return None

        motion = self._wrist_motion()
        if motion < self.motion_threshold:
            # Hands are at rest -- but only SUSTAINED rest is a phrase boundary.
            #
            # Most signs have a hold partway through where the wrists barely
            # move, and clearing last_label on the first quiet window let that
            # hold reopen the debounce mid-sign. The next window then re-emitted
            # the sign already in progress, which is how one WANT became
            # "want want want want want" and the phrase never settled long
            # enough to be sent. Requiring several consecutive quiet windows
            # distinguishes a hold inside a sign from a pause between signs.
            self.quiet_windows += 1
            if self.quiet_windows >= self.rest_windows:
                self.last_label = None
            return None

        self.quiet_windows = 0
        window = resample(np.stack(self.buffer), self.target_frames)
        return {"window": window, "motion": motion}

    def accept(self, label: str, confidence: float) -> bool:
        """Decide whether a prediction should actually be emitted.

        `last_label` is a latch on the sign currently being made, not a
        rate limit. The previous version re-emitted the same label every
        `debounce_frames`, because emitting reset the counter it was compared
        against -- so holding WANT for two seconds produced WANT three times,
        and holding it longer produced more. That is the "want want want want
        want" in the demo.

        A sign is emitted once when it starts. It can only be emitted again
        after a boundary: a different sign, or the sustained rest that `push`
        watches for.
        """
        if confidence < self.confidence_threshold:
            # An unsure reading breaks the run rather than extending it.
            self.candidate = None
            self.candidate_windows = 0
            return False

        # Hold the reading until it has been said the same way several times.
        if label != self.candidate:
            self.candidate = label
            self.candidate_windows = 1
            return False

        self.candidate_windows += 1
        if self.candidate_windows < self.stability_windows:
            return False

        if label == self.last_label:
            return False

        self.last_label = label
        self.frames_since_emit = 0
        return True

    def reset(self) -> None:
        self.buffer.clear()
        self.frames_seen = 0
        self.frames_since_emit = 10**9
        self.last_label = None
        self.quiet_windows = 0
        self.candidate = None
        self.candidate_windows = 0
