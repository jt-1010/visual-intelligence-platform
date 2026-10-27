import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import { FEATURE_SPEC, HAND_POINTS } from './featureSpec';
import {
  PresenceTracker,
  type PresenceState,
  type PresenceThresholds,
} from './presence';

/**
 * Turns a webcam stream into the 153-float landmark vector the model expects.
 *
 * Everything here runs on the user's device. The video element is never read
 * by anything but MediaPipe, and no pixel data leaves the browser -- only the
 * coordinate vector below. That is a real privacy property of the system, not
 * a side effect: a camera pointed at people ordering food is exactly the kind
 * of thing that should not be streamed to a server.
 */

export type CapturedFrame = {
  /** FEATURE_SPEC.featureLength floats: hands first, then the pose subset. */
  lm: number[];
  presence: PresenceState;
  arrived: boolean;
  departed: boolean;
  handsVisible: number;
  shoulderWidth: number | null;
  /**
   * The RAW model answer: did MediaPipe return a body this frame, at all?
   *
   * Exposed next to `presence` so the two can be compared on screen. This is
   * the honest way to settle "why not just use the model's output directly" --
   * watch how often the raw signal flickers while the gated one holds steady.
   */
  bodyDetected: boolean;
  /** Raw detected -> not-detected transitions in the last 30 seconds. */
  rawDropouts: number;
  /** How long the presence state machine has held its current state, in ms. */
  heldMs: number;
  /**
   * Raw normalised landmarks, kept for drawing the overlay.
   *
   * Worth the extra allocation per frame: being able to SEE the skeleton the
   * model is working from turns "detection feels flaky" into "the left wrist
   * drops out when the hand crosses the torso", which is a fixable statement.
   */
  overlay: {
    pose: { x: number; y: number; visibility?: number }[] | null;
    left: { x: number; y: number }[] | null;
    right: { x: number; y: number }[] | null;
  };
};

const EMPTY_POINT = [0, 0, 0];

function flatten(points: NormalizedLandmark[] | undefined, count: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    const p = points?.[i];
    if (p) out.push(p.x, p.y, p.z ?? 0);
    else out.push(...EMPTY_POINT);
  }
  return out;
}

export class LandmarkEngine {
  private pose?: PoseLandmarker;
  private hands?: HandLandmarker;
  private raf = 0;
  private lastTimestamp = -1;
  private lastVideoTime = -1;
  private presence = new PresenceTracker();
  private running = false;

  /** Timestamps of raw detection dropouts, trimmed to a 30s window. */
  private dropouts: number[] = [];
  private lastDetected = false;

  /** Live-tunable: the UI writes these while the camera keeps running. */
  setThresholds(next: Partial<PresenceThresholds>): void {
    this.presence.setThresholds(next);
  }

  async init(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');

    // GPU delegate where available; MediaPipe falls back to CPU on its own.
    [this.pose, this.hands] = await Promise.all([
      PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: '/models/pose_landmarker_lite.task', delegate: 'GPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
      }),
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: '/models/hand_landmarker.task', delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
      }),
    ]);
  }

  start(video: HTMLVideoElement, onFrame: (f: CapturedFrame) => void): void {
    if (!this.pose || !this.hands) throw new Error('LandmarkEngine.init() must run first');
    this.running = true;

    const tick = () => {
      if (!this.running) return;

      // requestAnimationFrame follows the monitor (240 Hz on some), not the camera.
      // The model's 32-frame window assumes ~fpsTarget fps, so skip repeated camera
      // images and anything faster than that; otherwise the window spans a fraction of a sign.
      const now = performance.now();
      const minGapMs = 1000 / FEATURE_SPEC.fpsTarget - 2;
      const fresh = video.currentTime !== this.lastVideoTime;
      if (video.readyState >= 2 && fresh && now - this.lastTimestamp >= minGapMs) {
        this.lastVideoTime = video.currentTime;
        // detectForVideo rejects non-increasing timestamps, and on a paused or
        // stalled stream performance.now() can repeat within a millisecond.
        const ts = Math.max(Math.round(now), this.lastTimestamp + 1);
        this.lastTimestamp = ts;

        const frame = this.capture(video, ts);
        if (frame) onFrame(frame);
      }

      this.raf = requestAnimationFrame(tick);
    };

    this.raf = requestAnimationFrame(tick);
  }

  private capture(video: HTMLVideoElement, ts: number): CapturedFrame | null {
    const poseResult = this.pose!.detectForVideo(video, ts);
    const handResult = this.hands!.detectForVideo(video, ts);

    const poseLm = poseResult.landmarks?.[0];

    // --- Hands, ordered [left, right] to match the training layout ---
    const perHand = FEATURE_SPEC.hands.pointsPerHand;
    let left: NormalizedLandmark[] | undefined;
    let right: NormalizedLandmark[] | undefined;

    handResult.landmarks?.forEach((points, i) => {
      // MediaPipe reports handedness from the camera's point of view. We keep
      // its label verbatim rather than correcting for the mirrored preview --
      // consistency with the training data is what matters, not anatomy.
      const label = handResult.handedness?.[i]?.[0]?.categoryName;
      if (label === 'Left') left = points;
      else if (label === 'Right') right = points;
      else if (!left) left = points;
      else right ??= points;
    });

    const lm = [...flatten(left, perHand), ...flatten(right, perHand)];

    // --- Pose subset ---
    for (const idx of FEATURE_SPEC.poseIndices) {
      const p = poseLm?.[idx];
      if (p) lm.push(p.x, p.y, p.z ?? 0);
      else lm.push(...EMPTY_POINT);
    }

    if (lm.length !== FEATURE_SPEC.featureLength) {
      console.error(`landmark vector is ${lm.length}, expected ${FEATURE_SPEC.featureLength}`);
      return null;
    }

    // --- Presence, from shoulder separation ---
    const ls = poseLm?.[11];
    const rs = poseLm?.[12];
    const shoulderWidth =
      ls && rs ? Math.hypot(ls.x - rs.x, ls.y - rs.y) : null;

    const bodyDetected = Boolean(poseLm);

    // Record every raw detected -> lost transition, so the UI can show how
    // unstable the unfiltered signal actually is.
    const now = performance.now();
    if (this.lastDetected && !bodyDetected) this.dropouts.push(now);
    this.lastDetected = bodyDetected;
    while (this.dropouts.length && now - this.dropouts[0] > 30_000) this.dropouts.shift();

    const presence = this.presence.update(shoulderWidth);

    return {
      lm,
      presence,
      arrived: this.presence.takeArrival(),
      departed: this.presence.takeDeparture(),
      handsVisible: (left ? 1 : 0) + (right ? 1 : 0),
      shoulderWidth,
      bodyDetected,
      rawDropouts: this.dropouts.length,
      heldMs: this.presence.heldFor(),
      overlay: {
        pose: poseLm ?? null,
        left: left ?? null,
        right: right ?? null,
      },
    };
  }

  stop(): void {
    this.running = false;
    this.dropouts = [];
    this.lastDetected = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.lastTimestamp = -1;
    this.presence.reset();
  }

  close(): void {
    this.stop();
    this.pose?.close();
    this.hands?.close();
    this.pose = undefined;
    this.hands = undefined;
  }
}

export { HAND_POINTS };
export type { PresenceState };
