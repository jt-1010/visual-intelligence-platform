import { PRESENCE } from './featureSpec';

/**
 * "Is someone standing at the terminal?"
 *
 * Pose detection does double duty here: the same model that gives us shoulder
 * and wrist landmarks for signing also tells us a person is there, so we do not
 * pay for a third model just to notice someone walked up.
 *
 * The two timers are the whole design. Greeting on the first detected frame
 * means greeting everyone who walks past; ending the session on the first
 * missed frame means the terminal gives up every time tracking blinks. Requiring
 * sustained presence to enter and sustained absence to leave makes both stable.
 */
export type PresenceState = 'absent' | 'arriving' | 'present' | 'leaving';

export type PresenceThresholds = {
  /** Normalised shoulder separation. Larger = closer to the camera. */
  minShoulderWidth: number;
  /** Sustained presence required before greeting, in ms. */
  enterMs: number;
  /** Sustained absence required before ending the session, in ms. */
  exitMs: number;
};

export const DEFAULT_THRESHOLDS: PresenceThresholds = {
  minShoulderWidth: PRESENCE.minShoulderWidth,
  enterMs: PRESENCE.enterMs,
  exitMs: PRESENCE.exitMs,
};

export class PresenceTracker {
  private state: PresenceState = 'absent';
  private since = 0;
  private t: PresenceThresholds;

  /**
   * Thresholds are injected rather than imported as constants because the
   * right values depend on the physical setup -- camera height, lens, how far
   * back the counter is. They cannot be derived, only measured, so the UI
   * exposes them as sliders and this class takes whatever it is given.
   */
  constructor(thresholds: PresenceThresholds = DEFAULT_THRESHOLDS) {
    this.t = { ...thresholds };
  }

  setThresholds(next: Partial<PresenceThresholds>) {
    this.t = { ...this.t, ...next };
  }

  get thresholds(): PresenceThresholds {
    return this.t;
  }

  /**
   * @param shoulderWidth normalised distance between shoulders, or null if no
   *   pose was detected this frame. Doubles as a proximity proxy: a person
   *   across the room has narrow shoulders in normalised coordinates.
   */
  update(
    shoulderWidth: number | null,
    now = performance.now(),
    handsVisible = 0,
    bodyDetected = false,
  ): PresenceState {
    const hasShoulders = shoulderWidth !== null && shoulderWidth >= this.t.minShoulderWidth;
    // Require proximity to begin arrival, then tolerate shoulders obscured by signing.
    const seen = hasShoulders || handsVisible > 0 || (bodyDetected && shoulderWidth !== null);

    switch (this.state) {
      case 'absent':
        if (hasShoulders) this.transition('arriving', now);
        break;

      case 'arriving':
        if (!seen) this.transition('absent', now);
        else if (now - this.since >= this.t.enterMs) this.transition('present', now);
        break;

      case 'present':
        if (!seen) this.transition('leaving', now);
        break;

      case 'leaving':
        if (seen) this.transition('present', now);
        else if (now - this.since >= this.t.exitMs) this.transition('absent', now);
        break;
    }

    return this.state;
  }

  /** True on the single frame where a person becomes established. */
  private justArrived = false;
  private justLeft = false;

  private transition(next: PresenceState, now: number) {
    if (next === 'present' && this.state === 'arriving') this.justArrived = true;
    if (next === 'absent' && this.state === 'leaving') this.justLeft = true;
    this.state = next;
    this.since = now;
  }

  /** Consume the arrival edge. Returns true once per arrival. */
  takeArrival(): boolean {
    const v = this.justArrived;
    this.justArrived = false;
    return v;
  }

  /** Consume the departure edge. Returns true once per departure. */
  takeDeparture(): boolean {
    const v = this.justLeft;
    this.justLeft = false;
    return v;
  }

  get current(): PresenceState {
    return this.state;
  }

  /** Milliseconds spent in the current state -- drives the progress readout. */
  heldFor(now = performance.now()): number {
    return this.since === 0 ? 0 : now - this.since;
  }

  reset() {
    this.state = 'absent';
    this.since = 0;
    this.justArrived = false;
    this.justLeft = false;
  }
}
