import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { PresenceTracker } from '@/lib/mediapipe/presence';

const tracker = () => new PresenceTracker({ minShoulderWidth: 0.2, enterMs: 100, exitMs: 500 });

describe('presence during signing', () => {
  it('requires close shoulders to begin arrival', () => {
    const p = tracker();
    assert.equal(p.update(null, 0, 2, true), 'absent');
    assert.equal(p.update(0.1, 100, 2, true), 'absent');
    assert.equal(p.update(0.3, 200), 'arriving');
    assert.equal(p.update(null, 300, 1), 'present');
    assert.equal(p.takeArrival(), true);
    assert.equal(p.takeArrival(), false);
  });

  it('holds presence through shoulder occlusion and narrow detected poses', () => {
    const p = tracker();
    p.update(0.3, 0);
    p.update(0.3, 100);
    assert.equal(p.update(null, 200, 2), 'present');
    assert.equal(p.update(0.05, 2000, 0, true), 'present');
    assert.equal(p.takeDeparture(), false);
  });

  it('recovers from leaving on hands or a detected body', () => {
    for (const [width, hands, body] of [[null, 1, false], [0.1, 0, true]] as const) {
      const p = tracker();
      p.update(0.3, 0);
      p.update(0.3, 100);
      assert.equal(p.update(null, 200), 'leaving');
      assert.equal(p.update(width, 600, hands, body), 'present');
      assert.equal(p.takeDeparture(), false);
    }
  });

  it('still departs after sustained loss of all usable signals', () => {
    const p = tracker();
    p.update(0.3, 0);
    p.update(0.3, 100);
    assert.equal(p.update(null, 200, 0, true), 'leaving');
    assert.equal(p.update(null, 699), 'leaving');
    assert.equal(p.update(null, 700), 'absent');
    assert.equal(p.takeDeparture(), true);
    assert.equal(p.takeDeparture(), false);
  });
});
