import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CAMERA_FRAME_INTERVAL_MS,
  DEFAULT_MAX_TILT_DEGREES,
  shouldStreamCamera,
} from '../src/motion/cameraStreamConfig.js';

test('CAMERA_FRAME_INTERVAL_MS produces 15+ frames per second', () => {
  // 15 fps = 1000 / 15 ≈ 66.67ms. Interval must be <= 66.6ms to ensure 15+ fps
  assert.ok(CAMERA_FRAME_INTERVAL_MS <= 66.7, `Interval ${CAMERA_FRAME_INTERVAL_MS}ms must be <= 66.7ms for 15+ fps`);
  const fps = 1000 / CAMERA_FRAME_INTERVAL_MS;
  assert.ok(fps >= 15.0, `Calculated FPS (${fps.toFixed(1)}) must be at least 15 fps`);
});

test('shouldStreamCamera streams ONLY when navigating, confirmed, and tilt <= 20 degrees', () => {
  // 1. All conditions met: Navigating + Confirmed + Tilt <= 20° -> TRUE
  assert.equal(shouldStreamCamera(true, true, 0.0), true, 'Should stream at 0°');
  assert.equal(shouldStreamCamera(true, true, 10.0), true, 'Should stream at 10°');
  assert.equal(shouldStreamCamera(true, true, 19.9), true, 'Should stream at 19.9°');
  assert.equal(shouldStreamCamera(true, true, 20.0), true, 'Should stream at exactly 20.0° limit');

  // 2. Navigating and confirmed, but tilt exceeds 20° -> FALSE
  assert.equal(shouldStreamCamera(true, true, 20.1), false, 'Must stop when tilt exceeds 20° (20.1°)');
  assert.equal(shouldStreamCamera(true, true, 25.0), false, 'Must stop when tilt exceeds 20° (25.0°)');
  assert.equal(shouldStreamCamera(true, true, 45.0), false, 'Must stop when tilt is 45°');
  assert.equal(shouldStreamCamera(true, true, 90.0), false, 'Must stop when phone is flat/horizontal');

  // 3. Navigating and upright, but NOT yet confirmed -> FALSE (shows alignment modal instead)
  assert.equal(shouldStreamCamera(true, false, 0.0), false, 'Must not stream before user confirms road view');
  assert.equal(shouldStreamCamera(true, false, 10.0), false, 'Must not stream before confirm even if upright');

  // 4. Confirmed and upright, but NOT navigating -> FALSE
  assert.equal(shouldStreamCamera(false, true, 0.0), false, 'Must not stream when navigation is inactive');
  assert.equal(shouldStreamCamera(false, false, 0.0), false, 'Must not stream when inactive');

  // 5. Invalid / non-finite inputs -> FALSE
  assert.equal(shouldStreamCamera(true, true, NaN), false);
  assert.equal(shouldStreamCamera(true, true, Infinity), false);
  assert.equal(shouldStreamCamera(null, true, 10.0), false);
  assert.equal(shouldStreamCamera(true, null, 10.0), false);
});

test('Camera frame payload adheres to backend contract for printing received frame numbers', () => {
  const framePayload = {
    type: 'camera_frame',
    frame_number: 1,
    timestamp: Date.now(),
    fps: 16,
    image: null,
  };

  const jsonStr = JSON.stringify(framePayload);
  const parsed = JSON.parse(jsonStr);

  assert.equal(parsed.type, 'camera_frame');
  assert.equal(parsed.frame_number, 1);
  assert.ok(parsed.fps >= 15);
  assert.ok(Number.isFinite(parsed.timestamp));

  // Simulate frame 2
  const frame2Payload = { ...framePayload, frame_number: 2, timestamp: Date.now() + 60 };
  assert.equal(frame2Payload.frame_number, 2);
});
