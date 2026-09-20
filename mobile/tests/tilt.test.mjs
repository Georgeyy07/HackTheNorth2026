import test from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateTiltAngle,
  smoothVector,
  shouldTriggerTiltAlert,
  shouldTriggerVibration,
  TILT_THRESHOLD_DEGREES,
} from '../src/motion/tilt.js';

const closeTo = (actual, expected, tolerance = 1e-4) => {
  assert.ok(
    Math.abs(actual - expected) < tolerance,
    `Expected ${actual} to be close to ${expected} (diff: ${Math.abs(actual - expected)})`
  );
};

test('straight up (0 degrees) on both Android and iOS', () => {
  // Android upright: y is +1.0g
  const androidUpright = calculateTiltAngle({ x: 0, y: 1, z: 0 }, 'android');
  closeTo(androidUpright, 0.0);

  // iOS upright: y is -1.0g (negative gravity convention)
  const iosUpright = calculateTiltAngle({ x: 0, y: -1, z: 0 }, 'ios');
  closeTo(iosUpright, 0.0);
});

test('pitch forward and backward calculation', () => {
  const angleRad = (30.0 * Math.PI) / 180.0;
  const cos30 = Math.cos(angleRad);
  const sin30 = Math.sin(angleRad);

  // Pitched forward 30 degrees on Android
  const androidPitchForward = calculateTiltAngle({ x: 0, y: cos30, z: sin30 }, 'android');
  closeTo(androidPitchForward, 30.0);

  // Pitched backward 30 degrees on Android
  const androidPitchBackward = calculateTiltAngle({ x: 0, y: cos30, z: -sin30 }, 'android');
  closeTo(androidPitchBackward, 30.0);

  // Pitched forward 30 degrees on iOS
  const iosPitchForward = calculateTiltAngle({ x: 0, y: -cos30, z: -sin30 }, 'ios');
  closeTo(iosPitchForward, 30.0);
});

test('roll left and right calculation', () => {
  const angleRad = (25.0 * Math.PI) / 180.0;
  const cos25 = Math.cos(angleRad);
  const sin25 = Math.sin(angleRad);

  // Rolled right 25 degrees on Android
  const rollRight = calculateTiltAngle({ x: sin25, y: cos25, z: 0 }, 'android');
  closeTo(rollRight, 25.0);

  // Rolled left 25 degrees on Android
  const rollLeft = calculateTiltAngle({ x: -sin25, y: cos25, z: 0 }, 'android');
  closeTo(rollLeft, 25.0);
});

test('20 degree threshold boundary', () => {
  assert.equal(TILT_THRESHOLD_DEGREES, 20.0);

  const rad19 = (19.9 * Math.PI) / 180.0;
  const angle19 = calculateTiltAngle({ x: Math.sin(rad19), y: Math.cos(rad19), z: 0 }, 'android');
  assert.ok(angle19 < 20.0);

  const rad20 = (20.0 * Math.PI) / 180.0;
  const angle20 = calculateTiltAngle({ x: Math.sin(rad20), y: Math.cos(rad20), z: 0 }, 'android');
  closeTo(angle20, 20.0);

  const rad20_1 = (20.1 * Math.PI) / 180.0;
  const angle20_1 = calculateTiltAngle({ x: Math.sin(rad20_1), y: Math.cos(rad20_1), z: 0 }, 'android');
  assert.ok(angle20_1 > 20.0);
});

test('flat on table (90 degrees) and upside down (180 degrees)', () => {
  // Resting flat face up on Android: z = 1
  const flatAndroid = calculateTiltAngle({ x: 0, y: 0, z: 1 }, 'android');
  closeTo(flatAndroid, 90.0);

  // Resting flat face up on iOS: z = -1
  const flatIos = calculateTiltAngle({ x: 0, y: 0, z: -1 }, 'ios');
  closeTo(flatIos, 90.0);

  // Upside down portrait on Android: y = -1
  const upsideDownAndroid = calculateTiltAngle({ x: 0, y: -1, z: 0 }, 'android');
  closeTo(upsideDownAndroid, 180.0);

  // Upside down portrait on iOS: y = 1
  const upsideDownIos = calculateTiltAngle({ x: 0, y: 1, z: 0 }, 'ios');
  closeTo(upsideDownIos, 180.0);
});

test('vector smoothing applies exponential moving average', () => {
  const v1 = { x: 0, y: 1, z: 0 };
  const v2 = { x: 1, y: 0, z: 1 };

  // First sample sets baseline
  const s0 = smoothVector(null, v1, 0.5);
  assert.deepEqual(s0, v1);

  // Second sample averages at alpha = 0.5
  const s1 = smoothVector(s0, v2, 0.5);
  closeTo(s1.x, 0.5);
  closeTo(s1.y, 0.5);
  closeTo(s1.z, 0.5);
});

test('handles zero or invalid readings safely', () => {
  assert.equal(calculateTiltAngle(null), 0);
  assert.equal(calculateTiltAngle({ x: 0, y: 0, z: 0 }), 0);
  assert.equal(calculateTiltAngle({ x: 'bad', y: 0, z: 0 }), 0);
});

test('tilt warning and vibration strictly require active navigation', () => {
  // When NOT navigating (typing address, browsing): NO warning, NO red UI, NO vibration whatsoever
  assert.equal(shouldTriggerTiltAlert(true, false), false);
  assert.equal(shouldTriggerTiltAlert(false, false), false);

  // When actively navigating:
  // Upright (<=20°) -> NO warning
  assert.equal(shouldTriggerTiltAlert(false, true), false);

  // Tilted (>20°) in mount while navigating -> Trigger red alert & vibration!
  assert.equal(shouldTriggerTiltAlert(true, true), true);
});
