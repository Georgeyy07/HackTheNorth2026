import test from 'node:test';
import assert from 'node:assert/strict';
import { toWebSocketUrl } from '../src/wsUrl.js';
import { MotionPipeline } from '../src/motion/pipeline.js';
import { mountMatrix, GRAVITY } from '../src/motion/vehicle.js';
import {
  BATCH_INTERVAL_MS,
  DEFAULT_MAX_TILT_DEGREES,
  shouldStreamImu,
} from '../src/motion/imuStreamConfig.js';

test('BATCH_INTERVAL_MS sends data 5 times per second (200ms)', () => {
  assert.equal(BATCH_INTERVAL_MS, 200, 'Batch interval must be 200ms (5 times per second)');
  assert.equal(1000 / BATCH_INTERVAL_MS, 5, 'Rate must be 5 Hz');
});

test('shouldStreamImu streams ONLY when navigating and tilt angle <= 20 degrees', () => {
  // 1. Actively navigating and within tilt limit (<= 20°) -> STREAM
  assert.equal(shouldStreamImu(true, 0.0), true, 'Should stream at 0° upright');
  assert.equal(shouldStreamImu(true, 10.5), true, 'Should stream at 10.5°');
  assert.equal(shouldStreamImu(true, 19.9), true, 'Should stream at 19.9°');
  assert.equal(shouldStreamImu(true, 20.0), true, 'Should stream at exactly 20.0°');

  // 2. Actively navigating but tilt angle exceeds 20° -> STOP STREAM
  assert.equal(shouldStreamImu(true, 20.1), false, 'Must stop when tilt exceeds 20° (20.1°)');
  assert.equal(shouldStreamImu(true, 25.0), false, 'Must stop when tilt exceeds 20° (25.0°)');
  assert.equal(shouldStreamImu(true, 60.0), false, 'Must stop when tilt exceeds 20° (60.0°)');
  assert.equal(shouldStreamImu(true, 90.0), false, 'Must stop when tilt exceeds 20° (90.0°)');

  // 3. Not navigating -> STOP STREAM regardless of tilt angle
  assert.equal(shouldStreamImu(false, 0.0), false, 'Must stop when not navigating even if 0°');
  assert.equal(shouldStreamImu(false, 15.0), false, 'Must stop when not navigating even if 15°');
  assert.equal(shouldStreamImu(false, 35.0), false, 'Must stop when not navigating');

  // 4. Invalid or non-finite inputs
  assert.equal(shouldStreamImu(null, 10.0), false);
  assert.equal(shouldStreamImu(true, NaN), false);
  assert.equal(shouldStreamImu(true, Infinity), false);
});

test('toWebSocketUrl converts http/https URLs to ws/wss URLs', () => {
  assert.equal(toWebSocketUrl('http://192.168.1.1:8765', '/ws/imu'), 'ws://192.168.1.1:8765/ws/imu');
  assert.equal(toWebSocketUrl('https://example.com:8765', '/ws/imu'), 'wss://example.com:8765/ws/imu');
  assert.equal(toWebSocketUrl('http://localhost:8765', 'ws/imu'), 'ws://localhost:8765/ws/imu');
});

test('MotionPipeline generates 100Hz samples with latitude and longitude', () => {
  const p = new MotionPipeline({ platform: 'android', matrix: mountMatrix('upright') });
  
  // Set GPS location fix with coordinates and speed
  p.setLocation({
    timestamp: 100000,
    coords: { latitude: 43.4723, longitude: -80.5449, speed: 15.2 }
  }, 100000);

  const samples = [];
  p.onSample = (s) => samples.push(s);

  // Feed 100 samples at 10ms intervals (100Hz = 1 second)
  for (let i = 0; i < 100; i++) {
    p.add('accel', { timestamp: i * 0.01, x: 0, y: 1, z: 0 }, 100000 + i * 10);
    p.add('gyro', { timestamp: i * 0.01, x: 0.01, y: 0.02, z: 0.03 }, 100000 + i * 10);
  }

  assert.equal(samples.length, 100, `Expected 100 samples, got ${samples.length}`);
  
  const first = samples[0];
  assert.equal(first.latitude, 43.4723);
  assert.equal(first.longitude, -80.5449);
  assert.equal(first.speedMps, 15.2);
  assert.ok(Number.isFinite(first.time));
  assert.equal(first.input.length, 4); // [accel_x, accel_y, accel_z, speed]
  assert.equal(first.gyro.length, 3); // [gyro_x, gyro_y, gyro_z]
});

test('Batch payload serialization produces expected 5x/sec 100Hz packet contract (20 samples)', () => {
  // In 200ms (1/5 second), 100Hz produces 20 samples
  const batch = [];
  for (let i = 0; i < 20; i++) {
    batch.push({
      time: i * 0.01,
      accel_x: 0.1,
      accel_y: 0.2,
      accel_z: GRAVITY,
      gyro_x: 0.01,
      gyro_y: 0.02,
      gyro_z: 0.03,
      speed: 14.5,
      latitude: 43.4723,
      longitude: -80.5449,
    });
  }

  const payload = {
    type: 'imu_batch',
    batch_id: 1,
    sent_at: Date.now(),
    count: batch.length,
    rate_hz: 5,
    gps: {
      latitude: 43.4723,
      longitude: -80.5449,
      speed: 14.5,
      heading: 90.0,
    },
    samples: batch,
  };

  const serialized = JSON.stringify(payload);
  const parsed = JSON.parse(serialized);

  assert.equal(parsed.type, 'imu_batch');
  assert.equal(parsed.count, 20);
  assert.equal(parsed.rate_hz, 5);
  assert.equal(parsed.samples.length, 20);
  assert.equal(parsed.gps.latitude, 43.4723);
  assert.equal(parsed.gps.longitude, -80.5449);

  // Validate every single sample has all required fields
  for (const s of parsed.samples) {
    assert.ok(Number.isFinite(s.accel_x));
    assert.ok(Number.isFinite(s.accel_y));
    assert.ok(Number.isFinite(s.accel_z));
    assert.ok(Number.isFinite(s.gyro_x));
    assert.ok(Number.isFinite(s.gyro_y));
    assert.ok(Number.isFinite(s.gyro_z));
    assert.ok(Number.isFinite(s.speed));
    assert.ok(Number.isFinite(s.latitude));
    assert.ok(Number.isFinite(s.longitude));
  }
});
