import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BasicVQF, conjugate, norm, rotate } from '../src/motion/vqf.js';
import { accelerationSI, GRAVITY, matrixRotate, mountMatrix } from '../src/motion/vehicle.js';
import { MotionPipeline } from '../src/motion/pipeline.js';
import { RecordingSession } from '../src/motion/session.js';

const near = (a, b, tolerance = 1e-8) => {
  assert.equal(a.length, b.length);
  a.forEach((v, i) => assert.ok(Math.abs(v-b[i]) < tolerance, `${a} != ${b}`));
};
const event = (time, xyz) => ({ timestamp: time, x: xyz[0], y: xyz[1], z: xyz[2] });
const feed = (pipeline, i, a = [0, 1, 0], g = [0, 0, 0], now = 100000+i*10) => {
  pipeline.add('accel', event(i/100, a), now);
  pipeline.add('gyro', event(i/100, g), now);
};

test('upright signs and SI conversion exactly meet the four-feature contract', () => {
  near(matrixRotate(mountMatrix(), [1, 2, 3]), [-3, -1, 2]);
  near(accelerationSI(event(0, [0, 1, 0]), 'android'), [0, GRAVITY, 0]);
  near(accelerationSI(event(0, [0, -1, 0]), 'ios'), [0, GRAVITY, 0]);
  const p = new MotionPipeline();
  p.setLocation({ timestamp: 100000, coords: { speed: 12.5 } }, 100000);
  feed(p, 0);
  near(p.latest.input, [0, 0, GRAVITY, 12.5]);
  assert.deepEqual(p.latest.mask, [true, true, true, true]);
  near(p.latest.earthAccel, [0, 0, GRAVITY]);
  assert.ok(Math.abs(p.latest.verticalLinear) < 1e-8);
});

test('sideways mounts and arbitrary corrections rotate all three axes without scaling', () => {
  near(matrixRotate(mountMatrix('left'), [GRAVITY, 0, 0]), [0, 0, GRAVITY]);
  near(matrixRotate(mountMatrix('right'), [-GRAVITY, 0, 0]), [0, 0, GRAVITY]);
  const m = mountMatrix('left', 21, -32, 18);
  const rows = [m.slice(0, 3), m.slice(3, 6), m.slice(6, 9)];
  rows.forEach((row) => assert.ok(Math.abs(norm(row)-1) < 1e-10));
  near(matrixRotate(m, rows[0]), [1, 0, 0]);
  const p = new MotionPipeline({ matrix: m });
  feed(p, 0, [1, 2, 3], [1, 2, 3]);
  near(p.latest.input.slice(0, 3).map((v) => v/GRAVITY), p.latest.gyro);
  assert.throws(() => mountMatrix('upright', NaN));
});

test('JS BasicVQF agrees with upstream C++ over initialization, motion, and reset', () => {
  const reference = new Map(JSON.parse(readFileSync(new URL('./reference/vqf.json', import.meta.url))).map((r) => [r.i, r.q]));
  const filter = new BasicVQF();
  for (let i = 0; i < 4000; i++) {
    if (i === 2000) filter.reset();
    const t = i*0.01;
    const q = filter.update([0.1*Math.sin(t), 0.07*Math.cos(0.3*t), 0.2],
      [0.4*Math.sin(2*t), GRAVITY+0.3*Math.cos(t), 0.6*Math.sin(t)]);
    if (reference.has(i)) near(q, reference.get(i), 1e-9);
  }
});

test('VQF tracks tilt without flattening a vertical bump, including inverted starts', () => {
  for (const a of [[0, GRAVITY, 0], [0, 0, -GRAVITY], [3, 4, Math.sqrt(GRAVITY**2-25)]]) {
    const filter = new BasicVQF();
    let q;
    for (let i = 0; i < 400; i++) q = filter.update([0, 0, 0], a);
    near(rotate(q, a), [0, 0, GRAVITY]);
    const bump = a.map((v) => v*1.5);
    q = filter.update([0, 0, 0], bump);
    near(rotate(q, bump), [0, 0, GRAVITY*1.5]);
    near(rotate(conjugate(q), [0, 0, GRAVITY]), a);
  }
  assert.throws(() => new BasicVQF(0));
  assert.throws(() => new BasicVQF().update([0, 0, NaN], [0, 1, 0]));
});

test('unaligned sensors interpolate to 100 Hz; invalid and duplicate timestamps are rejected', () => {
  const samples = [];
  const p = new MotionPipeline({ onSample: (s) => samples.push(s) });
  p.add('accel', event(1, [0, 1, 0]), 100000);
  p.add('gyro', event(1.005, [0, 0, 0]), 100005);
  assert.equal(samples.length, 0);
  p.add('accel', event(1.02, [0, 1.2, 0]), 100020);
  assert.equal(samples.length, 1);
  near(samples[0].input.slice(0, 3), [0, 0, 1.05*GRAVITY]);
  p.add('gyro', event(1.025, [0, 0, 0]), 100025);
  assert.equal(samples.length, 2);
  assert.ok(Math.abs(samples[1].time-samples[0].time-0.01) < 1e-8);
  p.add('gyro', event(1.025, [0, 0, 0]));
  p.add('gyro', event(1.02, [0, 0, 0]));
  p.add('accel', event(1.04, [NaN, 0, 0]));
  assert.equal(p.rejected, 3);
});

test('turning the car changes earth heading but never rotates vehicle model axes', () => {
  const p = new MotionPipeline();
  for (let i = 0; i < 500; i++) feed(p, i, [0, 1, -0.1], [0, 0.2, 0]);
  near(p.latest.input.slice(0, 3), [0.1*GRAVITY, 0, GRAVITY]);
  near(p.latest.gyro, [0, 0, 0.2]);
  assert.ok(Math.abs(p.latest.quaternion[3]) > 0.01);
});

test('sensor gaps reset fusion and never fill an outage with repeated data', () => {
  const samples = [];
  const p = new MotionPipeline({ onSample: (s) => samples.push(s) });
  feed(p, 0); feed(p, 1); feed(p, 100);
  assert.deepEqual(samples.map((s) => s.time), [0, 0.01, 1]);
  assert.ok(samples[2].segment > samples[1].segment);
  assert.equal(p.samples.length, 1);
  for (let i = 101; i < 1000; i++) p.add('accel', event(i/100, [0, 1, 0]));
  assert.ok(p.queues.accel.length <= 128);
});

test('speed stays in m/s; stale, invalid, and future fixes never become measured zeros', () => {
  const p = new MotionPipeline();
  feed(p, 0);
  assert.equal(p.latest.input[3], null);
  assert.equal(p.latest.mask[3], false);
  p.setLocation({ timestamp: 100000, coords: { speed: 7 } }, 100000);
  assert.equal(p.speedAt(103000), 7);
  assert.equal(p.speedAt(103001), null);
  assert.equal(p.speedAt(99999), null);
  p.setLocation({ timestamp: 100010, coords: { speed: -1 } }, 100010);
  assert.equal(p.speedAt(100010), null);
  p.setLocation({ timestamp: 100020, coords: { speed: 0 } }, 100020);
  assert.equal(p.speedAt(100020), 0);
});

test('parked VQF calibration corrects a tilted mount and freezes the body-frame transform', () => {
  const tilt = 25*Math.PI/180;
  const a = [0, Math.cos(tilt), Math.sin(tilt)];
  const p = new MotionPipeline();
  assert.throws(() => p.calibrateParked());
  for (let i = 0; i < 401; i++) {
    if (i % 100 === 0) p.setLocation({ timestamp: 100000+i*10, coords: { speed: 0 } }, 100000+i*10);
    feed(p, i, a);
  }
  assert.equal(p.latest.canCalibrate, true);
  p.calibrateParked(104000);
  const matrix = [...p.matrix];
  feed(p, 401, a);
  near(p.latest.input.slice(0, 3), [0, 0, GRAVITY]);
  // A real car pitch after calibration must remain in vehicle acceleration.
  for (let i = 402; i < 1000; i++) feed(p, i, [0, 1, 0]);
  assert.deepEqual(p.matrix, matrix);
  assert.ok(Math.abs(p.latest.input[0]) > 1);
  assert.throws(() => p.calibrateParked(200000));
});

test('raw IMU/GPS and stabilized samples persist independently beyond the preview limit', () => {
  const lines = [];
  let closed = 0;
  const session = new RecordingSession({ platform: 'android', sink: {
    append: (line) => lines.push(line), flush() {}, close: () => closed++,
  }, now: 100000 });
  session.location({ timestamp: 100000, coords: { speed: 3 } }, 100000);
  for (let i = 0; i < 1100; i++) {
    session.sensor('accel', event(i/100, [0, 1, 0]), 100000+i*10);
    session.sensor('gyro', event(i/100, [0, 0, 0]), 100000+i*10);
  }
  session.sensor('accel', event(12, [NaN, 0, 0]), 112000);
  assert.equal(session.pipeline.samples.length, 1024);
  session.close(); session.close();
  const records = lines.map(JSON.parse);
  assert.equal(closed, 1);
  assert.equal(records.filter((r) => r.type === 'stabilized').length, 1100);
  assert.equal(records.filter((r) => r.type === 'accelerometer').length, 1101);
  assert.equal(records.at(-2).valid, false);
  assert.equal(records.at(-1).type, 'end');
  const raw = records.find((r) => r.type === 'accelerometer');
  assert.equal(raw.event.y, 1);
  assert.equal(raw.si[1], GRAVITY);
  const fused = records.find((r) => r.type === 'stabilized');
  near(fused.earthAccel, [0, 0, GRAVITY]);
  near(fused.earthGyro, [0, 0, 0]);
  assert.equal(fused.speedMps, 3);
  near(fused.input, [0, 0, GRAVITY, 3]);
  assert.throws(() => session.sensor('accel', event(13, [0, 1, 0])));
});

test('session calibration is recorded, and write failures propagate while handles close', () => {
  const records = [];
  let closed = false;
  let fail = false;
  const session = new RecordingSession({ platform: 'android', sink: {
    append(line) { if (fail) throw new Error('disk full'); records.push(JSON.parse(line)); },
    flush() {}, close() { closed = true; },
  } });
  for (let i = 0; i < 401; i++) {
    if (i % 100 === 0) session.location({ timestamp: 100000+i*10, coords: { speed: 0 } }, 100000+i*10);
    session.sensor('accel', event(i/100, [0, 1, 0]), 100000+i*10);
    session.sensor('gyro', event(i/100, [0, 0, 0]), 100000+i*10);
  }
  session.calibrate(104000);
  assert.equal(records.at(-1).type, 'mount');
  assert.deepEqual(records.at(-1).deviceToVehicle, session.pipeline.matrix);
  assert.equal(records.at(-1).segment, 1);
  fail = true;
  assert.throws(() => session.close(), /disk full/);
  assert.equal(closed, true);
  assert.equal(session.closed, true);
});
