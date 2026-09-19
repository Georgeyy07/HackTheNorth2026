import { BasicVQF, conjugate, inclinationQuaternion, norm, rotate } from './vqf.js';
import { accelerationSI, CHANNELS, GRAVITY, matrixRotate, mountMatrix, rotateMatrix } from './vehicle.js';

const PERIOD = 0.01;
const MAX_GAP = 0.05;
const SPEED_MAX_AGE_MS = 3000;
const EPS = 1e-8;

function at(queue, time) {
  while (queue.length > 1 && queue[1].time <= time + EPS) queue.shift();
  const a = queue[0];
  if (Math.abs(a.time-time) < EPS) return a.values;
  const b = queue[1];
  if (!b || a.time > time || b.time-a.time > MAX_GAP + EPS) return null;
  const weight = (time-a.time)/(b.time-a.time);
  return a.values.map((v, i) => v + weight*(b.values[i]-v));
}

/** Source-time 100 Hz alignment. No extrapolation or integration across gaps. */
export class MotionPipeline {
  constructor({ platform = 'android', matrix = mountMatrix(), onSample = () => {} } = {}) {
    this.platform = platform;
    this.matrix = [...matrix];
    this.onSample = onSample;
    this.filter = new BasicVQF(PERIOD);
    this.segment = -1;
    this.rejected = 0;
    this.speedFix = null;
    this.reset();
  }

  reset() {
    this.filter.reset();
    this.queues = { accel: [], gyro: [] };
    this.next = null;
    this.start = null;
    this.stillSince = null;
    this.latest = null;
    this.samples = [];
    this.segment++;
  }

  setLocation(position, now = Date.now()) {
    const speed = position.coords.speed;
    const time = position.timestamp;
    // Invalid speed clears the previous hold; never turn missing GPS into 0.
    this.speedFix = Number.isFinite(speed) && speed >= 0 && Number.isFinite(time)
      && time <= now + 1000 && now-time <= SPEED_MAX_AGE_MS ? { speed, time } : null;
  }

  speedAt(now) {
    const fix = this.speedFix;
    return fix && now >= fix.time && now-fix.time <= SPEED_MAX_AGE_MS ? fix.speed : null;
  }

  add(kind, event, receivedMs = Date.now()) {
    if (!(kind in this.queues)) throw new Error('Unknown sensor.');
    if (![event.timestamp, event.x, event.y, event.z, receivedMs].every(Number.isFinite)
        || event.timestamp < 0) {
      this.rejected++;
      return;
    }
    let queue = this.queues[kind];
    const previous = queue.at(-1);
    if (previous && event.timestamp <= previous.time) {
      this.rejected++;
      return;
    }
    if ((previous && event.timestamp-previous.time > MAX_GAP + EPS) || queue.length >= 128) {
      this.reset();
      queue = this.queues[kind];
    }
    queue.push({ time: event.timestamp, values: kind === 'accel'
      ? accelerationSI(event, this.platform) : [event.x, event.y, event.z] });
    const { accel, gyro } = this.queues;
    if (!accel.length || !gyro.length) return;
    if (this.next === null) {
      this.next = Math.max(accel[0].time, gyro[0].time);
      this.start = this.next;
    }
    const until = Math.min(accel.at(-1).time, gyro.at(-1).time);
    while (this.next <= until + EPS) {
      const a = at(accel, this.next);
      const g = at(gyro, this.next);
      if (!a || !g) break;
      const quaternion = this.filter.update(g, a);
      const vehicle = matrixRotate(this.matrix, a);
      const earth = rotate(quaternion, a);
      const speed = this.speedAt(receivedMs);
      if (norm(g) < 0.035 && Math.abs(norm(a)-GRAVITY) < 0.2 && speed !== null && speed < 0.3) {
        this.stillSince ??= this.next;
      } else {
        this.stillSince = null;
      }
      const sample = {
        time: this.next, availableAt: receivedMs, segment: this.segment,
        input: [...vehicle, speed], mask: [true, true, true, speed !== null],
        speedMps: speed, speedTimestamp: speed === null ? null : this.speedFix.time,
        gyro: matrixRotate(this.matrix, g), quaternion,
        earthAccel: earth, earthGyro: rotate(quaternion, g),
        verticalLinear: earth[2]-GRAVITY,
        settling: this.next-this.start < 10,
        canCalibrate: this.next-this.start >= 3 && this.stillSince !== null && this.next-this.stillSince >= 2,
      };
      this.latest = sample;
      this.samples.push(sample);
      if (this.samples.length > 1024) this.samples.shift();
      this.onSample(sample);
      this.next += PERIOD;
    }
  }

  calibrateParked(now = Date.now()) {
    const sample = this.latest;
    if (!sample?.canCalibrate || now-sample.availableAt > 250 || this.speedAt(now) === null || this.speedAt(now) >= 0.3) {
      throw new Error('Park on level ground and hold the mount still with a fresh GPS fix.');
    }
    const gravityPhone = rotate(conjugate(sample.quaternion), [0, 0, 1]);
    const gravityVehicle = matrixRotate(this.matrix, gravityPhone);
    const correction = inclinationQuaternion(gravityVehicle);
    this.matrix = rotateMatrix(correction, this.matrix);
    // Freeze the mount rotation. Continuously levelling model samples would
    // erase real car pitch/roll and would no longer be a vehicle body frame.
    this.reset();
  }

  snapshot() {
    return {
      schema: 'roughroute.vehicle-imu.v1', channels: CHANNELS,
      units: ['m/s²', 'm/s²', 'm/s²', 'm/s'], accelerationIncludesGravity: true,
      axes: ['forward', 'left', 'up'], sampleRateHz: 100,
      deviceToVehicle: this.matrix, orientationFilter: 'BasicVQF 2.1.2 / 6D',
      speedAlignment: 'latest fresh GPS fix at sample availability (maximum age 3 s)',
      samples: [...this.samples],
    };
  }
}
