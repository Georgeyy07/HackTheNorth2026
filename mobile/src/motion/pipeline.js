import { BasicVQF, conjugate, inclinationQuaternion, norm, rotate } from './vqf.js';
import { accelerationSI, CHANNELS, GRAVITY, matrixRotate, mountMatrix, rotateMatrix } from './vehicle.js';
import { haversineDistanceM } from '../navigation.js';

const PERIOD = 0.01;
const MAX_GAP = 0.05;
// GPS fixes routinely arrive every 2-5s in real conditions (not the
// requested 1s interval -- that's a request, not a guarantee), so a 3s
// staleness window discarded the majority of samples between fixes even
// with good signal. Road speed doesn't change fast enough for a
// slightly-older fix to be meaningfully wrong.
const SPEED_MAX_AGE_MS = 6000;
// Below this, GPS position jitter (a few meters of fix noise) dominates the
// distance/time derivation and produces spurious speed spikes.
const MIN_DERIVE_DT_S = 0.5;
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
    this.lastPosition = null;
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
    const { speed, latitude, longitude } = position.coords;
    const time = position.timestamp;
    const fresh = Number.isFinite(time) && time <= now + 1000 && now-time <= SPEED_MAX_AGE_MS;
    let resolved = Number.isFinite(speed) && speed >= 0 ? speed : null;
    let derived = false;
    // The device speed field is frequently unavailable (many chipsets only report
    // it above some threshold, or not at all); fall back to distance/time between
    // consecutive fixes rather than losing the channel entirely.
    if (resolved === null && fresh && this.lastPosition
        && Number.isFinite(latitude) && Number.isFinite(longitude)) {
      const dtS = (time-this.lastPosition.time)/1000;
      if (dtS >= MIN_DERIVE_DT_S && dtS*1000 <= SPEED_MAX_AGE_MS) {
        const distanceM = haversineDistanceM(
          this.lastPosition.latitude, this.lastPosition.longitude, latitude, longitude);
        resolved = distanceM/dtS;
        derived = true;
      }
    }
    if (Number.isFinite(latitude) && Number.isFinite(longitude) && Number.isFinite(time)) {
      this.lastPosition = { latitude, longitude, time };
    }
    // Invalid speed clears the previous hold; never turn missing GPS into 0.
    this.speedFix = resolved !== null && fresh ? { speed: resolved, time, derived } : null;
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
        speedDerived: speed === null ? null : this.speedFix.derived,
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
      speedAlignment: 'latest fresh GPS fix at sample availability (maximum age 6 s)',
      samples: [...this.samples],
    };
  }
}
