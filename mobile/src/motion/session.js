import { MotionPipeline } from './pipeline.js';
import { accelerationSI } from './vehicle.js';

/** Append-only JSONL: original events and fused samples are separate records. */
export class RecordingSession {
  constructor({ sink, platform, matrix, now = Date.now() }) {
    this.sink = sink;
    this.platform = platform;
    this.closed = false;
    this.counts = { accelerometer: 0, gyroscope: 0, location: 0, stabilized: 0 };
    this.pipeline = new MotionPipeline({ platform, matrix, onSample: (sample) => {
      this.write('stabilized', sample);
    } });
    const { samples, ...contract } = this.pipeline.snapshot();
    this.write('header', {
      startedAt: now, platform, ...contract,
      raw: 'Original Expo events; accelerometer in g, gyro in rad/s, GPS speed in m/s. accel.si is positive-specific-force device XYZ in m/s².',
      stabilized: 'earthAccel includes gravity; earthGyro is rad/s. Both use quaternion device-to-earth, Z up, arbitrary yaw; NOT geographic or vehicle XY.',
      model: 'input is [vehicle accel_x, accel_y, accel_z, speed], with gravity; gyro uses the same fixed mount. Missing speed is null with mask=false.',
      timestamps: 'raw event.timestamp and stabilized.time are sensor monotonic seconds; location.timestamp and availableAt/receivedAt are Unix milliseconds. Speed is held at availability, not source-time interpolated.',
      speed: 'When the GPS fix reports no speed field, it is derived from the haversine distance between consecutive fixes divided by their time gap; stabilized.speedDerived flags this (true = derived, false = reported by the device, null = no fresh fix).',
    });
  }

  write(type, data) {
    if (this.closed) throw new Error('Recording is closed.');
    this.sink.append(JSON.stringify({ type, ...data }) + '\n');
    if (type in this.counts) this.counts[type]++;
  }

  sensor(kind, event, now = Date.now()) {
    const valid = [event.timestamp, event.x, event.y, event.z].every(Number.isFinite) && event.timestamp >= 0;
    this.write(kind === 'accel' ? 'accelerometer' : 'gyroscope', {
      receivedAt: now, event, valid,
      ...(kind === 'accel' && valid ? { si: accelerationSI(event, this.platform) } : {}),
    });
    this.pipeline.add(kind, event, now);
  }

  location(position, now = Date.now()) {
    this.write('location', { receivedAt: now, event: position });
    this.pipeline.setLocation(position, now);
  }

  calibrate(now = Date.now()) {
    this.pipeline.calibrateParked(now);
    this.write('mount', { receivedAt: now, segment: this.pipeline.segment, deviceToVehicle: this.pipeline.matrix });
  }

  close(reason = 'stopped', now = Date.now()) {
    if (this.closed) return;
    try {
      this.write('end', { endedAt: now, reason, counts: this.counts, rejectedFusionEvents: this.pipeline.rejected });
      this.sink.flush();
    } finally {
      this.closed = true;
      this.sink.close();
    }
  }
}
