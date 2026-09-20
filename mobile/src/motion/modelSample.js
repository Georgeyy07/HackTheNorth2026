import { rotate } from './vqf.js';

// Match the recorded sessions' vqf_vehicle preprocessing. Preserve gravity and SI units.
export function modelSample(sample, matrix, gpsReceivedAt) {
  const forward = rotate(sample.quaternion, matrix.slice(0, 3));
  const length = Math.hypot(forward[0], forward[1]);
  const valid = length > 1e-6 && sample.mask.slice(0, 3).every(Boolean);
  const fx = forward[0] / length, fy = forward[1] / length;
  const project = ([x, y, z]) => [fx*x + fy*y, -fy*x + fx*y, z];
  const accel = valid ? project(sample.earthAccel) : [null, null, null];
  const gyro = valid ? project(sample.earthGyro) : [null, null, null];
  return {
    time: sample.time, available_at_ms: sample.availableAt, segment: sample.segment,
    accel_x: accel[0], accel_y: accel[1], accel_z: accel[2],
    gyro_x: gyro[0], gyro_y: gyro[1], gyro_z: gyro[2],
    speed: sample.mask[3] ? sample.speedMps : null,
    latitude: sample.latitude, longitude: sample.longitude,
    gps_timestamp_ms: sample.gpsTimestamp, gps_received_at_ms: gpsReceivedAt ?? null,
    speed_timestamp_ms: sample.speedTimestamp, settling: sample.settling,
  };
}

export const inferenceHandshake = (client) => ({
  type: 'handshake', client, protocol: 'roughroute.imu.v1',
  sample_rate_hz: 100, acceleration_frame: 'vqf_vehicle',
});
