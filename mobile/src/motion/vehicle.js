import { rotate, multiply } from './vqf.js';

export const GRAVITY = 9.80665;
export const CHANNELS = ['accel_x', 'accel_y', 'accel_z', 'speed'];
export const MOUNTS = {
  upright: { label: 'Top up', matrix: [0, 0, -1, -1, 0, 0, 0, 1, 0] },
  left: { label: 'Top to car left', matrix: [0, 0, -1, 0, 1, 0, 1, 0, 0] },
  right: { label: 'Top to car right', matrix: [0, 0, -1, 0, -1, 0, -1, 0, 0] },
};

export function matrixRotate(m, v) {
  return [0, 1, 2].map((r) => m[3*r]*v[0] + m[3*r+1]*v[1] + m[3*r+2]*v[2]);
}

// Corrections are right-hand rotations around vehicle X, Y, Z, in that order.
// The same proper rotation is used for every acceleration and gyro sample.
export function mountMatrix(preset = 'upright', roll = 0, pitch = 0, yaw = 0) {
  if (!MOUNTS[preset] || ![roll, pitch, yaw].every(Number.isFinite)) {
    throw new Error('Select a mount and enter finite rotation angles.');
  }
  const quats = [roll, pitch, yaw].map((degrees, axis) => {
    const angle = degrees * Math.PI / 360;
    const q = [Math.cos(angle), 0, 0, 0];
    q[axis+1] = Math.sin(angle);
    return q;
  });
  const correction = multiply(quats[2], multiply(quats[1], quats[0]));
  return rotateMatrix(correction, MOUNTS[preset].matrix);
}

export function rotateMatrix(q, matrix) {
  const columns = [0, 1, 2].map((i) => rotate(q, [matrix[i], matrix[i+3], matrix[i+6]]));
  return [0, 1, 2].flatMap((r) => columns.map((c) => c[r]));
}

export function accelerationSI(event, platform) {
  // Expo passes through Core Motion's negative-gravity convention on iOS;
  // Android TYPE_ACCELEROMETER uses positive specific force. Gyro needs no flip.
  if (platform !== 'android' && platform !== 'ios') throw new Error('Use an Android or iOS device.');
  const scale = platform === 'ios' ? -GRAVITY : GRAVITY;
  return [event.x, event.y, event.z].map((v) => v * scale);
}
