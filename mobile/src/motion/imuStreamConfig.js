// 5 times per second (1000ms / 5 = 200ms)
export const BATCH_INTERVAL_MS = 200;
export const DEFAULT_MAX_TILT_DEGREES = 20.0;

/**
 * Determines whether IMU data should stream over WebSockets:
 * Must be navigating, and phone tilt angle must NOT exceed 20 degrees.
 */
export function shouldStreamImu(isNavigating, tiltAngle, maxTiltAngle = DEFAULT_MAX_TILT_DEGREES) {
  return Boolean(isNavigating && Number.isFinite(tiltAngle) && tiltAngle <= maxTiltAngle);
}
