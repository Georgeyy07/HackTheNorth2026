export const TILT_THRESHOLD_DEGREES = 20.0;
export const HYSTERESIS_DEGREES = 1.0;

/**
 * Calculates the angular deviation (in degrees) of the phone from "straight up" (vertical portrait).
 * 
 * In standard smartphone coordinate system:
 * - X points right across the screen
 * - Y points up along the screen toward the top edge
 * - Z points forward out of the screen toward the user
 * 
 * Straight up means the top edge (+Y) points vertically toward the sky (opposite gravity).
 * - Android TYPE_ACCELEROMETER measures specific force (+Y ≈ +1g when upright).
 * - iOS CMAccelerometerData uses negative gravity convention (+Y ≈ -1g when upright).
 * 
 * @param {{ x: number, y: number, z: number }} event - Accelerometer reading
 * @param {string} [platform='android'] - Platform identifier ('android' | 'ios')
 * @returns {number} Angle in degrees from straight up [0, 180]
 */
export function calculateTiltAngle(event, platform = 'android') {
  if (!event || typeof event.x !== 'number' || typeof event.y !== 'number' || typeof event.z !== 'number') {
    return 0;
  }

  // Convert sensor reading so that a_up points in the direction of "straight up" (against gravity)
  const scale = platform === 'ios' ? -1 : 1;
  const ax = event.x * scale;
  const ay = event.y * scale;
  const az = event.z * scale;

  const magnitude = Math.sqrt(ax * ax + ay * ay + az * az);
  if (magnitude < 1e-6) {
    return 0;
  }

  // The phone's upright axis is [0, 1, 0] (top of the phone).
  // Dot product with a_up is simply ay.
  const cosTheta = Math.max(-1.0, Math.min(1.0, ay / magnitude));
  return (Math.acos(cosTheta) * 180.0) / Math.PI;
}

/**
 * Exponential moving average for 3D acceleration vector to filter out momentary bumps.
 */
export function smoothVector(prev, current, alpha = 0.25) {
  if (!prev) return { x: current.x, y: current.y, z: current.z };
  return {
    x: prev.x * (1 - alpha) + current.x * alpha,
    y: prev.y * (1 - alpha) + current.y * alpha,
    z: prev.z * (1 - alpha) + current.z * alpha,
  };
}

/**
 * Checks whether an angle is exceeding tilt threshold with optional hysteresis.
 */
export function isExceedingTilt(angle, currentlyTilted, threshold = TILT_THRESHOLD_DEGREES, hysteresis = HYSTERESIS_DEGREES) {
  if (!currentlyTilted) {
    return angle > threshold;
  }
  return angle > (threshold - hysteresis);
}

/**
 * Decides whether tilt warning (red UI + vibration) should trigger.
 * Strictly active ONLY during navigation; otherwise no warning of tilt angle whatsoever.
 */
export function shouldTriggerTiltAlert(isTilted, isNavigating) {
  return Boolean(isTilted && isNavigating);
}

// Alias for backwards compatibility with previous test helper
export const shouldTriggerVibration = shouldTriggerTiltAlert;
