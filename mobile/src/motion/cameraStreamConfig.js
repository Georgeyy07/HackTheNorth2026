// Target 15+ fps (1000ms / 16ms ≈ 16 fps, or 60ms interval ≈ 16.6 fps)
export const CAMERA_FRAME_INTERVAL_MS = 60;
export const DEFAULT_MAX_TILT_DEGREES = 20.0;

export function shouldStreamCamera(isNavigating, isConfirmed, tiltAngle, maxTiltAngle = DEFAULT_MAX_TILT_DEGREES) {
  return Boolean(
    isNavigating &&
    isConfirmed &&
    Number.isFinite(tiltAngle) &&
    tiltAngle <= maxTiltAngle
  );
}
