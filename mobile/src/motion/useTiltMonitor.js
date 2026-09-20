import { useEffect, useRef, useState, useCallback } from 'react';
import { Platform, Vibration } from 'react-native';
import { Accelerometer } from 'expo-sensors';
import {
  calculateTiltAngle,
  smoothVector,
  isExceedingTilt,
  shouldTriggerTiltAlert,
  TILT_THRESHOLD_DEGREES,
} from './tilt';

/**
 * Custom React hook that monitors phone tilt relative to vertical.
 * 
 * Strict behavior:
 * - NO tilt warning or vibration whatsoever unless the user is actively navigating.
 * - During active navigation, if the phone tilt exceeds threshold (20°),
 *   it triggers vibration and turns the UI red.
 * - Does not cancel or interfere with vibrations from other sources.
 */
export function useTiltMonitor({
  threshold = TILT_THRESHOLD_DEGREES,
  updateIntervalMs = 50,
  smoothingAlpha = 0.25,
  enabled = true,
  isNavigating = false,
} = {}) {
  const [tiltAngle, setTiltAngle] = useState(0);
  const [isTilted, setIsTilted] = useState(false);
  const [sensorAvailable, setSensorAvailable] = useState(true);
  const [simulatedAngle, setSimulatedAngle] = useState(null);

  const smoothedReading = useRef(null);
  const isTiltedRef = useRef(false);
  const vibrationTimer = useRef(null);

  // Stop tilt vibration only if our timer was running (does not disrupt other vibrations)
  const stopTiltVibration = useCallback(() => {
    if (vibrationTimer.current) {
      clearInterval(vibrationTimer.current);
      vibrationTimer.current = null;
      Vibration.cancel();
    }
  }, []);

  const startTiltVibration = useCallback(() => {
    if (!vibrationTimer.current) {
      Vibration.vibrate(400);
      vibrationTimer.current = setInterval(() => {
        Vibration.vibrate(400);
      }, 800);
    }
  }, []);

  // Update tilt angle and raw tilt detection
  const updateTiltState = useCallback((angle) => {
    setTiltAngle(angle);
    const nextTilted = isExceedingTilt(angle, isTiltedRef.current, threshold);

    if (nextTilted !== isTiltedRef.current) {
      isTiltedRef.current = nextTilted;
      setIsTilted(nextTilted);
    }
  }, [threshold]);

  const isWarningActive = shouldTriggerTiltAlert(isTilted, isNavigating);

  // Handle vibration strictly during active navigation when tilted
  useEffect(() => {
    if (isWarningActive) {
      startTiltVibration();
    } else {
      stopTiltVibration();
    }
  }, [isWarningActive, startTiltVibration, stopTiltVibration]);

  // Support manual simulated angle (for simulator / testing / web)
  useEffect(() => {
    if (simulatedAngle !== null) {
      updateTiltState(simulatedAngle);
    }
  }, [simulatedAngle, updateTiltState]);

  useEffect(() => {
    if (!enabled || simulatedAngle !== null) return;

    let subscription = null;
    let isSubscribed = true;

    async function initSensor() {
      try {
        const available = await Accelerometer.isAvailableAsync();
        if (!isSubscribed) return;
        setSensorAvailable(available);

        if (!available) {
          return;
        }

        Accelerometer.setUpdateInterval(updateIntervalMs);
        subscription = Accelerometer.addListener((reading) => {
          if (!isSubscribed) return;
          smoothedReading.current = smoothVector(smoothedReading.current, reading, smoothingAlpha);
          const angle = calculateTiltAngle(smoothedReading.current, Platform.OS);
          updateTiltState(angle);
        });
      } catch {
        if (!isSubscribed) return;
        setSensorAvailable(false);
      }
    }

    initSensor();

    return () => {
      isSubscribed = false;
      if (subscription) {
        subscription.remove();
      }
      stopTiltVibration();
    };
  }, [enabled, updateIntervalMs, smoothingAlpha, simulatedAngle, updateTiltState, stopTiltVibration]);

  // Cleanup vibration only on unmount
  useEffect(() => {
    return () => {
      stopTiltVibration();
    };
  }, [stopTiltVibration]);

  return {
    tiltAngle: simulatedAngle !== null ? simulatedAngle : tiltAngle,
    isTilted: isWarningActive, // Warning is ONLY active during navigation
    rawIsTilted: isTilted,
    isWarningActive,
    threshold,
    sensorAvailable,
    simulatedAngle,
    setSimulatedAngle,
    isNavigating,
  };
}
