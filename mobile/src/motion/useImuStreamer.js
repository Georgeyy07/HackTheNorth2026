import { useCallback, useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import * as Location from 'expo-location';
import { getWebSocketUrl } from '../api';
import { MotionPipeline } from './pipeline';
import { mountMatrix, accelerationSI, matrixRotate } from './vehicle';

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

export function useImuStreamer({
  defaultUrl = null,
  batchIntervalMs = BATCH_INTERVAL_MS,
  isNavigating = null,
  tiltAngle = 0,
  maxTiltAngle = DEFAULT_MAX_TILT_DEGREES,
} = {}) {
  const [status, setStatus] = useState('idle'); // 'idle' | 'connecting' | 'streaming' | 'error'
  const [error, setError] = useState(null);
  const [wsUrl, setWsUrl] = useState(defaultUrl || getWebSocketUrl('/ws/imu'));
  const [stats, setStats] = useState({ packetsSent: 0, samplesSent: 0, lastSentAt: null });
  const [latestSample, setLatestSample] = useState(null);

  const socketRef = useRef(null);
  const bufferRef = useRef([]);
  const timerRef = useRef(null);
  const subscriptionsRef = useRef([]);
  const pipelineRef = useRef(null);
  const mountedRef = useRef(true);
  const activeRef = useRef(false);
  const latestGpsRef = useRef(null);
  const batchSeqRef = useRef(0);

  const flushBuffer = useCallback(() => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (bufferRef.current.length === 0) return;

    const samplesToSend = bufferRef.current;
    bufferRef.current = [];
    batchSeqRef.current += 1;

    const gps = latestGpsRef.current?.coords;
    const payload = {
      type: 'imu_batch',
      batch_id: batchSeqRef.current,
      sent_at: Date.now(),
      count: samplesToSend.length,
      rate_hz: 5, // 5 batches per second
      gps: gps ? {
        latitude: gps.latitude,
        longitude: gps.longitude,
        speed: gps.speed,
        heading: gps.heading,
        altitude: gps.altitude,
      } : null,
      samples: samplesToSend,
    };

    try {
      ws.send(JSON.stringify(payload));
      if (mountedRef.current) {
        setStats((prev) => ({
          packetsSent: prev.packetsSent + 1,
          samplesSent: prev.samplesSent + samplesToSend.length,
          lastSentAt: Date.now(),
        }));
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(`WebSocket send failed: ${err.message}`);
      }
    }
  }, []);

  const stop = useCallback(() => {
    activeRef.current = false;
    subscriptionsRef.current.forEach((sub) => {
      try { sub.remove(); } catch (_) {}
    });
    subscriptionsRef.current = [];

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (socketRef.current) {
      try {
        flushBuffer();
        socketRef.current.close(1000, 'Streaming stopped');
      } catch (_) {}
      socketRef.current = null;
    }

    bufferRef.current = [];
    pipelineRef.current = null;

    if (mountedRef.current) {
      setStatus('idle');
    }
  }, [flushBuffer]);

  const start = useCallback(async (matrix = mountMatrix('upright')) => {
    stop();
    activeRef.current = true;
    setError(null);
    setStatus('connecting');

    try {
      if (!['android', 'ios'].includes(Platform.OS)) {
        throw new Error('IMU streaming requires an Android or iOS device.');
      }

      const [accAvail, gyroAvail] = await Promise.all([
        Accelerometer.isAvailableAsync(),
        Gyroscope.isAvailableAsync(),
      ]);

      if (!activeRef.current) return;
      if (!accAvail || !gyroAvail) {
        throw new Error('Device is missing accelerometer or gyroscope.');
      }

      const motionPerm = await Accelerometer.requestPermissionsAsync();
      if (!activeRef.current) return;
      if (!motionPerm.granted) {
        throw new Error('Motion sensor permission is required.');
      }

      const locationPerm = await Location.requestForegroundPermissionsAsync();
      if (!activeRef.current) return;
      if (!locationPerm.granted) {
        throw new Error('Location permission is required for GPS and speed.');
      }

      // Initialize MotionPipeline
      const pipeline = new MotionPipeline({
        platform: Platform.OS,
        matrix,
        onSample: (sample) => {
          // If pipeline produces a sample, we can sync or observe it
        },
      });
      pipelineRef.current = pipeline;

      // Connect WebSocket
      const targetUrl = wsUrl || getWebSocketUrl('/ws/imu');
      const ws = new WebSocket(targetUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        if (!activeRef.current) {
          ws.close();
          return;
        }
        if (mountedRef.current) {
          setStatus('streaming');
        }
        try {
          ws.send(JSON.stringify({
            type: 'handshake',
            client: Platform.OS,
            timestamp: Date.now(),
            rate_hz: 5,
          }));
        } catch (_) {}
      };

      ws.onerror = (e) => {
        const msg = e.message || 'WebSocket error connecting to backend';
        if (mountedRef.current) {
          setError(msg);
          setStatus('error');
        }
      };

      ws.onclose = (e) => {
        if (activeRef.current && mountedRef.current) {
          setStatus('idle');
          if (e.code !== 1000) {
            setError(`WebSocket closed: code ${e.code} (${e.reason || 'network issue'})`);
          }
        }
      };

      // Set sensor intervals to 10ms (100 Hz)
      Accelerometer.setUpdateInterval(10);
      Gyroscope.setUpdateInterval(10);

      const latestGyro = { x: 0, y: 0, z: 0 };

      const gyroSub = Gyroscope.addListener((event) => {
        latestGyro.x = event.x;
        latestGyro.y = event.y;
        latestGyro.z = event.z;
        pipeline.add('gyro', event);
      });

      const accSub = Accelerometer.addListener((event) => {
        pipeline.add('accel', event);

        // Robust direct 100Hz sample generation with normalized timestamp
        const nowMs = Date.now();
        let ts = event.timestamp;
        if (!Number.isFinite(ts) || ts < 0) {
          ts = nowMs / 1000;
        } else if (ts > 1e11) {
          ts = ts / 1e9;
        } else if (ts > 1e8) {
          ts = ts / 1e3;
        }

        const siAccel = accelerationSI(event, Platform.OS);
        const vehicleAccel = matrixRotate(matrix, siAccel);
        const vehicleGyro = matrixRotate(matrix, [latestGyro.x, latestGyro.y, latestGyro.z]);
        const gps = latestGpsRef.current?.coords;

        const formatted = {
          time: ts,
          accel_x: vehicleAccel[0],
          accel_y: vehicleAccel[1],
          accel_z: vehicleAccel[2],
          gyro_x: vehicleGyro[0],
          gyro_y: vehicleGyro[1],
          gyro_z: vehicleGyro[2],
          speed: gps && Number.isFinite(gps.speed) ? gps.speed : null,
          latitude: gps && Number.isFinite(gps.latitude) ? gps.latitude : null,
          longitude: gps && Number.isFinite(gps.longitude) ? gps.longitude : null,
        };

        bufferRef.current.push(formatted);
        if (mountedRef.current) {
          setLatestSample(formatted);
        }
      });

      subscriptionsRef.current.push(accSub, gyroSub);

      // Watch GPS position
      const locSub = await Location.watchPositionAsync(
        { accuracy: Location.Accuracy.BestForNavigation, timeInterval: 1000, distanceInterval: 0 },
        (position) => {
          latestGpsRef.current = position;
          pipeline.setLocation(position);
        },
      );
      if (!activeRef.current) {
        locSub.remove();
        return;
      }
      subscriptionsRef.current.push(locSub);

      // Batch flush timer every 200ms (5 times per second)
      timerRef.current = setInterval(flushBuffer, batchIntervalMs);

    } catch (err) {
      if (mountedRef.current) {
        setError(err.message);
        setStatus('error');
      }
      stop();
    }
  }, [batchIntervalMs, flushBuffer, stop, wsUrl]);

  // Determine auto-stream condition if navigation is provided
  const shouldAutoStream = isNavigating !== null
    ? shouldStreamImu(isNavigating, tiltAngle, maxTiltAngle)
    : null;

  let pauseReason = null;
  if (isNavigating !== null) {
    if (!isNavigating) {
      pauseReason = 'not_navigating';
    } else if (tiltAngle > maxTiltAngle) {
      pauseReason = 'tilt_exceeded';
    }
  }

  useEffect(() => {
    if (shouldAutoStream === null) return;
    if (shouldAutoStream) {
      if (!activeRef.current) {
        start();
      }
    } else {
      if (activeRef.current) {
        stop();
      }
    }
  }, [shouldAutoStream, start, stop]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stop();
    };
  }, [stop]);

  return {
    status,
    isStreaming: status === 'streaming' || status === 'connecting',
    shouldAutoStream,
    pauseReason,
    isNavigating,
    tiltAngle,
    maxTiltAngle,
    batchIntervalMs,
    error,
    wsUrl,
    setWsUrl,
    stats,
    latestSample,
    start,
    stop,
    flushBuffer,
  };
}
