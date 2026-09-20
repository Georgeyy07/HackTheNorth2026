import { useCallback, useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import * as Location from 'expo-location';
import { getWebSocketUrl } from '../api.js';
import { MotionPipeline } from './pipeline.js';
import { mountMatrix } from './vehicle.js';
import { modelSample, inferenceHandshake } from './modelSample.js';
import { BATCH_INTERVAL_MS, DEFAULT_MAX_TILT_DEGREES, shouldStreamImu } from './imuStreamConfig.js';

export { BATCH_INTERVAL_MS, DEFAULT_MAX_TILT_DEGREES, shouldStreamImu };

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
  const pendingRef = useRef(null);
  const readyRef = useRef(false);
  const segmentRef = useRef(null);
  const [latestPrediction, setLatestPrediction] = useState(null);

  const flushBuffer = useCallback(() => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!readyRef.current) return;
    if (pendingRef.current) {
      const pending = pendingRef.current;
      if (Date.now() - pending.sentAt > 130000) {
        if (pending.retries >= 3) {
          ws.close(1011, 'Inference acknowledgement timeout');
          return;
        }
        pending.retries++;
        pending.sentAt = Date.now();
        ws.send(JSON.stringify(pending.payload));
      }
      return;
    }
    if (bufferRef.current.length === 0) return;
    const segment = bufferRef.current[0].segment;
    if (segmentRef.current !== null && segment !== segmentRef.current) {
      readyRef.current = false;
      segmentRef.current = segment;
      ws.send(JSON.stringify(inferenceHandshake(Platform.OS)));
      return;
    }
    segmentRef.current = segment;
    const boundary = bufferRef.current.findIndex((sample) => sample.segment !== segment);
    const count = Math.min(1024, boundary < 0 ? bufferRef.current.length : boundary);
    const samplesToSend = bufferRef.current.splice(0, count);
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
      pendingRef.current = { payload, sentAt: Date.now(), retries: 0 };
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
    pendingRef.current = null;
    readyRef.current = false;
    segmentRef.current = null;
    latestGpsRef.current = null;
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
          const formatted = modelSample(sample, pipeline.matrix, latestGpsRef.current?.receivedAt);
          if (bufferRef.current.length >= 15000) {
            stop();
            setError('Inference cannot keep up; reconnect to start a fresh session.');
            setStatus('error');
            return;
          }
          bufferRef.current.push(formatted);
          if (mountedRef.current) setLatestSample(formatted);
        },
      });
      pipelineRef.current = pipeline;

      // Connect WebSocket
      const targetUrl = wsUrl || getWebSocketUrl('/ws/imu');
      const ws = new WebSocket(targetUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        if (!activeRef.current || socketRef.current !== ws) {
          ws.close();
          return;
        }
        if (mountedRef.current) {
          setStatus('streaming');
        }
        try {
          ws.send(JSON.stringify(inferenceHandshake(Platform.OS)));
        } catch (_) {}
      };

      ws.onmessage = (event) => {
        if (socketRef.current !== ws) return;
        let message;
        try { message = JSON.parse(event.data); } catch (_) { return; }
        if (message.status === 'ready') {
          readyRef.current = true;
          flushBuffer();
        } else if (message.status === 'ok' && message.batch_id === pendingRef.current?.payload.batch_id) {
          pendingRef.current = null;
          const latest = message.updates?.filter((u) => u.is_final).at(-1);
          if (latest && mountedRef.current) setLatestPrediction(latest);
          flushBuffer();
        } else if (message.status === 'error') {
          if (message.retryable && pendingRef.current) {
            // Keep the exact batch until acknowledged; retry on the flush timer.
            pendingRef.current.sentAt = Date.now() - 129000;
          } else {
            stop();
            setError(message.message || 'Inference rejected the stream');
            setStatus('error');
          }
        }
      };

      ws.onerror = (e) => {
        if (socketRef.current !== ws) return;
        const msg = e.message || 'WebSocket error connecting to backend';
        if (mountedRef.current) {
          setError(msg);
          setStatus('error');
        }
      };

      ws.onclose = (e) => {
        if (socketRef.current !== ws) return;
        if (activeRef.current && mountedRef.current) {
          stop();
          if (e.code !== 1000) {
            setError(`WebSocket closed: code ${e.code} (${e.reason || 'network issue'})`);
          }
        }
      };

      // Set sensor intervals to 10ms (100 Hz)
      Accelerometer.setUpdateInterval(10);
      Gyroscope.setUpdateInterval(10);

      const gyroSub = Gyroscope.addListener((event) => pipeline.add('gyro', event));
      const accSub = Accelerometer.addListener((event) => pipeline.add('accel', event));

      subscriptionsRef.current.push(accSub, gyroSub);

      // Watch GPS position
      const locSub = await Location.watchPositionAsync(
        { accuracy: Location.Accuracy.BestForNavigation, timeInterval: 1000, distanceInterval: 0 },
        (position) => {
          latestGpsRef.current = { ...position, receivedAt: Date.now() };
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
    latestPrediction,
    start,
    stop,
    flushBuffer,
  };
}
