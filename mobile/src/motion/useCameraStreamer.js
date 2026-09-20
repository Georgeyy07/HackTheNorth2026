import { useCallback, useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';
import { getWebSocketUrl } from '../api';

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

export function useCameraStreamer({
  cameraRef = null,
  isNavigating = false,
  isConfirmed = false,
  tiltAngle = 0,
  maxTiltAngle = DEFAULT_MAX_TILT_DEGREES,
  defaultUrl = null,
  targetFps = 16,
} = {}) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [framesSent, setFramesSent] = useState(0);
  const [error, setError] = useState(null);

  const socketRef = useRef(null);
  const timerRef = useRef(null);
  const frameCountRef = useRef(0);
  const isCapturingRef = useRef(false);
  const lastImageRef = useRef(null);
  const activeRef = useRef(false);
  const mountedRef = useRef(true);

  const shouldStream = shouldStreamCamera(isNavigating, isConfirmed, tiltAngle, maxTiltAngle);

  const stop = useCallback(() => {
    activeRef.current = false;
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (socketRef.current) {
      try {
        socketRef.current.close(1000, 'Camera stream stopped');
      } catch (_) {}
      socketRef.current = null;
    }
    if (mountedRef.current) {
      setIsStreaming(false);
    }
  }, []);

  const sendFrame = useCallback(() => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !activeRef.current) return;

    frameCountRef.current += 1;
    const frameNumber = frameCountRef.current;
    const timestamp = Date.now();

    // Trigger capture in background if not already capturing to keep streaming rate at 15+ fps
    if (cameraRef && cameraRef.current && !isCapturingRef.current) {
      isCapturingRef.current = true;
      cameraRef.current.takePictureAsync({
        quality: 0.1,
        skipProcessing: true,
        shutterSound: false,
        base64: true,
      }).then((pic) => {
        if (pic?.base64) {
          lastImageRef.current = pic.base64;
        }
      }).catch(() => {})
      .finally(() => {
        isCapturingRef.current = false;
      });
    }

    const payload = {
      type: 'camera_frame',
      frame_number: frameNumber,
      timestamp,
      fps: targetFps,
      image: lastImageRef.current,
    };

    try {
      ws.send(JSON.stringify(payload));
      if (mountedRef.current) {
        setFramesSent(frameNumber);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(`Camera WebSocket send failed: ${err.message}`);
      }
    }
  }, [cameraRef, targetFps]);

  const start = useCallback(() => {
    stop();
    activeRef.current = true;
    setError(null);

    const wsUrl = defaultUrl || getWebSocketUrl('/ws/camera');
    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onopen = () => {
      if (!activeRef.current) {
        ws.close();
        return;
      }
      if (mountedRef.current) {
        setIsStreaming(true);
      }
      // Start 15+fps frame streaming interval (~60ms = ~16.6 fps)
      const intervalMs = Math.round(1000 / targetFps) || CAMERA_FRAME_INTERVAL_MS;
      timerRef.current = setInterval(sendFrame, intervalMs);
    };

    ws.onerror = (e) => {
      if (mountedRef.current) {
        setError(e.message || 'Camera WebSocket connection error');
      }
    };

    ws.onclose = () => {
      if (activeRef.current && mountedRef.current) {
        setIsStreaming(false);
      }
    };
  }, [defaultUrl, sendFrame, stop, targetFps]);

  useEffect(() => {
    if (shouldStream) {
      if (!activeRef.current) {
        start();
      }
    } else {
      if (activeRef.current) {
        stop();
      }
    }
  }, [shouldStream, start, stop]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stop();
    };
  }, [stop]);

  return {
    isStreaming,
    framesSent,
    error,
    shouldStream,
    start,
    stop,
  };
}
