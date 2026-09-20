import { useCallback, useEffect, useRef, useState } from 'react';
import * as Location from 'expo-location';
import { getWebSocketUrl } from '../api.js';
import { cameraFrame, frameId } from './cameraFrame.js';
import { CAMERA_FRAME_INTERVAL_MS, DEFAULT_MAX_TILT_DEGREES, shouldStreamCamera } from './cameraStreamConfig.js';

export { CAMERA_FRAME_INTERVAL_MS, DEFAULT_MAX_TILT_DEGREES, shouldStreamCamera };

export function useCameraStreamer({ cameraRef = null, isNavigating = false, isConfirmed = false,
  tiltAngle = 0, maxTiltAngle = DEFAULT_MAX_TILT_DEGREES, defaultUrl = null, targetFps = 16 } = {}) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [framesSent, setFramesSent] = useState(0);
  const [error, setError] = useState(null);
  const [latestPrediction, setLatestPrediction] = useState(null);
  const socketRef = useRef(null), timerRef = useRef(null), frameCountRef = useRef(0);
  const activeRef = useRef(false), mountedRef = useRef(true), capturingRef = useRef(false);
  const pendingRef = useRef(null), positionRef = useRef(null), locationSubRef = useRef(null);
  const deviceIdRef = useRef(frameId());
  const shouldStream = shouldStreamCamera(isNavigating, isConfirmed, tiltAngle, maxTiltAngle);

  const stop = useCallback(() => {
    activeRef.current = false;
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    locationSubRef.current?.remove();
    locationSubRef.current = null;
    const ws = socketRef.current;
    socketRef.current = null;
    if (ws) ws.close(1000, 'Camera stream stopped');
    pendingRef.current = null;
    positionRef.current = null;
    if (mountedRef.current) setIsStreaming(false);
  }, []);

  const sendFrame = useCallback(async () => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !activeRef.current) return;
    const pending = pendingRef.current;
    if (pending) {
      if (Date.now() - pending.sentAt >= 95000) {
        if (pending.retries >= 3) {
          stop();
          if (mountedRef.current) setError('Camera inference is unavailable; reconnect to resume.');
          return;
        }
        pending.retries++;
        pending.sentAt = Date.now();
        ws.send(JSON.stringify(pending.payload));
      }
      return;
    }
    if (capturingRef.current || !cameraRef?.current) return;
    capturingRef.current = true;
    const timestamp = Date.now(), position = positionRef.current;
    try {
      const picture = await cameraRef.current.takePictureAsync({
        quality: 0.5, skipProcessing: false, shutterSound: false, base64: true,
      });
      if (!activeRef.current || socketRef.current !== ws || ws.readyState !== WebSocket.OPEN) return;
      const number = ++frameCountRef.current;
      const payload = cameraFrame({ image: picture?.base64, id: frameId(), deviceId: deviceIdRef.current,
        number, timestamp, position });
      pendingRef.current = { payload, sentAt: Date.now(), retries: 0 };
      ws.send(JSON.stringify(payload));
      if (mountedRef.current) setFramesSent(number);
    } catch (err) {
      if (mountedRef.current && socketRef.current === ws) setError(err.message);
    } finally {
      capturingRef.current = false;
    }
  }, [cameraRef, stop]);

  const start = useCallback(() => {
    stop();
    activeRef.current = true;
    setError(null);
    const ws = new WebSocket(defaultUrl || getWebSocketUrl('/ws/camera'));
    socketRef.current = ws;
    ws.onopen = async () => {
      if (!activeRef.current || socketRef.current !== ws) { ws.close(); return; }
      if (mountedRef.current) setIsStreaming(true);
      timerRef.current = setInterval(sendFrame, Math.max(60, Math.round(1000/targetFps) || CAMERA_FRAME_INTERVAL_MS));
      try {
        const permission = await Location.requestForegroundPermissionsAsync();
        if (!permission.granted || socketRef.current !== ws) return;
        const sub = await Location.watchPositionAsync({ accuracy: Location.Accuracy.BestForNavigation,
          timeInterval: 1000, distanceInterval: 0 }, (position) => {
          if (socketRef.current === ws) positionRef.current = { ...position, receivedAt: Date.now() };
        });
        if (socketRef.current !== ws) sub.remove(); else locationSubRef.current = sub;
      } catch (err) {
        if (mountedRef.current && socketRef.current === ws) setError(`Camera GPS unavailable: ${err.message}`);
      }
    };
    ws.onmessage = (event) => {
      if (socketRef.current !== ws) return;
      let reply;
      try { reply = JSON.parse(event.data); } catch (_) { return; }
      const pending = pendingRef.current;
      if (!pending || (reply.frame_id && reply.frame_id !== pending.payload.frame_id)) return;
      if (reply.status === 'ok' && (reply.frame_id || reply.frame === pending.payload.frame_number)) {
        pendingRef.current = null;
        if (mountedRef.current) { setLatestPrediction(reply); setError(null); }
      } else if (reply.status === 'error') {
        if (reply.retryable) pending.sentAt = Date.now()-94000;
        else pendingRef.current = null;
        if (mountedRef.current) setError(reply.message);
      }
    };
    ws.onerror = (event) => {
      if (socketRef.current === ws && mountedRef.current) setError(event.message || 'Camera connection failed');
    };
    ws.onclose = () => { if (socketRef.current === ws) stop(); };
  }, [defaultUrl, sendFrame, stop, targetFps]);

  useEffect(() => {
    if (shouldStream && !activeRef.current) start();
    else if (!shouldStream && activeRef.current) stop();
  }, [shouldStream, start, stop]);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; stop(); };
  }, [stop]);
  return { isStreaming, framesSent, error, latestPrediction, shouldStream, start, stop };
}
