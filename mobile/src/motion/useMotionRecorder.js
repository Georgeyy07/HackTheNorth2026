import { useCallback, useEffect, useRef, useState } from 'react';
import { AppState, Platform } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import * as Location from 'expo-location';
import { activateKeepAwakeAsync, deactivateKeepAwake } from 'expo-keep-awake';
import { createRecordingSink, listRecordings, recordingTag } from './files';
import { RecordingSession } from './session';

export function useMotionRecorder() {
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [live, setLive] = useState(null);
  const [recordings, setRecordings] = useState([]);
  const current = useRef(null);
  const generation = useRef(0);
  const mounted = useRef(true);

  const refresh = useCallback(() => {
    if (!mounted.current) return;
    try { setRecordings(listRecordings()); } catch (err) { setError(err.message); }
  }, []);

  const stop = useCallback((reason = 'stopped') => {
    generation.current++;
    const active = current.current;
    current.current = null;
    if (active) {
      active.subscriptions.forEach((s) => s.remove());
      clearInterval(active.timer);
      try { active.session.close(reason); } catch (err) {
        if (mounted.current) setError(`Recording write failed: ${err.message}`);
      }
      deactivateKeepAwake(active.keepAwakeTag).catch(() => {});
    }
    if (mounted.current) {
      setStatus('idle');
      setLive(null);
      refresh();
    }
  }, [refresh]);

  useEffect(() => {
    mounted.current = true;
    refresh();
    const subscription = AppState.addEventListener('change', (state) => {
      if (state === 'background') stop('app-backgrounded');
    });
    return () => {
      mounted.current = false;
      subscription.remove();
      stop('screen-unmounted');
    };
  }, [refresh, stop]);

  const start = useCallback(async (matrix, videoTag = recordingTag()) => {
    if (current.current) return;
    const token = ++generation.current;
    const valid = () => mounted.current && generation.current === token;
    setStatus('starting');
    setError(null);
    setLive(null);
    try {
      if (!['android', 'ios'].includes(Platform.OS)) throw new Error('IMU recording requires an Android or iOS device.');
      const [acc, gyro] = await Promise.all([Accelerometer.isAvailableAsync(), Gyroscope.isAvailableAsync()]);
      if (!valid()) return;
      if (!acc || !gyro) throw new Error('This device needs both an accelerometer and a gyroscope.');
      const permission = await Accelerometer.requestPermissionsAsync();
      if (!valid()) return;
      if (!permission.granted) throw new Error('Motion permission is required to record IMU data.');
      const locationPermission = await Location.requestForegroundPermissionsAsync();
      if (!valid()) return;
      if (!locationPermission.granted) throw new Error('Location permission is required to record GPS speed.');
      const sink = createRecordingSink(videoTag);
      let session;
      try { session = new RecordingSession({ sink, platform: Platform.OS, matrix }); }
      catch (err) { sink.close(); throw err; }
      const active = { session, subscriptions: [], keepAwakeTag: `motion-${token}`, timer: null };
      current.current = active;
      const guard = (fn) => (...args) => {
        if (!valid()) return;
        try { fn(...args); } catch (err) {
          setError(err.message);
          stop('recording-error');
        }
      };
      Accelerometer.setUpdateInterval(10);
      Gyroscope.setUpdateInterval(10);
      active.subscriptions.push(Accelerometer.addListener(guard((e) => session.sensor('accel', e))));
      active.subscriptions.push(Gyroscope.addListener(guard((e) => session.sensor('gyro', e))));
      active.timer = setInterval(guard(() => {
        sink.flush();
        const latest = session.pipeline.latest;
        const fresh = latest && Date.now()-latest.availableAt <= 250;
        setLive({ sample: fresh ? latest : null, counts: { ...session.counts }, rejected: session.pipeline.rejected });
      }), 200);
      // The late result must remove itself if Stop/background/unmount won the race.
      const location = await Location.watchPositionAsync(
        { accuracy: Location.Accuracy.BestForNavigation, timeInterval: 1000, distanceInterval: 0 },
        guard((position) => session.location(position)),
        guard((message) => { session.pipeline.speedFix = null; setError(`GPS: ${message}`); }),
      );
      if (!valid()) { location.remove(); return; }
      active.subscriptions.push(location);
      await activateKeepAwakeAsync(active.keepAwakeTag);
      if (!valid()) { await deactivateKeepAwake(active.keepAwakeTag); return false; }
      setStatus('recording');
      return true;
    } catch (err) {
      if (!valid()) return false;
      setError(err.message);
      stop('start-error');
      return false;
    }
  }, [stop]);

  const calibrate = useCallback(() => {
    const session = current.current?.session;
    if (!session) return;
    const previousSegment = session.pipeline.segment;
    try {
      session.calibrate();
      setLive(null);
      setError(null);
    } catch (err) {
      setError(err.message);
      // If rotation changed but writing its metadata failed, stop rather than
      // allowing samples whose mount can no longer be reconstructed.
      if (session.pipeline.segment !== previousSegment) stop('calibration-write-error');
    }
  }, [stop]);

  return { status, error, live, recordings, start, stop, calibrate, refresh };
}
