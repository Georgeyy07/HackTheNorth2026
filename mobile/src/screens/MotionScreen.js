import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, TextInput, ScrollView, StyleSheet, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as Sharing from 'expo-sharing';
import { CameraView, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';
import { MOUNTS, mountMatrix } from '../motion/vehicle';
import { useMotionRecorder } from '../motion/useMotionRecorder';
import { useImuStream } from '../context/ImuStreamContext';
import { recordingTag, saveRecordingVideo } from '../motion/files';

const value = (v) => Number.isFinite(v) ? v.toFixed(3) : '—';

function Button({ children, onPress, disabled = false }) {
  return <TouchableOpacity accessibilityRole="button" disabled={disabled} onPress={onPress}
    style={[styles.button, disabled && styles.disabled]}><Text style={styles.buttonText}>{children}</Text></TouchableOpacity>;
}

export default function MotionScreen() {
  const recorder = useMotionRecorder();
  const streamer = useImuStream();
  const [mount, setMount] = useState('upright');
  const [angles, setAngles] = useState(['0', '0', '0']);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [micPermission, requestMicPermission] = useMicrophonePermissions();
  const cameraRef = useRef(null);
  const filming = useRef(null); // { tag, pending: Promise<{uri}> } while a demo clip is recording
  const cameraReadyRef = useRef(false); // set synchronously by onCameraReady; read via polling below
  const active = recorder.status !== 'idle';
  const sample = recorder.live?.sample;
  // Mounted as soon as permission is granted, not gated on `active`: mounting
  // it only when recording starts raced onCameraReady against recordAsync()
  // and threw CameraOutputNotReadyException. Mounting early gives the native
  // camera time to warm up before Start is ever pressed.
  const showCamera = cameraPermission?.granted;

  // onCameraReady can fire before the native recording pipeline (not just
  // the preview) is actually ready to record -- a known expo-camera gap, not
  // just an ordering bug here. Poll a ref (always current, unlike state
  // captured in this closure) and add a short buffer after it flips.
  const waitForCameraReady = (timeoutMs = 4000) => new Promise((resolve) => {
    const deadline = Date.now() + timeoutMs;
    const check = () => {
      if (cameraReadyRef.current) resolve(true);
      else if (Date.now() >= deadline) resolve(false);
      else setTimeout(check, 100);
    };
    check();
  });

  // Runs on every stop path (button, app-backgrounded, screen-unmounted), not
  // just the explicit Stop button, so a backgrounded clip is never orphaned
  // in cache while its JSONL session still gets closed and saved normally.
  useEffect(() => {
    recorder.onStopRef.current = () => {
      const clip = filming.current;
      filming.current = null;
      if (!clip) return;
      cameraRef.current?.stopRecording();
      clip.pending
        .then(({ uri }) => saveRecordingVideo(clip.tag, uri))
        .then(() => recorder.refresh())
        .catch((err) => Alert.alert('Demo video', `Recording saved without its video: ${err.message}`));
    };
  }, [recorder]);

  const start = async () => {
    try {
      if (angles.some((a) => !a.trim())) throw new Error('Enter all three correction angles.');
      const camera = await requestCameraPermission();
      const mic = await requestMicPermission();
      const tag = recordingTag();
      const readyPromise = camera.granted ? waitForCameraReady() : null;
      const ok = await recorder.start(mountMatrix(mount, ...angles.map(Number)), tag);
      // The demo clip is a bonus for showing the run later; never let it block
      // or fail the IMU/GPS recording that the model actually depends on.
      if (ok && readyPromise && await readyPromise && cameraRef.current) {
        // onCameraReady flipping true still isn't a hard guarantee the
        // recording pipeline specifically is ready (confirmed: this exact
        // exception recurred with the plain ready-check alone) -- a short
        // settle delay closes that gap in practice.
        await new Promise((resolve) => setTimeout(resolve, 400));
        filming.current = { tag, pending: cameraRef.current.recordAsync({ mute: !mic.granted }) };
      }
    } catch (err) { Alert.alert('Mount settings', err.message); }
  };

  const share = async (file) => {
    try {
      if (!await Sharing.isAvailableAsync()) throw new Error('File sharing is unavailable on this device.');
      await Sharing.shareAsync(file.uri, { mimeType: 'application/x-ndjson', UTI: 'public.plain-text', dialogTitle: 'Export raw + VQF recording' });
    } catch (err) { Alert.alert('Export recording', err.message); }
  };

  const shareVideo = async (video) => {
    try {
      if (!await Sharing.isAvailableAsync()) throw new Error('File sharing is unavailable on this device.');
      await Sharing.shareAsync(video.uri, { mimeType: 'video/mp4', dialogTitle: 'Export demonstration video' });
    } catch (err) { Alert.alert('Export video', err.message); }
  };

  return <SafeAreaView style={styles.container} edges={['top']}>
    <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
      <Text style={styles.title}>Motion recorder</Text>
      <Text style={styles.description}>Record original IMU and GPS events alongside VQF-stabilized motion. Keep the phone rigidly mounted and the app in the foreground.</Text>
      <View style={styles.card}>
        <Text style={styles.heading}>Phone mount</Text>
        <Text style={styles.description}>Screen faces the occupants, toward the rear of the car.</Text>
        <View style={styles.row}>{Object.entries(MOUNTS).map(([key, entry]) =>
          <TouchableOpacity key={key} accessibilityRole="button" accessibilityState={{ selected: mount === key }} disabled={active}
            onPress={() => setMount(key)} style={[styles.chip, mount === key && styles.selected]}>
            <Text style={styles.text}>{entry.label}</Text>
          </TouchableOpacity>)}</View>
        <Text style={styles.description}>Optional mount corrections in degrees: rotate around vehicle X (roll), then Y (pitch), then Z (yaw), using the right-hand rule.</Text>
        <View style={styles.row}>{['Roll', 'Pitch', 'Yaw'].map((label, i) =>
          <View key={label} style={styles.angle}><Text style={styles.text}>{label}</Text>
            <TextInput accessibilityLabel={`${label} correction in degrees`} editable={!active} value={angles[i]}
              onChangeText={(text) => setAngles((old) => old.map((v, j) => i === j ? text : v))}
              style={styles.input} keyboardType="numbers-and-punctuation" autoCorrect={false} />
          </View>)}</View>
        <Text style={styles.description}>For tilt calibration, park on level ground with a fresh GPS fix and keep the mount still. Calibration cannot determine yaw; align the screen toward the rear or set yaw manually.</Text>
        <Button disabled={!sample?.canCalibrate || recorder.status !== 'recording'} onPress={recorder.calibrate}>Calibrate parked tilt</Button>
      </View>
      {recorder.error && <Text accessibilityRole="alert" style={styles.error}>{recorder.error}</Text>}
      <Button onPress={active ? () => recorder.stop() : start}>{active ? 'Stop and save recording' : 'Start recording'}</Button>
      <Text style={styles.description}>{recorder.status === 'starting' ? 'Starting sensors and GPS…' : active
        ? 'Recording to device storage · stops and saves if the app goes to the background.'
        : 'Sessions are saved locally. Export a session below to share all raw and stabilized samples.'}</Text>

      <View style={styles.card}>
        <View style={styles.streamHeader}>
          <Text style={styles.heading}>IMU WebSocket Stream (100Hz)</Text>
          <View style={[
            styles.statusBadge,
            streamer.isStreaming
              ? styles.statusStreaming
              : streamer.pauseReason === 'tilt_exceeded'
              ? styles.statusAlert
              : styles.statusIdle
          ]}>
            <View style={[
              styles.statusDot,
              streamer.isStreaming
                ? styles.dotStreaming
                : streamer.pauseReason === 'tilt_exceeded'
                ? styles.dotAlert
                : styles.dotIdle
            ]} />
            <Text style={[
              styles.statusText,
              streamer.isStreaming && styles.statusTextStreaming,
              streamer.pauseReason === 'tilt_exceeded' && styles.statusTextAlert
            ]}>
              {streamer.isStreaming
                ? 'Streaming (5x/sec)'
                : streamer.pauseReason === 'tilt_exceeded'
                ? 'Paused (Tilt > 20°)'
                : streamer.isNavigating
                ? 'Connecting…'
                : 'Stopped (Not Navigating)'}
            </Text>
          </View>
        </View>

        <Text style={styles.description}>
          Streams live 100Hz IMU data to backend 5 times/second (20 samples/batch). Active during navigation; stops if navigation ends or tilt exceeds 20°.
        </Text>

        <View style={styles.streamConditionsRow}>
          <Text style={styles.conditionText}>
            Navigation: <Text style={{ color: streamer.isNavigating ? '#34d399' : '#9ca3af', fontWeight: '700' }}>{streamer.isNavigating ? 'Active' : 'Inactive'}</Text>
          </Text>
          <Text style={styles.conditionText}>
            Tilt: <Text style={{ color: streamer.tiltAngle > 20 ? '#ef4444' : '#34d399', fontWeight: '700' }}>{streamer.tiltAngle ? streamer.tiltAngle.toFixed(1) : '0.0'}°</Text> (Max: 20°)
          </Text>
        </View>

        <View style={styles.urlRow}>
          <Text style={styles.urlLabel}>WebSocket Endpoint:</Text>
          <TextInput
            accessibilityLabel="WebSocket Server URL"
            editable={!streamer.isStreaming}
            value={streamer.wsUrl}
            onChangeText={streamer.setWsUrl}
            style={styles.urlInput}
            autoCapitalize="none"
            autoCorrect={false}
          />
        </View>

        {streamer.error && <Text accessibilityRole="alert" style={styles.error}>{streamer.error}</Text>}

        <Button
          onPress={streamer.isStreaming
            ? () => streamer.stop()
            : () => streamer.start(mountMatrix(mount, ...angles.map(Number)))}
        >
          {streamer.isStreaming ? 'Stop WebSocket Stream' : 'Start 100Hz WebSocket Stream'}
        </Button>

        {streamer.isStreaming && (
          <View style={styles.streamStats}>
            <View style={styles.statBox}>
              <Text style={styles.statLabel}>Batches Sent</Text>
              <Text style={styles.statNumber}>{streamer.stats.packetsSent}</Text>
            </View>
            <View style={styles.statBox}>
              <Text style={styles.statLabel}>100Hz Samples</Text>
              <Text style={styles.statNumber}>{streamer.stats.samplesSent}</Text>
            </View>
          </View>
        )}

        {streamer.latestSample && (
          <View style={styles.liveReadingsBox}>
            <Text style={styles.subheading}>Live Streamed Values</Text>
            <View style={styles.reading}>
              <Text style={styles.text}>Accel (X, Y, Z)</Text>
              <Text style={styles.subNumber}>
                [{value(streamer.latestSample.accel_x)}, {value(streamer.latestSample.accel_y)}, {value(streamer.latestSample.accel_z)}] m/s^2
              </Text>
            </View>
            <View style={styles.reading}>
              <Text style={styles.text}>Gyro (X, Y, Z)</Text>
              <Text style={styles.subNumber}>
                [{value(streamer.latestSample.gyro_x)}, {value(streamer.latestSample.gyro_y)}, {value(streamer.latestSample.gyro_z)}] rad/s
              </Text>
            </View>
            <View style={styles.reading}>
              <Text style={styles.text}>Speed</Text>
              <Text style={styles.subNumber}>
                {value(streamer.latestSample.speed)} m/s
              </Text>
            </View>
            <View style={styles.reading}>
              <Text style={styles.text}>GPS Fix</Text>
              <Text style={styles.subNumber}>
                {streamer.latestSample.latitude !== null
                  ? `${streamer.latestSample.latitude.toFixed(5)}, ${streamer.latestSample.longitude.toFixed(5)}`
                  : 'Acquiring GPS…'}
              </Text>
            </View>
          </View>
        )}
      </View>
      {showCamera && <View style={styles.card}>
        <Text style={styles.heading}>Demonstration video</Text>
        <CameraView ref={cameraRef} style={styles.camera} facing="back" mode="video"
          onCameraReady={() => { cameraReadyRef.current = true; }} />
        <Text style={styles.description}>Filming alongside the recording, for a visual demo of the run · saved next to this session's export.</Text>
      </View>}
      <View style={styles.card}>
        <Text style={styles.heading}>Vehicle model input</Text>
        <Text style={styles.description}>Acceleration includes gravity · m/s². X forward, Y left, Z up. Speed in m/s.</Text>
        {['X · forward', 'Y · left', 'Z · up', 'Speed'].map((label, i) =>
          <View style={styles.reading} key={label}><Text style={styles.text}>{label}</Text><Text style={styles.number}>{value(sample?.input[i])}</Text></View>)}
        <Text style={styles.description}>Stationary and level: approximately [0, 0, +9.81]. Missing GPS speed remains blank.</Text>
      </View>
      <View style={styles.card}>
        <Text style={styles.heading}>VQF-stabilized motion</Text>
        <Text style={styles.description}>Earth frame · Z up, arbitrary horizontal heading. Full acceleration and gyro XYZ are saved; this vertical preview subtracts gravity.</Text>
        <Text style={styles.number}>{value(sample?.verticalLinear)} m/s² vertical</Text>
        <Text style={styles.description}>{!sample ? 'Waiting for fresh accelerometer + gyro samples.'
          : sample.settling ? 'Filter settling · first 10 seconds of this segment.' : 'Filter running.'}</Text>
        <Text style={styles.description}>BasicVQF 6D · no magnetometer or gyro-bias estimator.</Text>
      </View>
      {recorder.live && <Text style={styles.description}>
        Saved events: {recorder.live.counts.accelerometer} accel · {recorder.live.counts.gyroscope} gyro · {recorder.live.counts.location} GPS · {recorder.live.counts.stabilized} fused
      </Text>}
      <Text style={styles.heading}>Saved recordings</Text>
      {!recorder.recordings.length && <Text style={styles.description}>No recordings yet.</Text>}
      {recorder.recordings.map((file) => <View style={styles.card} key={file.uri}>
        <Text style={styles.text}>{file.name}</Text>
        <Text style={styles.description}>{(file.size/1024/1024).toFixed(2)} MB · JSONL · raw + stabilized</Text>
        <Button disabled={active} onPress={() => share(file)}>Export session</Button>
        {file.video
          ? <Button disabled={active} onPress={() => shareVideo(file.video)}>Export demo video ({(file.video.size/1024/1024).toFixed(1)} MB)</Button>
          : <Text style={styles.description}>No demo video for this session.</Text>}
      </View>)}
    </ScrollView>
  </SafeAreaView>;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0b0f19' },
  content: { padding: 16, gap: 12, paddingBottom: 40 },
  title: { color: '#00f2fe', fontWeight: '700', fontSize: 24 },
  heading: { color: '#e6f4fe', fontWeight: '700', fontSize: 17 },
  description: { color: '#9ca3af', fontSize: 13, lineHeight: 19 },
  card: { backgroundColor: '#111827', borderRadius: 12, padding: 16, gap: 12 },
  camera: { width: '100%', aspectRatio: 16/9, borderRadius: 8, overflow: 'hidden', backgroundColor: '#000' },
  text: { color: '#e6f4fe', fontSize: 13 },
  row: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
  chip: { borderColor: '#374151', borderWidth: 1, borderRadius: 8, padding: 10 },
  selected: { borderColor: '#00f2fe', backgroundColor: '#153846' },
  angle: { flex: 1, gap: 6 },
  input: { borderWidth: 1, borderColor: '#374151', padding: 10, borderRadius: 8, color: '#fff' },
  button: { backgroundColor: '#00f2fe', padding: 14, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#0b0f19', fontWeight: '700', fontSize: 14 },
  disabled: { opacity: 0.4 },
  error: { color: '#fecaca', backgroundColor: '#7f1d1d', padding: 12, borderRadius: 8 },
  reading: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  number: { color: '#00f2fe', fontSize: 20, fontVariant: ['tabular-nums'] },
  subNumber: { color: '#00f2fe', fontSize: 13, fontVariant: ['tabular-nums'] },
  streamHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  statusBadge: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingVertical: 4, paddingHorizontal: 8, borderRadius: 12 },
  statusStreaming: { backgroundColor: '#064e3b' },
  statusAlert: { backgroundColor: '#7f1d1d' },
  statusIdle: { backgroundColor: '#1f2937' },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  dotStreaming: { backgroundColor: '#34d399' },
  dotAlert: { backgroundColor: '#ef4444' },
  dotIdle: { backgroundColor: '#6b7280' },
  statusText: { fontSize: 11, color: '#9ca3af', fontWeight: '600' },
  statusTextStreaming: { color: '#a7f3d0' },
  statusTextAlert: { color: '#fca5a5' },
  streamConditionsRow: { flexDirection: 'row', justifyContent: 'space-between', backgroundColor: '#1f2937', padding: 8, borderRadius: 8 },
  conditionText: { color: '#d1d5db', fontSize: 12 },
  urlRow: { gap: 6 },
  urlLabel: { color: '#9ca3af', fontSize: 12 },
  urlInput: { borderWidth: 1, borderColor: '#374151', padding: 8, borderRadius: 8, color: '#fff', fontSize: 12 },
  streamStats: { flexDirection: 'row', gap: 12 },
  statBox: { flex: 1, backgroundColor: '#1f2937', padding: 12, borderRadius: 8, alignItems: 'center', gap: 4 },
  statLabel: { color: '#9ca3af', fontSize: 12 },
  statNumber: { color: '#00f2fe', fontSize: 18, fontWeight: '700', fontVariant: ['tabular-nums'] },
  liveReadingsBox: { gap: 6, borderTopWidth: 1, borderTopColor: '#1f2937', paddingTop: 8 },
  subheading: { color: '#e6f4fe', fontSize: 14, fontWeight: '600' },
});
