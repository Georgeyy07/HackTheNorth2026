import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, TextInput, ScrollView, StyleSheet, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as Sharing from 'expo-sharing';
import { CameraView, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';
import { MOUNTS, mountMatrix } from '../motion/vehicle';
import { useMotionRecorder } from '../motion/useMotionRecorder';
import { useImuStream } from '../context/ImuStreamContext';
import { recordingTag, saveRecordingVideo } from '../motion/files';
import { COLORS, WoodButton } from '../theme';

const value = (v) => Number.isFinite(v) ? v.toFixed(3) : '—';

function Button({ children, onPress, disabled = false, variant = 'default' }) {
  return <WoodButton onPress={onPress} disabled={disabled} variant={variant}>{children}</WoodButton>;
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
    <View style={styles.header}>
      <View>
        <Text style={styles.headerTitle}>Motion Recorder</Text>
        <Text style={styles.headerSub}>Raw IMU & GPS · VQF stabilized</Text>
      </View>
      <View style={[styles.recordBadge, active && styles.recordBadgeActive]}>
        <View style={[styles.recordDot, active && styles.recordDotActive]} />
        <Text style={[styles.recordBadgeText, active && styles.recordBadgeTextActive]}>
          {recorder.status === 'starting' ? 'Starting' : active ? 'Recording' : 'Idle'}
        </Text>
      </View>
    </View>

    <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
      <Text style={styles.description}>Record original IMU and GPS events alongside VQF-stabilized motion. Keep the phone rigidly mounted and the app in the foreground.</Text>
      <View style={styles.card}>
        <Text style={styles.heading}>Phone mount</Text>
        <Text style={styles.description}>Screen faces the occupants, toward the rear of the car.</Text>
        <View style={styles.row}>{Object.entries(MOUNTS).map(([key, entry]) =>
          <TouchableOpacity key={key} accessibilityRole="button" accessibilityState={{ selected: mount === key }} disabled={active}
            onPress={() => setMount(key)} style={[styles.chip, mount === key && styles.selected, active && styles.chipDisabled]}>
            <Text style={[styles.text, mount === key && styles.selectedText]}>{entry.label}</Text>
          </TouchableOpacity>)}</View>
        <Text style={styles.description}>Optional mount corrections in degrees: rotate around vehicle X (roll), then Y (pitch), then Z (yaw), using the right-hand rule.</Text>
        <View style={styles.row}>{['Roll', 'Pitch', 'Yaw'].map((label, i) =>
          <View key={label} style={styles.angle}><Text style={styles.label}>{label}</Text>
            <TextInput accessibilityLabel={`${label} correction in degrees`} editable={!active} value={angles[i]}
              onChangeText={(text) => setAngles((old) => old.map((v, j) => i === j ? text : v))}
              style={[styles.input, active && styles.inputDisabled]} keyboardType="numbers-and-punctuation" autoCorrect={false}
              placeholderTextColor={COLORS.inkMuted} />
          </View>)}</View>
        <Text style={styles.description}>For tilt calibration, park on level ground with a fresh GPS fix and keep the mount still. Calibration cannot determine yaw; align the screen toward the rear or set yaw manually.</Text>
        <Button variant="forest" disabled={!sample?.canCalibrate || recorder.status !== 'recording'} onPress={recorder.calibrate}>Calibrate parked tilt</Button>
      </View>
      {recorder.error && <View style={styles.errorBanner}><Text accessibilityRole="alert" style={styles.errorText}>{recorder.error}</Text></View>}
      <Button variant={active ? 'danger' : 'default'} onPress={active ? () => recorder.stop() : start}>{active ? 'Stop and save recording' : 'Start recording'}</Button>
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
            Navigation: <Text style={streamer.isNavigating ? styles.conditionOn : styles.conditionOff}>{streamer.isNavigating ? 'Active' : 'Inactive'}</Text>
          </Text>
          <Text style={styles.conditionText}>
            {/* Tilt: <Text style={{ color: streamer.tiltAngle > 20 ? '#ef4444' : '#34d399', fontWeight: '700' }}>{streamer.tiltAngle ? streamer.tiltAngle.toFixed(1) : '0.0'}°</Text> (Max: 20°) */}
          </Text>
        </View>

        <View style={styles.urlRow}>
          <Text style={styles.label}>WebSocket Endpoint:</Text>
          <TextInput
            accessibilityLabel="WebSocket Server URL"
            editable={!streamer.isStreaming}
            value={streamer.wsUrl}
            onChangeText={streamer.setWsUrl}
            style={[styles.input, styles.urlInput, streamer.isStreaming && styles.inputDisabled]}
            autoCapitalize="none"
            autoCorrect={false}
            placeholderTextColor={COLORS.inkMuted}
          />
        </View>

        {streamer.error && <View style={styles.errorBanner}><Text accessibilityRole="alert" style={styles.errorText}>{streamer.error}</Text></View>}

        <Button
          variant={streamer.isStreaming ? 'danger' : 'default'}
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
        <Text style={styles.bigNumber}>{value(sample?.verticalLinear)} m/s² vertical</Text>
        <Text style={styles.description}>{!sample ? 'Waiting for fresh accelerometer + gyro samples.'
          : sample.settling ? 'Filter settling · first 10 seconds of this segment.' : 'Filter running.'}</Text>
        <Text style={styles.description}>BasicVQF 6D · no magnetometer or gyro-bias estimator.</Text>
      </View>
      {recorder.live && <Text style={styles.description}>
        Saved events: {recorder.live.counts.accelerometer} accel · {recorder.live.counts.gyroscope} gyro · {recorder.live.counts.location} GPS · {recorder.live.counts.stabilized} fused
      </Text>}
      <Text style={styles.sectionTitle}>Saved recordings</Text>
      {!recorder.recordings.length && <Text style={styles.description}>No recordings yet.</Text>}
      {recorder.recordings.map((file) => <View style={styles.card} key={file.uri}>
        <Text style={styles.fileName}>{file.name}</Text>
        <Text style={styles.description}>{(file.size / 1024 / 1024).toFixed(2)} MB · JSONL · raw + stabilized</Text>
        <Button disabled={active} onPress={() => share(file)}>Export session</Button>
        {file.video
          ? <Button variant="forest" disabled={active} onPress={() => shareVideo(file.video)}>{`Export demo video (${(file.video.size / 1024 / 1024).toFixed(1)} MB)`}</Button>
          : <Text style={styles.description}>No demo video for this session.</Text>}
      </View>)}
    </ScrollView>
  </SafeAreaView>;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.parchmentBg },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: COLORS.parchmentSurface,
    borderBottomWidth: 1.5,
    borderBottomColor: COLORS.parchmentBorder,
  },
  headerTitle: { fontSize: 17, fontWeight: '800', color: COLORS.inkPrimary, fontFamily: 'serif' },
  headerSub: { fontSize: 11, color: COLORS.inkSecondary, fontFamily: 'serif', marginTop: 1 },
  recordBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: COLORS.parchmentBorderDark,
    backgroundColor: COLORS.parchmentCard,
  },
  recordBadgeActive: { borderColor: COLORS.terracotta, backgroundColor: '#fbe4d6' },
  recordDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.inkMuted },
  recordDotActive: { backgroundColor: COLORS.terracotta },
  recordBadgeText: { fontSize: 11, fontWeight: '700', color: COLORS.inkSecondary, fontFamily: 'serif' },
  recordBadgeTextActive: { color: COLORS.terracottaDark },

  content: { padding: 16, gap: 12, paddingBottom: 40 },
  sectionTitle: { color: COLORS.inkPrimary, fontWeight: '800', fontSize: 17, fontFamily: 'serif', marginTop: 4 },
  heading: { color: COLORS.inkPrimary, fontWeight: '700', fontSize: 16, fontFamily: 'serif' },
  subheading: { color: COLORS.inkPrimary, fontSize: 14, fontWeight: '700', fontFamily: 'serif' },
  description: { color: COLORS.inkSecondary, fontSize: 12, lineHeight: 18, fontFamily: 'serif' },
  label: { color: COLORS.inkMuted, fontSize: 11, fontFamily: 'serif', fontWeight: '700' },
  card: {
    backgroundColor: COLORS.parchmentCard,
    borderRadius: 16,
    borderWidth: 1.5,
    borderColor: COLORS.parchmentBorderDark,
    padding: 14,
    gap: 10,
  },
  camera: {
    width: '100%',
    aspectRatio: 16 / 9,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: COLORS.parchmentBorderDark,
    overflow: 'hidden',
    backgroundColor: '#2a1408',
  },
  text: { color: COLORS.inkPrimary, fontSize: 12, fontFamily: 'serif' },
  fileName: { color: COLORS.inkPrimary, fontSize: 13, fontWeight: '700', fontFamily: 'serif' },
  row: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
  chip: {
    borderColor: COLORS.parchmentBorderDark,
    backgroundColor: COLORS.parchmentInput,
    borderWidth: 1.5,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  selected: { borderColor: COLORS.terracotta, backgroundColor: '#fbe4d6' },
  selectedText: { color: COLORS.terracottaDark, fontWeight: '700' },
  chipDisabled: { opacity: 0.55 },
  angle: { flex: 1, gap: 5 },
  input: {
    borderWidth: 1.5,
    borderColor: COLORS.parchmentBorder,
    backgroundColor: COLORS.parchmentInput,
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 8,
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
    fontSize: 13,
  },
  inputDisabled: { opacity: 0.7, backgroundColor: COLORS.parchmentSurface },
  errorBanner: {
    backgroundColor: '#f5dedb',
    borderColor: COLORS.hazardCritical,
    borderWidth: 1,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  errorText: { fontSize: 11, color: COLORS.hazardCritical, fontFamily: 'serif' },
  reading: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  number: { color: COLORS.terracottaDark, fontSize: 17, fontFamily: 'serif', fontWeight: '700', fontVariant: ['tabular-nums'] },
  bigNumber: { color: COLORS.terracottaDark, fontSize: 20, fontFamily: 'serif', fontWeight: '800', fontVariant: ['tabular-nums'] },
  subNumber: { color: COLORS.terracottaDark, fontSize: 12, fontFamily: 'serif', fontVariant: ['tabular-nums'] },

  streamHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 6 },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 12,
    borderWidth: 1,
  },
  statusStreaming: { backgroundColor: COLORS.forestTint, borderColor: COLORS.forestMoss },
  statusAlert: { backgroundColor: '#f5dedb', borderColor: COLORS.hazardCritical },
  statusIdle: { backgroundColor: COLORS.parchmentInput, borderColor: COLORS.parchmentBorderDark },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  dotStreaming: { backgroundColor: COLORS.forestPine },
  dotAlert: { backgroundColor: COLORS.hazardCritical },
  dotIdle: { backgroundColor: COLORS.inkMuted },
  statusText: { fontSize: 10, color: COLORS.inkSecondary, fontWeight: '700', fontFamily: 'serif' },
  statusTextStreaming: { color: COLORS.forestDark },
  statusTextAlert: { color: COLORS.hazardCritical },
  streamConditionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: COLORS.parchmentInput,
    borderWidth: 1,
    borderColor: COLORS.parchmentBorder,
    padding: 8,
    borderRadius: 8,
  },
  conditionText: { color: COLORS.inkSecondary, fontSize: 11, fontFamily: 'serif' },
  conditionOn: { color: COLORS.forestPine, fontWeight: '700' },
  conditionOff: { color: COLORS.inkMuted, fontWeight: '700' },
  urlRow: { gap: 5 },
  urlInput: { fontSize: 11 },
  streamStats: { flexDirection: 'row', gap: 10 },
  statBox: {
    flex: 1,
    backgroundColor: COLORS.parchmentInput,
    borderWidth: 1,
    borderColor: COLORS.parchmentBorder,
    padding: 10,
    borderRadius: 10,
    alignItems: 'center',
    gap: 3,
  },
  statLabel: { color: COLORS.inkMuted, fontSize: 11, fontFamily: 'serif' },
  statNumber: { color: COLORS.terracottaDark, fontSize: 18, fontWeight: '800', fontFamily: 'serif', fontVariant: ['tabular-nums'] },
  liveReadingsBox: { gap: 6, borderTopWidth: 1, borderTopColor: COLORS.parchmentBorder, paddingTop: 8 },
});
