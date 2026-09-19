import React, { useState } from 'react';
import { View, Text, TouchableOpacity, TextInput, ScrollView, StyleSheet, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as Sharing from 'expo-sharing';
import { MOUNTS, mountMatrix } from '../motion/vehicle';
import { useMotionRecorder } from '../motion/useMotionRecorder';

const value = (v) => Number.isFinite(v) ? v.toFixed(3) : '—';

function Button({ children, onPress, disabled = false }) {
  return <TouchableOpacity accessibilityRole="button" disabled={disabled} onPress={onPress}
    style={[styles.button, disabled && styles.disabled]}><Text style={styles.buttonText}>{children}</Text></TouchableOpacity>;
}

export default function MotionScreen() {
  const recorder = useMotionRecorder();
  const [mount, setMount] = useState('upright');
  const [angles, setAngles] = useState(['0', '0', '0']);
  const active = recorder.status !== 'idle';
  const sample = recorder.live?.sample;

  const start = () => {
    try {
      if (angles.some((a) => !a.trim())) throw new Error('Enter all three correction angles.');
      recorder.start(mountMatrix(mount, ...angles.map(Number)));
    } catch (err) { Alert.alert('Mount settings', err.message); }
  };

  const share = async (file) => {
    try {
      if (!await Sharing.isAvailableAsync()) throw new Error('File sharing is unavailable on this device.');
      await Sharing.shareAsync(file.uri, { mimeType: 'application/x-ndjson', UTI: 'public.plain-text', dialogTitle: 'Export raw + VQF recording' });
    } catch (err) { Alert.alert('Export recording', err.message); }
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
});
