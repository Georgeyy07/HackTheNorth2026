import React, { useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Modal } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';

export default function RoadAlignmentModal({
  visible = false,
  isConfirmed = false,
  tiltAngle = 0,
  maxTiltAngle = 20,
  cameraRef = null,
  isStreaming = false,
  framesSent = 0,
  onConfirm = () => {},
  onRealign = () => {},
  onCancel = () => {},
}) {
  const [permission, requestPermission] = useCameraPermissions();
  const isUpright = Number.isFinite(tiltAngle) && tiltAngle <= maxTiltAngle;

  // Automatically request camera permission when alignment modal is shown
  useEffect(() => {
    if (visible && !isConfirmed && (!permission || !permission.granted)) {
      requestPermission().catch(() => {});
    }
  }, [visible, isConfirmed, permission, requestPermission]);

  if (!visible) return null;

  return (
    <>
      {/* 1. Native Full-Screen Modal: Guaranteed to render above maps, fragments, and navigation */}
      <Modal
        visible={Boolean(visible && !isConfirmed)}
        animationType="slide"
        transparent={false}
        statusBarTranslucent
        onRequestClose={onCancel}
      >
        <SafeAreaView style={styles.fullContainer} edges={['top', 'bottom']}>
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.badge}>ROAD CAMERA SETUP</Text>
            <Text style={styles.title}>Align Camera to Road</Text>
            <Text style={styles.instruction}>
              Make sure the camera has a clear, unobstructed view of the road ahead. Keep the phone mounted upright.
            </Text>
          </View>

          {/* Live Camera View with Alignment Guide */}
          <View style={styles.cameraContainer}>
            {permission?.granted ? (
              <CameraView ref={cameraRef} style={styles.camera} facing="back" mode="picture">
                {/* Overlay Crosshairs and Horizon Guide */}
                <View style={styles.overlay} pointerEvents="none">
                  <View style={styles.horizonLine} />
                  <View style={styles.centerTarget}>
                    <View style={styles.centerDot} />
                  </View>
                  <View style={styles.laneGuideLeft} />
                  <View style={styles.laneGuideRight} />
                  <Text style={styles.guideText}>Keep road centered in view</Text>
                </View>
              </CameraView>
            ) : (
              <View style={styles.permissionBox}>
                <Text style={styles.permissionIcon}>📷</Text>
                <Text style={styles.permissionText}>Camera permission needed to preview road view.</Text>
                <TouchableOpacity style={styles.permButton} onPress={() => requestPermission()}>
                  <Text style={styles.permButtonText}>Grant Camera Permission</Text>
                </TouchableOpacity>
              </View>
            )}

            {/* Tilt Status Banner */}
            <View style={[styles.tiltBanner, isUpright ? styles.tiltGood : styles.tiltBad]}>
              <Text style={styles.tiltIcon}>{isUpright ? '✓' : '⚠️'}</Text>
              <Text style={styles.tiltText}>
                Tilt: <Text style={styles.tiltBold}>{tiltAngle.toFixed(1)}°</Text>
                {isUpright ? ' · Upright (OK)' : ` · Exceeds ${maxTiltAngle}° limit!`}
              </Text>
            </View>
          </View>

          {/* Action Controls */}
          <View style={styles.actions}>
            <TouchableOpacity
              style={[styles.confirmButton, !isUpright && styles.confirmDisabled]}
              disabled={!isUpright}
              onPress={onConfirm}
            >
              <Text style={styles.confirmText}>
                {isUpright ? 'Confirm Road View & Start Stream' : `Straighten Phone (<= ${maxTiltAngle}°)`}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.cancelButton} onPress={onCancel}>
              <Text style={styles.cancelText}>Cancel Navigation</Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </Modal>

      {/* 2. Compact Picture-in-Picture (PIP) badge during active navigation once confirmed */}
      {Boolean(visible && isConfirmed) && (
        <View style={styles.pipRoot} pointerEvents="box-none">
          <TouchableOpacity
            activeOpacity={0.85}
            onPress={onRealign}
            style={[styles.pipContainer, !isUpright && styles.pipContainerAlert]}
          >
            {permission?.granted ? (
              <CameraView ref={cameraRef} style={styles.pipCamera} facing="back" mode="picture" />
            ) : (
              <View style={styles.pipFallback}>
                <Text style={{ color: '#fff', fontSize: 10 }}>No Cam</Text>
              </View>
            )}

            {/* Streaming & Tilt Badge */}
            <View style={[styles.pipBadge, isUpright ? styles.pipBadgeLive : styles.pipBadgeAlert]}>
              <Text style={styles.pipBadgeText}>
                {isUpright ? `● ${isStreaming ? '16 FPS' : 'LIVE'}` : '⚠️ >20°'}
              </Text>
            </View>

            <View style={styles.pipFooter}>
              <Text style={styles.pipFooterText}>
                {tiltAngle.toFixed(0)}° · {framesSent > 0 ? `#${framesSent}` : 'Tap to adjust'}
              </Text>
            </View>
          </TouchableOpacity>
        </View>
      )}
    </>
  );
}

const styles = StyleSheet.create({
  fullContainer: {
    flex: 1,
    backgroundColor: '#0b0f19',
    padding: 16,
    justifyContent: 'space-between',
  },
  header: { gap: 6, marginBottom: 12 },
  badge: { color: '#00f2fe', fontSize: 11, fontWeight: '700', letterSpacing: 1 },
  title: { color: '#f3f4f6', fontSize: 22, fontWeight: '700' },
  instruction: { color: '#9ca3af', fontSize: 13, lineHeight: 18 },
  cameraContainer: {
    flex: 1,
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: '#111827',
    position: 'relative',
  },
  camera: { flex: 1, width: '100%' },
  overlay: { ...StyleSheet.absoluteFillObject, justifyContent: 'center', alignItems: 'center' },
  horizonLine: { width: '80%', height: 1.5, backgroundColor: 'rgba(0, 242, 254, 0.4)' },
  centerTarget: {
    position: 'absolute',
    width: 48,
    height: 48,
    borderRadius: 24,
    borderWidth: 2,
    borderColor: 'rgba(0, 242, 254, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  centerDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: '#00f2fe' },
  laneGuideLeft: {
    position: 'absolute',
    bottom: 20,
    left: '20%',
    width: 2,
    height: 80,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    transform: [{ rotate: '25deg' }],
  },
  laneGuideRight: {
    position: 'absolute',
    bottom: 20,
    right: '20%',
    width: 2,
    height: 80,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    transform: [{ rotate: '-25deg' }],
  },
  guideText: { position: 'absolute', bottom: 12, color: 'rgba(255, 255, 255, 0.7)', fontSize: 11, fontWeight: '600' },
  permissionBox: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20, gap: 12 },
  permissionIcon: { fontSize: 40 },
  permissionText: { color: '#f3f4f6', textAlign: 'center', fontSize: 14 },
  permButton: { backgroundColor: '#00f2fe', paddingVertical: 10, paddingHorizontal: 16, borderRadius: 8 },
  permButtonText: { color: '#0b0f19', fontWeight: '700', fontSize: 13 },
  tiltBanner: {
    position: 'absolute',
    top: 12,
    left: 12,
    right: 12,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  tiltGood: { backgroundColor: 'rgba(6, 78, 59, 0.85)', borderWidth: 1, borderColor: '#34d399' },
  tiltBad: { backgroundColor: 'rgba(127, 29, 29, 0.85)', borderWidth: 1, borderColor: '#ef4444' },
  tiltIcon: { fontSize: 14 },
  tiltText: { color: '#f3f4f6', fontSize: 12 },
  tiltBold: { fontWeight: '700', color: '#fff' },
  actions: { gap: 10, marginTop: 14 },
  confirmButton: { backgroundColor: '#00f2fe', paddingVertical: 14, borderRadius: 10, alignItems: 'center' },
  confirmDisabled: { backgroundColor: '#374151', opacity: 0.6 },
  confirmText: { color: '#0b0f19', fontWeight: '700', fontSize: 15 },
  cancelButton: { paddingVertical: 10, alignItems: 'center' },
  cancelText: { color: '#9ca3af', fontSize: 13 },

  // Compact Picture-in-Picture styles
  pipRoot: {
    position: 'absolute',
    bottom: 72,
    right: 14,
    zIndex: 9999,
    elevation: 10,
  },
  pipContainer: {
    width: 120,
    height: 90,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#111827',
    borderWidth: 2,
    borderColor: '#00f2fe',
    elevation: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 6,
  },
  pipContainerAlert: {
    borderColor: '#ef4444',
  },
  pipCamera: {
    flex: 1,
    width: '100%',
  },
  pipFallback: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1f2937',
  },
  pipBadge: {
    position: 'absolute',
    top: 4,
    left: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  pipBadgeLive: {
    backgroundColor: 'rgba(5, 150, 105, 0.85)',
  },
  pipBadgeAlert: {
    backgroundColor: 'rgba(220, 38, 38, 0.9)',
  },
  pipBadgeText: {
    color: '#ffffff',
    fontSize: 9,
    fontWeight: '700',
  },
  pipFooter: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(11, 15, 25, 0.75)',
    paddingVertical: 2,
    paddingHorizontal: 4,
    alignItems: 'center',
  },
  pipFooterText: {
    color: '#d1d5db',
    fontSize: 9,
    fontWeight: '600',
  },
});
