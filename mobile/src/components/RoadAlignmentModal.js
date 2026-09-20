import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Modal } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { COLORS, WoodButton } from '../theme';

export default function RoadAlignmentModal({
  visible = false,
  isConfirmed = false,
  tiltAngle = 0,
  maxTiltAngle = 20,
  cameraRef = null,
  isStreaming = false,
  framesSent = 0,
  onConfirm = () => { },
  onRealign = () => { },
  onCancel = () => { },
}) {
  const [permission, requestPermission] = useCameraPermissions();
  const [showPip, setShowPip] = useState(false); // Defaults to OFF as requested
  const isUpright = Number.isFinite(tiltAngle) && tiltAngle <= maxTiltAngle;

  // Reset toggle to OFF whenever navigation ends
  useEffect(() => {
    if (!visible) {
      setShowPip(false);
    }
  }, [visible]);

  // Automatically request camera permission when alignment modal is shown
  useEffect(() => {
    if (visible && !isConfirmed && (!permission || !permission.granted)) {
      requestPermission().catch(() => { });
    }
  }, [visible, isConfirmed, permission, requestPermission]);

  if (!visible) return null;

  return (
    <>
      {/* 1. Native Full-Screen Modal: Warm storybook parchment & handcrafted wood theme */}
      <Modal
        visible={Boolean(visible && !isConfirmed)}
        animationType="slide"
        transparent={false}
        statusBarTranslucent
        onRequestClose={onCancel}
      >
        <SafeAreaView style={styles.fullContainer} edges={['top', 'bottom']}>
          {/* Header Card in Parchment Surface */}
          <View style={styles.header}>
            <Text style={styles.badge}>ROAD CAMERA SETUP</Text>
            <Text style={styles.title}>Align Camera to Road</Text>
            <Text style={styles.instruction}>
              Make sure the camera has a clear, unobstructed view of the road ahead. Keep your phone mounted upright.
            </Text>
          </View>

          {/* Live Camera View with Vintage Alignment Guide */}
          <View style={styles.cameraContainer}>
            {permission?.granted ? (
              <CameraView ref={cameraRef} style={styles.camera} facing="back" mode="picture">
                {/* Overlay Reticle and Horizon Guide */}
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
                <WoodButton onPress={() => requestPermission()} small style={{ marginTop: 8 }}>
                  Grant Camera Permission
                </WoodButton>
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

          {/* Action Controls in Handcrafted Wood */}
          <View style={styles.actions}>
            <WoodButton
              onPress={onConfirm}
              disabled={!isUpright}
              style={styles.confirmWoodBtn}
            >
              {isUpright ? 'Confirm Road View & Start Stream' : `Straighten Phone (<= ${maxTiltAngle}°)`}
            </WoodButton>

            <TouchableOpacity style={styles.cancelButton} onPress={onCancel}>
              <Text style={styles.cancelText}>Cancel Navigation</Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </Modal>

      {/* 2. Compact Picture-in-Picture (PIP) badge during active navigation once confirmed */}
      {Boolean(visible && isConfirmed) && (
        <View style={styles.pipRoot} pointerEvents="box-none">
          {/* Default: OFF. Shows small paper toggle button to turn preview on if desired */}
          {!showPip ? (
            <>
              {/* Keep camera mounted in background so 15+fps stream continues */}
              <View style={styles.hiddenCameraContainer} pointerEvents="none">
                {permission?.granted && (
                  <CameraView ref={cameraRef} style={styles.hiddenCamera} facing="back" mode="picture" />
                )}
              </View>

              <TouchableOpacity
                style={[styles.pipToggleBtn, !isUpright && styles.pipToggleBtnAlert]}
                onPress={() => setShowPip(true)}
                activeOpacity={0.8}
                accessibilityLabel="Show road camera preview"
              >
                <Text style={styles.pipToggleIcon}>📷</Text>
                <Text style={styles.pipToggleText}>Cam</Text>
                {isStreaming && isUpright && <View style={styles.pipStreamingDot} />}
              </TouchableOpacity>
            </>
          ) : (
            /* Toggled ON: Floating paper-styled camera preview card with close button */
            <View style={[styles.pipContainer, !isUpright && styles.pipContainerAlert]}>
              <TouchableOpacity activeOpacity={0.9} onPress={onRealign} style={{ flex: 1 }}>
                {permission?.granted ? (
                  <CameraView ref={cameraRef} style={styles.pipCamera} facing="back" mode="picture" />
                ) : (
                  <View style={styles.pipFallback}>
                    <Text style={{ color: COLORS.inkPrimary, fontSize: 11, fontFamily: 'serif' }}>No Cam</Text>
                  </View>
                )}

                <View style={styles.pipFooter}>
                  <Text style={styles.pipFooterText}>
                    {tiltAngle.toFixed(0)}° · {framesSent > 0 ? `#${framesSent}` : 'Tap to adjust'}
                  </Text>
                </View>
              </TouchableOpacity>

              {/* Close/minimize button to toggle back off */}
              <TouchableOpacity
                style={styles.pipCloseBtn}
                onPress={() => setShowPip(false)}
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                accessibilityLabel="Hide road camera preview"
              >
                <Text style={styles.pipCloseText}>✕</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      )}
    </>
  );
}

const styles = StyleSheet.create({
  fullContainer: {
    flex: 1,
    backgroundColor: COLORS.parchmentBg, // #f6efdc storybook background
    padding: 16,
    justifyContent: 'space-between',
  },
  header: {
    backgroundColor: COLORS.parchmentSurface, // #fbf5e6
    borderColor: COLORS.parchmentBorderDark,  // #beaa8d
    borderWidth: 1.5,
    borderRadius: 12,
    padding: 14,
    gap: 4,
    marginBottom: 12,
    shadowColor: COLORS.inkPrimary,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 3,
  },
  badge: {
    color: COLORS.forestPine, // #3b6138 deep forest green
    fontSize: 11,
    fontFamily: 'serif',
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  title: {
    color: COLORS.inkPrimary, // #3d2816 sepia ink
    fontSize: 22,
    fontFamily: 'serif',
    fontWeight: '700',
  },
  instruction: {
    color: COLORS.inkSecondary, // #664e37
    fontSize: 13,
    fontFamily: 'serif',
    lineHeight: 18,
    marginTop: 2,
  },
  cameraContainer: {
    flex: 1,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#1b1713', // vintage dark lens frame
    borderWidth: 2,
    borderColor: COLORS.parchmentBorderDark,
    position: 'relative',
    shadowColor: COLORS.inkPrimary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 6,
    elevation: 6,
  },
  camera: { flex: 1, width: '100%' },
  overlay: { ...StyleSheet.absoluteFillObject, justifyContent: 'center', alignItems: 'center' },
  horizonLine: {
    width: '80%',
    height: 2,
    backgroundColor: 'rgba(223, 110, 53, 0.75)', // warm terracotta horizon
  },
  centerTarget: {
    position: 'absolute',
    width: 48,
    height: 48,
    borderRadius: 24,
    borderWidth: 2,
    borderColor: 'rgba(223, 110, 53, 0.85)', // warm terracotta ring
    justifyContent: 'center',
    alignItems: 'center',
  },
  centerDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.terracotta,
  },
  laneGuideLeft: {
    position: 'absolute',
    bottom: 20,
    left: '20%',
    width: 2,
    height: 80,
    backgroundColor: 'rgba(251, 245, 230, 0.55)',
    transform: [{ rotate: '25deg' }],
  },
  laneGuideRight: {
    position: 'absolute',
    bottom: 20,
    right: '20%',
    width: 2,
    height: 80,
    backgroundColor: 'rgba(251, 245, 230, 0.55)',
    transform: [{ rotate: '-25deg' }],
  },
  guideText: {
    position: 'absolute',
    bottom: 12,
    color: '#fcf8ee',
    fontSize: 11,
    fontFamily: 'serif',
    fontWeight: '700',
    letterSpacing: 0.3,
    textShadowColor: 'rgba(61, 40, 22, 0.8)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  permissionBox: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    gap: 10,
    backgroundColor: COLORS.parchmentSurface,
  },
  permissionIcon: { fontSize: 38 },
  permissionText: {
    color: COLORS.inkPrimary,
    textAlign: 'center',
    fontSize: 14,
    fontFamily: 'serif',
    lineHeight: 20,
  },
  tiltBanner: {
    position: 'absolute',
    top: 12,
    left: 12,
    right: 12,
    paddingVertical: 9,
    paddingHorizontal: 14,
    borderRadius: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    shadowColor: COLORS.inkPrimary,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.18,
    shadowRadius: 4,
    elevation: 4,
  },
  tiltGood: {
    backgroundColor: 'rgba(244, 251, 240, 0.96)', // pale forest parchment tint
    borderWidth: 1.5,
    borderColor: COLORS.forestPine,
  },
  tiltBad: {
    backgroundColor: 'rgba(254, 242, 242, 0.96)', // soft crimson hazard wash
    borderWidth: 1.5,
    borderColor: COLORS.hazardCritical,
  },
  tiltIcon: { fontSize: 14 },
  tiltText: {
    color: COLORS.inkPrimary,
    fontSize: 12,
    fontFamily: 'serif',
  },
  tiltBold: {
    fontWeight: '800',
    color: COLORS.inkPrimary,
  },
  actions: { gap: 10, marginTop: 14 },
  confirmWoodBtn: {
    width: '100%',
  },
  cancelButton: {
    paddingVertical: 10,
    alignItems: 'center',
  },
  cancelText: {
    color: COLORS.inkMuted,
    fontSize: 13,
    fontFamily: 'serif',
    fontWeight: '600',
    textDecorationLine: 'underline',
  },

  // =========================================================================
  // CAMERA TOGGLE & PICTURE-IN-PICTURE (PIP) POSITIONING & PAPER STYLING
  // =========================================================================
  // >>> ADJUST VERTICAL/HORIZONTAL POSITION HERE: <<<
  pipRoot: {
    position: 'absolute',
    bottom: 150, // <-- Adjust vertical height from bottom of screen here (default: 150)
    right: 16,   // <-- Adjust horizontal positioning from right edge here (default: 16)
    zIndex: 9999,
    elevation: 10,
  },
  hiddenCameraContainer: {
    position: 'absolute',
    left: -9999,
    width: 1,
    height: 1,
    overflow: 'hidden',
  },
  hiddenCamera: {
    width: 1,
    height: 1,
    opacity: 0.01,
  },
  // Paper-styled floating toggle pill: [ 📷 Cam ]
  pipToggleBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.parchmentSurface, // warm parchment paper #fbf5e6
    borderColor: COLORS.parchmentBorderDark,  // handcrafted sepia border #beaa8d
    borderWidth: 1.5,
    borderRadius: 20,
    paddingVertical: 7,
    paddingHorizontal: 12,
    gap: 6,
    elevation: 6,
    shadowColor: COLORS.inkPrimary,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.25,
    shadowRadius: 5,
  },
  pipToggleBtnAlert: {
    borderColor: COLORS.hazardCritical, // crimson hazard #b83824
    backgroundColor: '#fee2e2',
  },
  pipToggleIcon: {
    fontSize: 13,
  },
  pipToggleText: {
    color: COLORS.inkPrimary, // sepia ink #3d2816
    fontSize: 12,
    fontFamily: 'serif',
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  pipStreamingDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.forestPine, // forest green #3b6138
    marginLeft: 2,
  },
  // Paper-styled floating camera card
  pipContainer: {
    width: 124,
    height: 96,
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: COLORS.parchmentSurface,
    borderWidth: 2,
    borderColor: COLORS.parchmentBorderDark,
    elevation: 8,
    shadowColor: COLORS.inkPrimary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.28,
    shadowRadius: 6,
  },
  pipContainerAlert: {
    borderColor: COLORS.hazardCritical,
  },
  pipCamera: {
    flex: 1,
    width: '100%',
  },
  pipFallback: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.parchmentCard,
  },
  pipCloseBtn: {
    position: 'absolute',
    top: 4,
    right: 4,
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: COLORS.parchmentSurface,
    borderColor: COLORS.parchmentBorderDark,
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
    shadowColor: COLORS.inkPrimary,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  pipCloseText: {
    color: COLORS.inkPrimary,
    fontSize: 10,
    fontFamily: 'serif',
    fontWeight: '800',
    lineHeight: 12,
  },
  pipFooter: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(251, 245, 230, 0.94)', // translucent parchment overlay
    borderTopWidth: 1,
    borderTopColor: COLORS.parchmentBorder,
    paddingVertical: 3,
    paddingHorizontal: 4,
    alignItems: 'center',
  },
  pipFooterText: {
    color: COLORS.inkPrimary,
    fontSize: 10,
    fontFamily: 'serif',
    fontWeight: '700',
    letterSpacing: 0.2,
  },
});
