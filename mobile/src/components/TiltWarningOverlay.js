import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function TiltWarningOverlay({
  tiltAngle,
  isTilted,
  threshold = 20,
  sensorAvailable = true,
  simulatedAngle = null,
  onSimulateTilt = () => { },
  isNavigating = false,
  onToggleNavigating = () => { },
}) {
  // If phone is tilted beyond threshold (20°), render red alert overlay
  return (
    <View style={styles.rootContainer} pointerEvents="box-none">
      {/* Full-screen red alert overlay when tilted */}
      {isTilted && (
        <View style={styles.redTintOverlay} pointerEvents="none">
          <SafeAreaView
            style={[styles.safeArea, !isNavigating && styles.safeAreaBottom]}
            edges={isNavigating ? ['top'] : ['bottom']}
            pointerEvents="none"
          >
            <View style={styles.alertCard} pointerEvents="none">
              <View style={styles.alertHeader}>
                <Text style={styles.warningIcon}>⚠️</Text>
                <View style={styles.alertHeaderTextGroup}>
                  <Text style={styles.alertTitle}>STRAIGHTEN PHONE </Text>
                  <Text style={styles.alertSubtitle}>
                    Current: <Text style={styles.angleHighlight}>{tiltAngle.toFixed(1)}°</Text> · Limit: {threshold}°
                  </Text>
                </View>
              </View>

              {/* Angle gauge visual bar */}
              <View style={styles.gaugeContainer}>
                <View style={styles.gaugeTrack}>
                  {/* 20 degree limit tick line */}
                  <View style={[styles.gaugeLimitMarker, { left: `${(threshold / 90) * 100}%` }]} />
                  {/* Current fill bar */}
                  <View
                    style={[
                      styles.gaugeFill,
                      { width: `${Math.min(100, (tiltAngle / 90) * 100)}%` },
                    ]}
                  />
                </View>
                <View style={styles.gaugeLabels}>
                  <Text style={styles.gaugeLabelText}>0° (Straight Up)</Text>
                  <Text style={[styles.gaugeLabelText, styles.gaugeLimitText]}>{threshold}° Limit</Text>
                  <Text style={styles.gaugeLabelText}>90° (Flat)</Text>
                </View>
              </View>

              <Text style={styles.instructionText}>
                {isNavigating
                  ? 'Please straighten your phone in the mount to continue recording road motion accurately. IMU WebSocket streaming is paused while tilted.'
                  : 'Phone is tilted > 20°. Vibration is paused while typing/browsing and activates during navigation.'}
              </Text>
            </View>
          </SafeAreaView>
        </View>
      )}

      {/* Dev / Simulator test toggle button (useful on web & desktop simulator without physical sensors) */}
      <View style={styles.devContainer} pointerEvents="box-none">
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  rootContainer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 9999,
  },
  redTintOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(220, 38, 38, 0.42)', // Red wash over UI
    borderWidth: 5,
    borderColor: '#ef4444',
  },
  safeArea: {
    paddingHorizontal: 16,
    paddingTop: 10,
  },
  safeAreaBottom: {
    position: 'absolute',
    bottom: 66, // Sits comfortably above bottom tab bar without covering top address inputs
    left: 0,
    right: 0,
    paddingBottom: 4,
  },
  alertCard: {
    backgroundColor: 'rgba(127, 29, 29, 0.95)', // Deep crimson card
    borderColor: '#f87171',
    borderWidth: 2,
    borderRadius: 14,
    padding: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  alertHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  warningIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  alertHeaderTextGroup: {
    flex: 1,
  },
  alertTitle: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '900',
    letterSpacing: 0.5,
  },
  alertSubtitle: {
    color: '#fecaca',
    fontSize: 11,
    marginTop: 2,
  },
  angleHighlight: {
    color: '#ffffff',
    fontWeight: '800',
  },
  liveBadge: {
    backgroundColor: '#dc2626',
    borderColor: '#ffffff',
    borderWidth: 1,
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  mutedBadge: {
    backgroundColor: '#4b5563',
    borderColor: '#9ca3af',
  },
  liveBadgeText: {
    color: '#ffffff',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  gaugeContainer: {
    marginVertical: 4,
  },
  gaugeTrack: {
    height: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    borderRadius: 4,
    overflow: 'hidden',
    position: 'relative',
  },
  gaugeFill: {
    height: '100%',
    backgroundColor: '#ef4444',
    borderRadius: 4,
  },
  gaugeLimitMarker: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: '#ffffff',
    zIndex: 2,
  },
  gaugeLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 3,
  },
  gaugeLabelText: {
    color: '#fecaca',
    fontSize: 9,
  },
  gaugeLimitText: {
    color: '#ffffff',
    fontWeight: '700',
  },
  instructionText: {
    color: '#fee2e2',
    fontSize: 11,
    lineHeight: 15,
    marginTop: 4,
    fontStyle: 'italic',
  },
  devContainer: {
    position: 'absolute',
    bottom: 64, // Positioned just above the bottom tab bar
    right: 12,
    alignItems: 'flex-end',
  },
  devBadge: {
    backgroundColor: 'rgba(40, 70, 39, 0.9)',
    borderColor: '#6f966a',
    borderWidth: 1,
    borderRadius: 20,
    paddingHorizontal: 10,
    paddingVertical: 5,
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  devBadgeAlert: {
    backgroundColor: 'rgba(185, 28, 28, 0.95)',
    borderColor: '#fca5a5',
  },
  devBadgeText: {
    color: '#ffffff',
    fontSize: 11,
    fontWeight: '700',
    fontVariant: ['tabular-nums'],
  },
  devPanel: {
    backgroundColor: 'rgba(27, 38, 26, 0.95)',
    borderColor: '#6f966a',
    borderWidth: 1.5,
    borderRadius: 12,
    padding: 12,
    width: 280,
    shadowColor: '#000',
    shadowOpacity: 0.4,
    shadowRadius: 6,
    elevation: 6,
  },
  devHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  devTitle: {
    color: '#f6efdc',
    fontSize: 12,
    fontWeight: '700',
  },
  devClose: {
    color: '#d8c5aa',
    fontSize: 14,
    fontWeight: '700',
    padding: 2,
  },
  devSubtitle: {
    color: '#beaa8d',
    fontSize: 10,
    marginBottom: 8,
  },
  navToggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#3b6138',
  },
  navToggleLabel: {
    color: '#f6efdc',
    fontSize: 10,
  },
  navToggleBtn: {
    backgroundColor: '#374151',
    borderColor: '#6b7280',
    borderWidth: 1,
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  navToggleBtnActive: {
    backgroundColor: '#059669',
    borderColor: '#34d399',
  },
  navToggleBtnText: {
    color: '#9ca3af',
    fontSize: 10,
    fontWeight: '700',
  },
  navToggleBtnTextActive: {
    color: '#ffffff',
    fontWeight: '800',
  },
  presetRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  presetBtn: {
    backgroundColor: '#284627',
    borderColor: '#537d4f',
    borderWidth: 1,
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 5,
  },
  presetBtnActive: {
    backgroundColor: '#df6e35',
    borderColor: '#ea864c',
  },
  presetBtnText: {
    color: '#f6efdc',
    fontSize: 10,
    fontWeight: '600',
  },
  presetBtnTextActive: {
    color: '#ffffff',
    fontWeight: '800',
  },
});
