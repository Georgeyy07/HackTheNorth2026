import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { COLORS } from '../theme';

/**
 * Clean vector geometric icons matching the reference image's storybook style.
 * 1. Zero emojis
 * 2. Crisp, responsive, high-contrast cartographic glyphs
 */

export function SearchIcon({ size = 18, color = COLORS.terracotta }) {
  return (
    <View style={{ width: size, height: size, justifyContent: 'center', alignItems: 'center' }}>
      <View
        style={{
          width: size - 6,
          height: size - 6,
          borderRadius: (size - 6) / 2,
          borderWidth: 2,
          borderColor: color,
        }}
      />
      <View
        style={{
          position: 'absolute',
          right: 1,
          bottom: 1,
          width: 5,
          height: 2,
          backgroundColor: color,
          transform: [{ rotate: '45deg' }],
        }}
      />
    </View>
  );
}

export function PlusCircleIcon({ size = 20, color = COLORS.inkSecondary }) {
  return (
    <View
      style={{
        width: size,
        height: size,
        borderRadius: size / 2,
        borderWidth: 1.5,
        borderColor: color,
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <View style={{ width: 8, height: 1.5, backgroundColor: color }} />
      <View style={{ position: 'absolute', width: 1.5, height: 8, backgroundColor: color }} />
    </View>
  );
}

export function CrosshairIcon({ size = 20, color = COLORS.inkPrimary }) {
  return (
    <View style={{ width: size, height: size, justifyContent: 'center', alignItems: 'center' }}>
      <View
        style={{
          width: size - 4,
          height: size - 4,
          borderRadius: (size - 4) / 2,
          borderWidth: 2,
          borderColor: color,
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <View style={{ width: 4, height: 4, borderRadius: 2, backgroundColor: color }} />
      </View>
      <View style={{ position: 'absolute', top: 0, width: 2, height: 4, backgroundColor: color }} />
      <View style={{ position: 'absolute', bottom: 0, width: 2, height: 4, backgroundColor: color }} />
      <View style={{ position: 'absolute', left: 0, width: 4, height: 2, backgroundColor: color }} />
      <View style={{ position: 'absolute', right: 0, width: 4, height: 2, backgroundColor: color }} />
    </View>
  );
}

export function LayersIcon({ size = 18, color = COLORS.inkPrimary }) {
  return (
    <View style={{ width: size, height: size, justifyContent: 'center', alignItems: 'center' }}>
      <View
        style={{
          width: size - 4,
          height: 6,
          borderWidth: 1.5,
          borderColor: color,
          borderRadius: 1.5,
          marginBottom: 3,
        }}
      />
      <View
        style={{
          width: size - 4,
          height: 6,
          borderWidth: 1.5,
          borderColor: color,
          borderRadius: 1.5,
        }}
      />
    </View>
  );
}

export function SwapIcon({ size = 18, color = COLORS.inkSecondary }) {
  return (
    <View style={{ width: size, height: size, justifyContent: 'center', alignItems: 'center', flexDirection: 'row', gap: 4 }}>
      <View style={{ alignItems: 'center' }}>
        <View
          style={{
            width: 0,
            height: 0,
            borderLeftWidth: 3,
            borderRightWidth: 3,
            borderBottomWidth: 4,
            borderLeftColor: 'transparent',
            borderRightColor: 'transparent',
            borderBottomColor: color,
          }}
        />
        <View style={{ width: 1.5, height: 7, backgroundColor: color }} />
      </View>
      <View style={{ alignItems: 'center' }}>
        <View style={{ width: 1.5, height: 7, backgroundColor: color }} />
        <View
          style={{
            width: 0,
            height: 0,
            borderLeftWidth: 3,
            borderRightWidth: 3,
            borderTopWidth: 4,
            borderLeftColor: 'transparent',
            borderRightColor: 'transparent',
            borderTopColor: color,
          }}
        />
      </View>
    </View>
  );
}

export function CloseIcon({ size = 14, color = COLORS.inkPrimary }) {
  return (
    <View style={{ width: size, height: size, justifyContent: 'center', alignItems: 'center' }}>
      <View
        style={{
          position: 'absolute',
          width: size,
          height: 2,
          backgroundColor: color,
          transform: [{ rotate: '45deg' }],
        }}
      />
      <View
        style={{
          position: 'absolute',
          width: size,
          height: 2,
          backgroundColor: color,
          transform: [{ rotate: '-45deg' }],
        }}
      />
    </View>
  );
}

// Illustrated Teardrop Pin matching reference image
export function HazardPin({ severity = 'MEDIUM', count }) {
  let bg = COLORS.terracotta;
  if (severity === 'CRITICAL') bg = COLORS.hazardCritical;
  else if (severity === 'LOW') bg = COLORS.forestPine;

  return (
    <View style={pinStyles.container}>
      <View style={[pinStyles.badge, { backgroundColor: bg }]}>
        <View style={pinStyles.innerRing}>
          <Text style={pinStyles.badgeText}>{count !== undefined ? count : '!'}</Text>
        </View>
      </View>
      <View style={[pinStyles.point, { borderTopColor: bg }]} />
    </View>
  );
}

// Active Tapped Marker Pin for adding/editing potholes
export function TargetMarkerPin({ severity = 'MEDIUM', label = '+' }) {
  let bg = COLORS.terracotta;
  if (severity === 'CRITICAL') bg = COLORS.hazardCritical;
  else if (severity === 'HIGH') bg = COLORS.hazardHigh;
  else if (severity === 'LOW') bg = COLORS.forestPine;

  return (
    <View style={pinStyles.targetContainer}>
      <View style={[pinStyles.targetPulseRing, { borderColor: bg }]}>
        <View style={[pinStyles.targetBadge, { backgroundColor: bg }]}>
          <Text style={pinStyles.targetBadgeText}>{label}</Text>
        </View>
      </View>
      <View style={[pinStyles.point, { borderTopColor: bg }]} />
    </View>
  );
}

// Illustrated Storybook Compass Rose matching reference image
export function CompassRose({ size = 46 }) {
  return (
    <View
      style={{
        width: size,
        height: size,
        borderRadius: size / 2,
        backgroundColor: '#703c1e',
        borderWidth: 3,
        borderColor: '#43683f',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <View
        style={{
          width: size - 12,
          height: size - 12,
          borderRadius: (size - 12) / 2,
          borderWidth: 1.5,
          borderColor: '#e8dcbe',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <View style={{ position: 'absolute', width: 2, height: size - 16, backgroundColor: '#e8dcbe' }} />
        <View style={{ position: 'absolute', height: 2, width: size - 16, backgroundColor: '#e8dcbe' }} />
        <View
          style={{
            width: 6,
            height: 6,
            borderRadius: 3,
            backgroundColor: '#e8dcbe',
          }}
        />
      </View>
    </View>
  );
}

const pinStyles = StyleSheet.create({
  container: {
    alignItems: 'center',
  },
  badge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 2,
    borderColor: '#422110',
    justifyContent: 'center',
    alignItems: 'center',
  },
  innerRing: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(255, 240, 220, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  badgeText: {
    color: '#fff8ee',
    fontWeight: '800',
    fontSize: 10,
    fontFamily: 'serif',
  },
  point: {
    width: 0,
    height: 0,
    borderLeftWidth: 5,
    borderRightWidth: 5,
    borderTopWidth: 7,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    marginTop: -1,
  },
  targetContainer: {
    alignItems: 'center',
  },
  targetPulseRing: {
    padding: 3,
    borderRadius: 20,
    borderWidth: 2,
    borderStyle: 'dashed',
    backgroundColor: 'rgba(255, 248, 238, 0.5)',
  },
  targetBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 2,
    borderColor: '#422110',
    justifyContent: 'center',
    alignItems: 'center',
  },
  targetBadgeText: {
    color: '#fff8ee',
    fontWeight: '900',
    fontSize: 14,
    fontFamily: 'serif',
  },
});
