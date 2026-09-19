import React from 'react';
import { TouchableOpacity, Text, StyleSheet, View } from 'react-native';

export const COLORS = {
  // Warm storybook parchment palette (matching reference image)
  parchmentBg: '#f6efdc',
  parchmentSurface: '#fbf5e6',
  parchmentCard: '#fcf7ea',
  parchmentInput: '#fcf8ee',
  parchmentBorder: '#d8c5aa',
  parchmentBorderDark: '#beaa8d',
  parchmentCrease: '#9e8568',

  // Handcrafted sepia ink typography
  inkPrimary: '#3d2816',
  inkSecondary: '#664e37',
  inkMuted: '#937d67',
  inkCursive: '#66361a',

  // Honey apricot wood & terracotta palette (matching reference image)
  terracotta: '#df6e35',
  terracottaDark: '#964114',
  terracottaLight: '#ea864c',
  woodBevelTop: '#f79f5f',
  woodBevelBottom: '#6e2b08',
  woodOutline: '#964114',
  woodText: '#2a1408',

  // Deep forest accents
  forestDark: '#284627',
  forestPine: '#3b6138',
  forestMoss: '#537d4f',
  forestFern: '#6f966a',
  forestTint: '#e1ecd9',

  // Mineral hazard stamps
  hazardCritical: '#b83824',
  hazardHigh: '#d4652c',
  hazardMedium: '#d88b32',
  hazardLow: '#4c7847',
};

/**
 * Pure dynamic Honey Wood Button:
 * Handcrafted with warm caramel apricot layering, carved bevel borders,
 * subtle wood grain highlights, and zero photographic artifacts or external edges.
 * Perfectly matches "Start Drive" in the reference image.
 */
export function WoodButton({
  onPress,
  children,
  style,
  textStyle,
  disabled = false,
  small = false,
  variant = 'default', // 'default', 'danger', 'forest'
}) {
  const isDanger = variant === 'danger';
  const isForest = variant === 'forest';

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled}
      activeOpacity={0.85}
      style={[
        styles.woodPillOuter,
        small && styles.woodPillSmall,
        isDanger && styles.outerDanger,
        isForest && styles.outerForest,
        disabled && styles.disabledPill,
        style,
      ]}
    >
      <View
        style={[
          styles.woodBody,
          small && styles.woodBodySmall,
          isDanger && styles.bodyDanger,
          isForest && styles.bodyForest,
        ]}
      >
        {/* Subtle wood grain accent lines */}
        <View style={styles.woodGrainLineTop} />
        <View style={styles.woodGrainLineBottom} />

        <View style={styles.woodInnerContent}>
          {typeof children === 'string' ? (
            <Text
              style={[
                styles.woodText,
                small && styles.woodTextSmall,
                isDanger && styles.textDanger,
                textStyle,
              ]}
            >
              {children}
            </Text>
          ) : (
            children
          )}
        </View>
      </View>
    </TouchableOpacity>
  );
}

// Google Maps custom style: warm antique parchment & soft storybook forest
export const MAP_STYLE_PARCHMENT = [
  {
    elementType: 'geometry',
    stylers: [{ color: '#f5ecd8' }],
  },
  {
    elementType: 'labels.text.fill',
    stylers: [{ color: '#4a3622' }],
  },
  {
    elementType: 'labels.text.stroke',
    stylers: [{ color: '#fcf7ea' }, { weight: 3 }],
  },
  {
    featureType: 'administrative.land_parcel',
    elementType: 'geometry.stroke',
    stylers: [{ color: '#decdb2' }],
  },
  {
    featureType: 'landscape.natural',
    elementType: 'geometry',
    stylers: [{ color: '#ebe0c4' }],
  },
  {
    featureType: 'poi',
    elementType: 'geometry',
    stylers: [{ color: '#dfd1b3' }],
  },
  {
    featureType: 'poi.park',
    elementType: 'geometry',
    stylers: [{ color: '#d2e4cb' }],
  },
  {
    featureType: 'poi.park',
    elementType: 'labels.text.fill',
    stylers: [{ color: '#385e35' }],
  },
  {
    featureType: 'road',
    elementType: 'geometry',
    stylers: [{ color: '#efe6d2' }],
  },
  {
    featureType: 'road',
    elementType: 'geometry.stroke',
    stylers: [{ color: '#d5bea1' }, { weight: 1.5 }],
  },
  {
    featureType: 'road.arterial',
    elementType: 'geometry',
    stylers: [{ color: '#e5d1ae' }],
  },
  {
    featureType: 'road.highway',
    elementType: 'geometry',
    stylers: [{ color: '#deb886' }],
  },
  {
    featureType: 'road.highway',
    elementType: 'geometry.stroke',
    stylers: [{ color: '#ab824e' }, { weight: 2 }],
  },
  {
    featureType: 'water',
    elementType: 'geometry',
    stylers: [{ color: '#a5c9be' }],
  },
  {
    featureType: 'water',
    elementType: 'labels.text.fill',
    stylers: [{ color: '#32544a' }],
  },
];

const styles = StyleSheet.create({
  woodPillOuter: {
    backgroundColor: '#964114',
    padding: 2.5,
    borderRadius: 28,
    borderTopColor: '#f79f5f',
    borderBottomColor: '#6e2b08',
    borderWidth: 1.5,
  },
  woodPillSmall: {
    borderRadius: 18,
    padding: 1.5,
    borderWidth: 1,
  },
  woodBody: {
    backgroundColor: '#df6e35',
    borderRadius: 25,
    paddingVertical: 13,
    paddingHorizontal: 28,
    alignItems: 'center',
    justifyContent: 'center',
    borderTopWidth: 1.5,
    borderTopColor: '#fca76a',
    borderBottomWidth: 1.5,
    borderBottomColor: '#b8511d',
    position: 'relative',
    overflow: 'hidden',
  },
  woodBodySmall: {
    borderRadius: 16,
    paddingVertical: 7,
    paddingHorizontal: 16,
    borderTopWidth: 1,
    borderBottomWidth: 1,
  },
  woodGrainLineTop: {
    position: 'absolute',
    top: 5,
    left: '14%',
    right: '20%',
    height: 1.5,
    backgroundColor: 'rgba(255, 255, 255, 0.25)',
    borderRadius: 1,
  },
  woodGrainLineBottom: {
    position: 'absolute',
    bottom: 5,
    left: '22%',
    right: '14%',
    height: 1.5,
    backgroundColor: 'rgba(100, 35, 10, 0.22)',
    borderRadius: 1,
  },
  woodInnerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  woodText: {
    color: '#2a1408',
    fontSize: 17,
    fontWeight: '800',
    fontFamily: 'serif',
    letterSpacing: 0.3,
  },
  woodTextSmall: {
    fontSize: 12,
  },
  outerDanger: {
    backgroundColor: '#781d13',
    borderTopColor: '#ea6655',
    borderBottomColor: '#450d06',
  },
  bodyDanger: {
    backgroundColor: '#c03827',
    borderTopColor: '#e86a59',
    borderBottomColor: '#962314',
  },
  textDanger: {
    color: '#fff',
  },
  outerForest: {
    backgroundColor: '#1b3d22',
    borderTopColor: '#629e6c',
    borderBottomColor: '#0f2414',
  },
  bodyForest: {
    backgroundColor: '#2f5e38',
    borderTopColor: '#538f5f',
    borderBottomColor: '#204327',
  },
  disabledPill: {
    opacity: 0.5,
  },
});
