import React, { forwardRef, useState, createContext, useContext } from 'react';
import { View, StyleSheet, Platform, Text, TouchableOpacity } from 'react-native';
import MapView, {
  Marker as NativeMarker,
  Polyline as NativePolyline,
  Callout as NativeCallout,
} from 'react-native-maps';
import { COLORS } from '../theme';
import { CompassRose, HazardPin } from './Icons';

export const ParchmentMapContext = createContext({
  initialRegion: { latitude: 43.4723, longitude: -80.5449, latitudeDelta: 0.05, longitudeDelta: 0.05 },
  mapSize: { width: 360, height: 260 },
});

// Universal Marker: Native uses react-native-maps; Web calculates coordinates on illustrated map
export const Marker =
  Platform.OS === 'web'
    ? function WebMarker({ coordinate, title, children, onPress, style }) {
        const { initialRegion } = useContext(ParchmentMapContext);
        if (!coordinate || coordinate.latitude == null || coordinate.longitude == null) {
          return null;
        }
        const centerLat = initialRegion?.latitude ?? 43.4723;
        const centerLon = initialRegion?.longitude ?? -80.5449;
        const latDelta = initialRegion?.latitudeDelta ?? 0.05;
        const lonDelta = initialRegion?.longitudeDelta ?? 0.05;

        // In map projection, higher latitude is toward top, higher longitude is toward right
        const topPct = Math.min(94, Math.max(6, ((centerLat + latDelta / 2 - coordinate.latitude) / latDelta) * 100));
        const leftPct = Math.min(94, Math.max(6, ((coordinate.longitude - (centerLon - lonDelta / 2)) / lonDelta) * 100));

        return (
          <TouchableOpacity
            activeOpacity={0.8}
            onPress={onPress}
            style={[
              {
                position: 'absolute',
                top: `${topPct}%`,
                left: `${leftPct}%`,
                transform: [{ translateX: -14 }, { translateY: -14 }],
                zIndex: 15,
              },
              style,
            ]}
          >
            {children || <HazardPin severity="MEDIUM" />}
          </TouchableOpacity>
        );
      }
    : NativeMarker;

export const Polyline =
  Platform.OS === 'web'
    ? function WebPolyline() {
        return null;
      }
    : NativePolyline;

export const Callout =
  Platform.OS === 'web'
    ? function WebCallout({ children }) {
        return <View>{children}</View>;
      }
    : NativeCallout;

/**
 * Universal Parchment Map:
 * - On Native (Android / iOS): Renders native MapView with 100% width/height and custom style.
 * - On Web (where react-native-maps is unsupported): Renders an illustrated parchment map matching the user's reference image!
 * - Supports interactive onPress to tap and place markers on both Web and Native.
 */
export const ParchmentMap = forwardRef(function ParchmentMap(
  {
    children,
    style,
    customMapStyle,
    initialRegion = { latitude: 43.4723, longitude: -80.5449, latitudeDelta: 0.05, longitudeDelta: 0.05 },
    routes = [],
    selectedIndex = 0,
    origin,
    destination,
    potholes = [],
    onPress,
    ...props
  },
  ref
) {
  const [mapSize, setMapSize] = useState({ width: 360, height: 260 });

  // If a fixed height or relative positioning is passed, adapt wrapper style
  const flattenedStyle = StyleSheet.flatten(style) || {};
  const isEmbedded = Boolean(flattenedStyle.height || flattenedStyle.maxHeight);

  if (Platform.OS !== 'web') {
    return (
      <View
        style={[
          styles.mapWrapper,
          isEmbedded && styles.embeddedWrapper,
          style,
        ]}
      >
        <MapView
          ref={ref}
          provider={Platform.OS === 'android' ? 'google' : undefined}
          customMapStyle={Platform.OS === 'android' ? customMapStyle : undefined}
          style={styles.nativeMap}
          initialRegion={initialRegion}
          onPress={onPress}
          {...props}
        >
          {children}
        </MapView>
      </View>
    );
  }

  // Handle tap on web map to compute lat/lon
  const handleWebPress = (e) => {
    if (!onPress) return;
    const { locationX = 180, locationY = 130 } = e.nativeEvent || {};
    const w = mapSize.width || 360;
    const h = mapSize.height || 260;
    const centerLat = initialRegion?.latitude ?? 43.4723;
    const centerLon = initialRegion?.longitude ?? -80.5449;
    const latDelta = initialRegion?.latitudeDelta ?? 0.05;
    const lonDelta = initialRegion?.longitudeDelta ?? 0.05;

    const fracX = Math.min(1, Math.max(0, locationX / w));
    const fracY = Math.min(1, Math.max(0, locationY / h));

    const latitude = Number((centerLat + (0.5 - fracY) * latDelta).toFixed(5));
    const longitude = Number((centerLon + (fracX - 0.5) * lonDelta).toFixed(5));

    onPress({
      nativeEvent: {
        coordinate: { latitude, longitude },
      },
    });
  };

  // Web Illustrated Storybook Map (matching user reference image)
  return (
    <ParchmentMapContext.Provider value={{ initialRegion, mapSize }}>
      <View
        style={[
          styles.mapWrapper,
          isEmbedded && styles.embeddedWrapper,
          style,
        ]}
        onLayout={(e) => {
          const { width, height } = e.nativeEvent.layout;
          if (width && height) {
            setMapSize({ width, height });
          }
        }}
      >
        <TouchableOpacity
          activeOpacity={1}
          onPress={handleWebPress}
          style={styles.webMapContainer}
        >
          {/* Decorative river curves */}
          <View style={styles.webRiverMain} />
          <View style={styles.webRiverTributary} />

          {/* Decorative pine tree clusters */}
          <View style={[styles.pineCluster, { top: '15%', left: '22%' }]}>
            <Text style={styles.treeGlyph}>▲▲</Text>
            <Text style={[styles.treeGlyph, { marginTop: -4 }]}>▲▲▲</Text>
          </View>
          <View style={[styles.pineCluster, { top: '38%', left: '8%' }]}>
            <Text style={styles.treeGlyph}>▲</Text>
            <Text style={[styles.treeGlyph, { marginTop: -4 }]}>▲▲</Text>
          </View>
          <View style={[styles.pineCluster, { top: '48%', right: '14%' }]}>
            <Text style={styles.treeGlyph}>▲▲</Text>
            <Text style={[styles.treeGlyph, { marginTop: -4 }]}>▲▲▲</Text>
          </View>
          <View style={[styles.pineCluster, { top: '80%', left: '18%' }]}>
            <Text style={styles.treeGlyph}>▲▲</Text>
          </View>

          {/* Road network grid */}
          <View style={styles.webRoadHorizontal1} />
          <View style={styles.webRoadHorizontal2} />
          <View style={styles.webRoadVertical1} />

          {/* Dashed Terracotta Route Path (if routes provided and not embedded) */}
          {routes.length > 0 && !isEmbedded && (
            <View style={styles.webRoutePathWrapper}>
              <View style={styles.webRouteDash1} />
              <View style={styles.webRouteDash2} />
              <View style={styles.webRouteDash3} />
              <View style={styles.webRouteDash4} />
              <View style={styles.webRouteDash5} />
            </View>
          )}

          {/* Compass Rose */}
          <View style={[styles.webCompassPosition, isEmbedded && styles.webCompassEmbedded]}>
            <CompassRose size={isEmbedded ? 34 : 44} />
          </View>

          {/* Render children: Markers, Polylines, Pins */}
          {children}
        </TouchableOpacity>
      </View>
    </ParchmentMapContext.Provider>
  );
});

const styles = StyleSheet.create({
  mapWrapper: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    width: '100%',
    height: '100%',
    backgroundColor: COLORS.parchmentBg,
  },
  embeddedWrapper: {
    position: 'relative',
    top: undefined,
    left: undefined,
    right: undefined,
    bottom: undefined,
  },
  nativeMap: {
    width: '100%',
    height: '100%',
  },
  webMapContainer: {
    width: '100%',
    height: '100%',
    position: 'relative',
    overflow: 'hidden',
  },

  // Rivers
  webRiverMain: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: '42%',
    width: 20,
    backgroundColor: '#b0d1c7',
    transform: [{ rotate: '12deg' }],
    borderLeftWidth: 1.5,
    borderRightWidth: 1.5,
    borderColor: '#86a89f',
  },
  webRiverTributary: {
    position: 'absolute',
    top: '30%',
    left: 0,
    right: '50%',
    height: 12,
    backgroundColor: '#b0d1c7',
    transform: [{ rotate: '-18deg' }],
    borderTopWidth: 1.5,
    borderBottomWidth: 1.5,
    borderColor: '#86a89f',
  },

  // Roads
  webRoadHorizontal1: {
    position: 'absolute',
    top: '25%',
    left: 0,
    right: 0,
    height: 6,
    backgroundColor: '#e6d3b4',
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: '#cdb592',
  },
  webRoadHorizontal2: {
    position: 'absolute',
    top: '60%',
    left: 0,
    right: 0,
    height: 8,
    backgroundColor: '#e6d3b4',
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: '#cdb592',
  },
  webRoadVertical1: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    right: '32%',
    width: 6,
    backgroundColor: '#e6d3b4',
    borderLeftWidth: 1,
    borderRightWidth: 1,
    borderColor: '#cdb592',
  },

  // Dashed terracotta route line
  webRoutePathWrapper: {
    position: 'absolute',
    top: '20%',
    bottom: '28%',
    left: '52%',
    width: 8,
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  webRouteDash1: { width: 5, height: 16, backgroundColor: COLORS.terracotta, borderRadius: 2 },
  webRouteDash2: { width: 5, height: 16, backgroundColor: COLORS.terracotta, borderRadius: 2, transform: [{ rotate: '-15deg' }] },
  webRouteDash3: { width: 5, height: 16, backgroundColor: COLORS.terracotta, borderRadius: 2, transform: [{ rotate: '10deg' }] },
  webRouteDash4: { width: 5, height: 16, backgroundColor: COLORS.terracotta, borderRadius: 2, transform: [{ rotate: '-10deg' }] },
  webRouteDash5: { width: 5, height: 16, backgroundColor: COLORS.terracotta, borderRadius: 2 },

  // Trees
  pineCluster: {
    position: 'absolute',
    alignItems: 'center',
    opacity: 0.75,
  },
  treeGlyph: {
    fontSize: 12,
    color: '#3d6338',
    letterSpacing: -2,
    fontWeight: '800',
  },

  webCompassPosition: {
    position: 'absolute',
    top: '12%',
    right: '12%',
    zIndex: 5,
  },
  webCompassEmbedded: {
    top: '8%',
    right: '8%',
  },
});
