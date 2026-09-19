import React, { useRef, useEffect } from 'react';
import { StyleSheet, View, Text, Platform, TouchableOpacity } from 'react-native';
import { Pothole } from '../types/pothole';
import { COLORS, SEVERITY_COLORS } from '../constants/theme';

let MapView: any = null;
let Marker: any = null;
let Callout: any = null;

if (Platform.OS !== 'web') {
  try {
    const Maps = require('react-native-maps');
    MapView = Maps.default;
    Marker = Maps.Marker;
    Callout = Maps.Callout;
  } catch (e) {
    console.warn('react-native-maps could not be loaded:', e);
  }
}

interface PotholeMapProps {
  potholes: Pothole[];
  selectedPothole: Pothole | null;
  onSelectPothole: (pothole: Pothole | null) => void;
  onMovePothole: (id: number, latitude: number, longitude: number) => void;
  onMapPress: (coordinate: { latitude: number; longitude: number }) => void;
  onDeletePothole: (id: number) => void;
  onEditPothole: (pothole: Pothole) => void;
}

export const PotholeMap: React.FC<PotholeMapProps> = ({
  potholes,
  selectedPothole,
  onSelectPothole,
  onMovePothole,
  onMapPress,
  onDeletePothole,
  onEditPothole,
}) => {
  const mapRef = useRef<any>(null);
  const safePotholes = Array.isArray(potholes) ? potholes : [];

  // Default region: Waterloo / Larisa or first pothole
  const initialRegion = {
    latitude: safePotholes.length > 0 ? safePotholes[0].latitude : 43.4723,
    longitude: safePotholes.length > 0 ? safePotholes[0].longitude : -80.5449,
    latitudeDelta: 0.04,
    longitudeDelta: 0.04,
  };

  useEffect(() => {
    if (selectedPothole && mapRef.current) {
      mapRef.current.animateToRegion(
        {
          latitude: selectedPothole.latitude,
          longitude: selectedPothole.longitude,
          latitudeDelta: 0.015,
          longitudeDelta: 0.015,
        },
        500
      );
    }
  }, [selectedPothole]);

  if (Platform.OS === 'web' || !MapView) {
    return (
      <View style={styles.webFallbackContainer}>
        <Text style={styles.webTitle}>🗺️ Interactive Pothole Map</Text>
        <Text style={styles.webSubtitle}>
          {safePotholes.length} hazards recorded in Tiger Data database.
        </Text>
        <View style={styles.webPinGrid}>
          {safePotholes.map((ph) => {
            const sev = SEVERITY_COLORS[ph.severity];
            return (
              <TouchableOpacity
                key={ph.id}
                style={[
                  styles.webPinCard,
                  selectedPothole?.id === ph.id && styles.webPinCardSelected,
                ]}
                onPress={() => onSelectPothole(ph)}
              >
                <View style={[styles.badgeDot, { backgroundColor: sev.pin }]} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.webPinTitle}>
                    #{ph.id} · {ph.severity}
                  </Text>
                  <Text style={styles.webPinCoords}>
                    {ph.latitude.toFixed(4)}, {ph.longitude.toFixed(4)}
                  </Text>
                </View>
                <TouchableOpacity
                  onPress={() => onEditPothole(ph)}
                  style={styles.actionIconBtn}
                >
                  <Text style={{ fontSize: 13 }}>✏️</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => onDeletePothole(ph.id)}
                  style={styles.actionIconBtn}
                >
                  <Text style={{ fontSize: 13 }}>🗑️</Text>
                </TouchableOpacity>
              </TouchableOpacity>
            );
          })}
        </View>
        <Text style={styles.webHint}>
          💡 Run in Expo Go on iOS / Android to use native Apple Maps & Google Maps with drag-and-drop pin moving.
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <MapView
        ref={mapRef}
        style={styles.map}
        initialRegion={initialRegion}
        onPress={(e: any) => {
          if (e.nativeEvent.coordinate) {
            onMapPress(e.nativeEvent.coordinate);
          }
        }}
      >
        {safePotholes.map((pothole) => {
          const colors = SEVERITY_COLORS[pothole.severity];
          const isSelected = selectedPothole?.id === pothole.id;

          return (
            <Marker
              key={pothole.id}
              coordinate={{
                latitude: pothole.latitude,
                longitude: pothole.longitude,
              }}
              draggable
              onDragEnd={(e: any) => {
                const { latitude, longitude } = e.nativeEvent.coordinate;
                onMovePothole(pothole.id, latitude, longitude);
              }}
              onPress={() => onSelectPothole(pothole)}
            >
              <View
                style={[
                  styles.markerHalo,
                  { borderColor: colors.pin },
                  isSelected && styles.markerHaloSelected,
                ]}
              >
                <View style={[styles.markerCore, { backgroundColor: colors.pin }]} />
              </View>

              <Callout tooltip>
                <View style={styles.calloutCard}>
                  <View style={styles.calloutHeader}>
                    <Text style={[styles.calloutBadge, { backgroundColor: colors.pin }]}>
                      {pothole.severity}
                    </Text>
                    <Text style={styles.calloutId}>Tiger DB #{pothole.id}</Text>
                  </View>
                  <Text style={styles.calloutCoords}>
                    📍 {pothole.latitude.toFixed(4)}, {pothole.longitude.toFixed(4)}
                  </Text>
                  <Text style={styles.calloutTip}>Drag marker to move location</Text>
                  <View style={styles.calloutActions}>
                    <TouchableOpacity
                      style={[styles.calloutBtn, { backgroundColor: '#1b4435' }]}
                      onPress={() => onEditPothole(pothole)}
                    >
                      <Text style={styles.calloutBtnText}>Edit</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.calloutBtn, { backgroundColor: COLORS.coral }]}
                      onPress={() => onDeletePothole(pothole.id)}
                    >
                      <Text style={[styles.calloutBtnText, { color: '#fff' }]}>Delete</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              </Callout>
            </Marker>
          );
        })}
      </MapView>

      <View style={styles.mapTipOverlay}>
        <Text style={styles.mapTipText}>
          💡 Tap map to report · Long press / drag pin to move
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.bgDark,
  },
  map: {
    width: '100%',
    height: '100%',
  },
  markerHalo: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    borderWidth: 2.5,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.5,
    shadowRadius: 5,
    elevation: 6,
  },
  markerHaloSelected: {
    transform: [{ scale: 1.25 }],
    borderColor: COLORS.cyan,
  },
  markerCore: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  calloutCard: {
    width: 200,
    padding: 12,
    backgroundColor: COLORS.bgCardElevated,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.borderLight,
    shadowColor: '#000',
    shadowOpacity: 0.4,
    shadowRadius: 8,
  },
  calloutHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  calloutBadge: {
    fontSize: 9,
    fontWeight: '700',
    color: '#fff',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    textTransform: 'uppercase',
  },
  calloutId: {
    fontSize: 10,
    color: COLORS.cyan,
    fontWeight: '600',
  },
  calloutCoords: {
    fontSize: 10,
    color: COLORS.textMuted,
    marginBottom: 4,
  },
  calloutTip: {
    fontSize: 9,
    color: COLORS.emeraldGlow,
    fontStyle: 'italic',
    marginBottom: 8,
  },
  calloutActions: {
    flexDirection: 'row',
    gap: 8,
  },
  calloutBtn: {
    flex: 1,
    paddingVertical: 5,
    alignItems: 'center',
    borderRadius: 5,
  },
  calloutBtnText: {
    fontSize: 10,
    fontWeight: '600',
    color: COLORS.text,
  },
  mapTipOverlay: {
    position: 'absolute',
    top: 12,
    alignSelf: 'center',
    backgroundColor: 'rgba(8, 18, 14, 0.85)',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  mapTipText: {
    color: COLORS.emeraldGlow,
    fontSize: 11,
    fontWeight: '600',
  },
  // Web fallback styles
  webFallbackContainer: {
    flex: 1,
    padding: 16,
    backgroundColor: COLORS.bgDark,
  },
  webTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: COLORS.text,
    marginBottom: 4,
  },
  webSubtitle: {
    fontSize: 12,
    color: COLORS.textMuted,
    marginBottom: 12,
  },
  webPinGrid: {
    gap: 8,
  },
  webPinCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.bgCard,
    borderRadius: 8,
    padding: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
    gap: 10,
  },
  webPinCardSelected: {
    borderColor: COLORS.cyan,
    backgroundColor: COLORS.bgCardElevated,
  },
  badgeDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  webPinTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: COLORS.text,
  },
  webPinCoords: {
    fontSize: 11,
    color: COLORS.textMuted,
  },
  actionIconBtn: {
    padding: 6,
  },
  webHint: {
    marginTop: 20,
    fontSize: 11,
    color: COLORS.emeraldGlow,
    textAlign: 'center',
    lineHeight: 16,
  },
});
