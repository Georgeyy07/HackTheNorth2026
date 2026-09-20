import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from 'react-native';
import Slider from '@react-native-community/slider';
import * as Location from 'expo-location';
import * as Notifications from 'expo-notifications';

import { getRoute, getPotholes } from '../api';
import { TurnByTurnTracker } from '../navigation';
import AddressInput from '../components/AddressInput';
import { COLORS, WoodButton, MAP_STYLE_PARCHMENT } from '../theme';
import {
  SearchIcon,
  PlusCircleIcon,
  CrosshairIcon,
  SwapIcon,
  HazardPin,
  CompassRose,
} from '../components/Icons';
import { ParchmentMap, Marker, Polyline } from '../components/ParchmentMap';
import { useNavigationStatus } from '../context/NavigationContext';

function toLatLng(coords) {
  return coords.map(([lat, lon]) => ({ latitude: lat, longitude: lon }));
}

async function announce(title, body) {
  await Notifications.scheduleNotificationAsync({ content: { title, body }, trigger: null });
}

const MAX_POTHOLE_CARE = 15;

export default function RouteFinderScreen() {
  const [origin, setOrigin] = useState('University of Waterloo, Waterloo, ON');
  const [destination, setDestination] = useState('Waterloo Public Square, Waterloo, ON');
  const [originCoords, setOriginCoords] = useState({ lat: 43.4723, lon: -80.5449 });
  const [destCoords, setDestCoords] = useState({ lat: 43.4623, lon: -80.5224 });

  // Map tap selection mode: null | 'origin' | 'destination'
  const [mapPickTarget, setMapPickTarget] = useState(null);
  const [topSectionHeight, setTopSectionHeight] = useState(60);

  const [potholeCare, setPotholeCare] = useState(6);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Route planner modal/editor state
  const [isEditingRoute, setIsEditingRoute] = useState(false);

  // Potholes in database
  const [allPotholes, setAllPotholes] = useState([]);

  // Navigation mode (shared with app-level tilt monitor)
  const { isNavigating: navigating, setIsNavigating: setNavigating } = useNavigationStatus();
  const [currentInstruction, setCurrentInstruction] = useState(null);
  const [nextTurnDistance, setNextTurnDistance] = useState(null);
  const [userLocation, setUserLocation] = useState(null);

  const trackerRef = useRef(null);
  const subscriptionRef = useRef(null);
  const mapRef = useRef(null);

  // Tap map to set Departure or Arrival point directly on the map
  const handleMapPress = (e) => {
    const coord = e.nativeEvent?.coordinate;
    if (!coord) return;
    const lat = Number(coord.latitude.toFixed(5));
    const lon = Number(coord.longitude.toFixed(5));

    if (mapPickTarget === 'origin') {
      setOriginCoords({ lat, lon });
      setOrigin(`${lat.toFixed(4)}, ${lon.toFixed(4)}`);
      // Auto advance to destination for smooth 2-tap route setup
      setMapPickTarget('destination');
    } else if (mapPickTarget === 'destination') {
      setDestCoords({ lat, lon });
      setDestination(`${lat.toFixed(4)}, ${lon.toFixed(4)}`);
      setMapPickTarget(null);
    } else {
      // Default: set destination point when tapped
      setDestCoords({ lat, lon });
      setDestination(`${lat.toFixed(4)}, ${lon.toFixed(4)}`);
    }
  };

  useEffect(() => {
    getPotholes()
      .then((data) => setAllPotholes(data || []))
      .catch(() => { });
    // Initial fetch to populate default route
    handleFindRoutes();
  }, []);

  const handleSwapAddresses = () => {
    const tempO = origin;
    const tempOCoords = originCoords;
    setOrigin(destination);
    setOriginCoords(destCoords);
    setDestination(tempO);
    setDestCoords(tempOCoords);
  };

  const handleFindRoutes = async () => {
    if (!origin.trim() || !destination.trim()) {
      setError('Please provide origin and destination.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await getRoute(
        origin.trim(),
        destination.trim(),
        potholeCare,
        originCoords,
        destCoords
      );
      // Ensure 'Recommended' is always first in the list of routes
      const sortedRoutes = [...(data.routes || [])].sort((a, b) => {
        if (a.label === 'Recommended') return -1;
        if (b.label === 'Recommended') return 1;
        return 0;
      });
      data.routes = sortedRoutes;
      setResult(data);
      if (data.origin) {
        setOriginCoords({ lat: data.origin.lat, lon: data.origin.lon });
      }
      if (data.destination) {
        setDestCoords({ lat: data.destination.lat, lon: data.destination.lon });
      }
      setSelectedIndex(0);
      setIsEditingRoute(false);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // Re-fit map when routes arrive
  useEffect(() => {
    if (result && result.routes.length > 0 && mapRef.current?.fitToCoordinates) {
      const allCoords = result.routes.flatMap((r) => toLatLng(r.coords));
      mapRef.current.fitToCoordinates(allCoords, {
        edgePadding: { top: 140, right: 50, bottom: 260, left: 50 },
        animated: true,
      });
    }
  }, [result]);

  const recenterMap = () => {
    if (userLocation && mapRef.current?.animateToRegion) {
      mapRef.current.animateToRegion({
        latitude: userLocation.latitude,
        longitude: userLocation.longitude,
        latitudeDelta: 0.02,
        longitudeDelta: 0.02,
      });
    } else if (result?.origin && mapRef.current?.animateToRegion) {
      mapRef.current.animateToRegion({
        latitude: result.origin.lat,
        longitude: result.origin.lon,
        latitudeDelta: 0.03,
        longitudeDelta: 0.03,
      });
    }
  };

  const stopNavigation = useCallback(() => {
    subscriptionRef.current?.remove();
    subscriptionRef.current = null;
    trackerRef.current = null;
    setNavigating(false);
    setCurrentInstruction(null);
    setNextTurnDistance(null);
  }, []);

  const startNavigation = async () => {
    const route = result?.routes[selectedIndex];
    if (!route) return;

    const [locationPerm, notificationPerm] = await Promise.all([
      Location.requestForegroundPermissionsAsync(),
      Notifications.requestPermissionsAsync(),
    ]);

    if (locationPerm.status !== 'granted') {
      setError('Location permission is required to navigate.');
      return;
    }
    if (notificationPerm.status !== 'granted') {
      setError('Notification permission denied: turn alerts will show in-app only.');
    }

    trackerRef.current = new TurnByTurnTracker(route.directions);
    setNavigating(true);
    setIsEditingRoute(false);
    setCurrentInstruction(trackerRef.current.currentStep?.instruction || 'Proceed to route');

    subscriptionRef.current = await Location.watchPositionAsync(
      { accuracy: Location.Accuracy.High, timeInterval: 2000, distanceInterval: 5 },
      (position) => {
        const { latitude, longitude } = position.coords;
        setUserLocation({ latitude, longitude });

        if (mapRef.current?.animateToRegion) {
          mapRef.current.animateToRegion({
            latitude,
            longitude,
            latitudeDelta: 0.008,
            longitudeDelta: 0.008,
          });
        }

        const tracker = trackerRef.current;
        if (!tracker) return;
        const event = tracker.update(latitude, longitude);
        if (!event) return;

        if (event.type === 'turn_ahead') {
          const dist = Math.round(event.distance);
          setNextTurnDistance(dist);
          const msg = `${event.step.instruction} in ${dist}m`;
          setCurrentInstruction(msg);
          announce('Upcoming turn', msg);
        } else if (event.type === 'turn_now') {
          setNextTurnDistance(0);
          setCurrentInstruction(event.step.instruction);
          announce('Turn now', event.step.instruction);
        } else if (event.type === 'arrived') {
          setCurrentInstruction('You have arrived');
          announce('Arrived', 'You have reached your destination');
          stopNavigation();
        }
      }
    );
  };

  const initialRegion = {
    latitude: result?.origin.lat ?? 43.4723,
    longitude: result?.origin.lon ?? -80.5449,
    latitudeDelta: 0.05,
    longitudeDelta: 0.05,
  };

  const selectedRoute = result?.routes[selectedIndex];
  const selectedRoutePotholes = selectedRoute?.potholes_encountered || [];
  const riskPercentage = Math.min(100, Math.round((potholeCare / MAX_POTHOLE_CARE) * 100));

  // Destination display label
  const destinationShort = destination.split(',')[0].trim() || 'destination';

  return (
    <View style={styles.container}>
      {/* UNIVERSAL PARCHMENT MAP */}
      <ParchmentMap
        ref={mapRef}
        customMapStyle={MAP_STYLE_PARCHMENT}
        initialRegion={initialRegion}
        routes={result?.routes}
        selectedIndex={selectedIndex}
        origin={originCoords || result?.origin}
        destination={destCoords || result?.destination}
        potholes={selectedRoutePotholes.length > 0 ? selectedRoutePotholes : allPotholes}
        onPress={handleMapPress}
      >
        {/* Dashed Terracotta Route Polyline */}
        {result &&
          result.routes.map((route, i) => {
            const isSelected = i === selectedIndex;
            return (
              <Polyline
                key={`route-${route.label}-${i}`}
                coordinates={toLatLng(route.coords)}
                strokeColor={isSelected ? COLORS.terracotta : '#bba282'}
                strokeWidth={isSelected ? 6 : 4}
                lineDashPattern={isSelected ? [8, 5] : undefined}
                zIndex={isSelected ? 10 : 5}
              />
            );
          })}

        {/* Departure Marker A (Always rendered when coordinate exists) */}
        {originCoords && (
          <Marker
            coordinate={{ latitude: originCoords.lat, longitude: originCoords.lon }}
            title="Departure (A)"
          >
            <View style={styles.originMarker}>
              <Text style={styles.markerText}>A</Text>
            </View>
          </Marker>
        )}

        {/* Arrival Marker B (Always rendered when coordinate exists) */}
        {destCoords && (
          <Marker
            coordinate={{ latitude: destCoords.lat, longitude: destCoords.lon }}
            title="Arrival (B)"
          >
            <View style={styles.destMarker}>
              <Text style={styles.markerText}>B</Text>
            </View>
          </Marker>
        )}

        {/* Route pothole pins */}
        {selectedRoutePotholes.map((p) => (
          <Marker
            key={`route-pothole-${p.id}`}
            coordinate={{ latitude: p.lat, longitude: p.lon }}
            title={`${p.severity} Pothole`}
          >
            <HazardPin severity={p.severity} />
          </Marker>
        ))}

        {/* User GPS location marker */}
        {userLocation && (
          <Marker coordinate={userLocation} title="You">
            <View style={styles.userLocationMarker}>
              <View style={styles.userLocationDot} />
            </View>
          </Marker>
        )}
      </ParchmentMap>

      {/* FLOATING HUD & CONTROLS OVERLAY */}
      <SafeAreaView pointerEvents="box-none" style={styles.safeOverlay}>
        {/* TOP SECTION: SEARCH PILL OR EXPANDED ROUTE PLANNER */}
        {!navigating ? (
          <View
            style={styles.topSection}
            onLayout={(e) => {
              const h = e.nativeEvent?.layout?.height;
              if (h && h > 0) setTopSectionHeight(h);
            }}
          >
            {!isEditingRoute ? (
              /* COLLAPSED SEARCH PILL: "to [destination]" WITH EDIT (+) BUTTON */
              <TouchableOpacity
                style={styles.searchPillBar}
                activeOpacity={0.88}
                onPress={() => setIsEditingRoute(true)}
              >
                <SearchIcon size={20} color={COLORS.terracotta} />
                <Text style={styles.searchPillText} numberOfLines={1}>
                  to {destinationShort}
                </Text>
              </TouchableOpacity>
            ) : (
              /* EXPANDED ROUTE PLANNER: INPUTS -> SLIDER -> FIND ROUTES BUTTON */
              <View style={styles.routePlannerCard}>
                <View style={styles.plannerHeaderRow}>
                  <Text style={styles.plannerTitle}>Route Planner</Text>
                  <TouchableOpacity
                    style={styles.closePlannerBtn}
                    onPress={() => {
                      setIsEditingRoute(false);
                      setMapPickTarget(null);
                    }}
                  >
                    <Text style={styles.closePlannerText}>Done</Text>
                  </TouchableOpacity>
                </View>

                <View style={styles.inputsRow}>
                  <View style={styles.inputsColumn}>
                    <AddressInput
                      placeholder="Origin address..."
                      value={origin}
                      onChangeText={(text) => {
                        setOrigin(text);
                        setOriginCoords(null);
                      }}
                      onSelectAddress={(item) => {
                        if (item && item.lat != null && item.lon != null) {
                          setOriginCoords({ lat: item.lat, lon: item.lon });
                        } else {
                          setOriginCoords(null);
                        }
                      }}
                    />
                    <AddressInput
                      placeholder="Destination address..."
                      value={destination}
                      onChangeText={(text) => {
                        setDestination(text);
                        setDestCoords(null);
                      }}
                      onSelectAddress={(item) => {
                        if (item && item.lat != null && item.lon != null) {
                          setDestCoords({ lat: item.lat, lon: item.lon });
                        } else {
                          setDestCoords(null);
                        }
                      }}
                    />
                  </View>
                  <TouchableOpacity style={styles.swapBtn} onPress={handleSwapAddresses}>
                    <SwapIcon size={18} color={COLORS.inkPrimary} />
                  </TouchableOpacity>
                </View>

                {/* TAP-ON-MAP SHORTCUT SELECTORS */}
                <View style={styles.mapPickSelectorRow}>
                  <Text style={styles.mapPickSelectorLabel}>Pick on map:</Text>
                  <TouchableOpacity
                    style={[
                      styles.mapPickChip,
                      mapPickTarget === 'origin' && styles.mapPickChipActive,
                    ]}
                    onPress={() =>
                      setMapPickTarget(mapPickTarget === 'origin' ? null : 'origin')
                    }
                  >
                    <Text
                      style={[
                        styles.mapPickChipText,
                        mapPickTarget === 'origin' && styles.mapPickChipTextActive,
                      ]}
                    >
                      Departure (A)
                    </Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[
                      styles.mapPickChip,
                      mapPickTarget === 'destination' && styles.mapPickChipActive,
                    ]}
                    onPress={() =>
                      setMapPickTarget(
                        mapPickTarget === 'destination' ? null : 'destination'
                      )
                    }
                  >
                    <Text
                      style={[
                        styles.mapPickChipText,
                        mapPickTarget === 'destination' && styles.mapPickChipTextActive,
                      ]}
                    >
                      Arrival (B)
                    </Text>
                  </TouchableOpacity>
                </View>

                {/* SLIDER SECTION INSIDE ROUTE PLANNER */}
                <View style={styles.plannerSliderSection}>
                  <View style={styles.sliderHeaderLine}>
                    <Text style={styles.riskLabelSmall}>Pothole Avoidance Level:</Text>
                    <Text style={styles.riskPercentLarge}>{riskPercentage}%</Text>
                  </View>

                  <Slider
                    style={styles.sliderTrackControl}
                    minimumValue={0}
                    maximumValue={MAX_POTHOLE_CARE}
                    step={1}
                    value={potholeCare}
                    onValueChange={setPotholeCare}
                    minimumTrackTintColor={COLORS.terracotta}
                    maximumTrackTintColor="#c7b396"
                    thumbTintColor={COLORS.terracotta}
                  />
                  <View style={styles.sliderTickLabels}>
                    <Text style={styles.sliderTickText}>Fastest</Text>
                    <Text style={styles.sliderTickText}>Cautious</Text>
                    <Text style={styles.sliderTickText}>Avoid All</Text>
                  </View>
                </View>

                {error && (
                  <View style={styles.errorBox}>
                    <Text style={styles.errorText}>{error}</Text>
                  </View>
                )}

                {/* FIND ROUTES BUTTON DIRECTLY NEAR THE SLIDER */}
                <WoodButton
                  onPress={handleFindRoutes}
                  disabled={loading}
                  style={styles.findRoutesBtn}
                >
                  {loading ? 'Finding Routes...' : 'Find Routes'}
                </WoodButton>
              </View>
            )}
          </View>
        ) : (
          /* TURN-BY-TURN NAVIGATION HUD */
          <View style={styles.navigationHud}>
            <View style={styles.navTurnIconCircle}>
              <View style={styles.navTurnTriangle} />
            </View>
            <View style={styles.navInstructionBlock}>
              {nextTurnDistance !== null && (
                <Text style={styles.navDistanceText}>{nextTurnDistance} meters</Text>
              )}
              <Text style={styles.navInstructionText} numberOfLines={2}>
                {currentInstruction}
              </Text>
            </View>
          </View>
        )
        }

        {/* MAP PICK ACTIVE FLOATING BANNER */}
        {
          mapPickTarget && !navigating && (
            <View
              style={[
                styles.mapPickFloatingBanner,
                { top: topSectionHeight + 12 },
              ]}
            >
              <CrosshairIcon size={14} color={COLORS.terracotta} />
              <Text style={styles.mapPickFloatingBannerText}>
                {mapPickTarget === 'origin'
                  ? 'Tap map to set Departure (A)'
                  : 'Tap map to set Arrival (B)'}
              </Text>
              <TouchableOpacity
                onPress={() => setMapPickTarget(null)}
                style={styles.mapPickCancelBtn}
              >
                <CloseIcon size={12} color={COLORS.inkMuted} />
              </TouchableOpacity>
            </View>
          )
        }

        {/* FLOATING RECENTER CROSSHAIR BUTTON (POSITIONED RIGHT BELOW ROUTE PLANNER, NEVER OVERLAPPING) */}
        {
          !navigating && (
            <View
              style={[
                styles.rightFloatingControls,
                { top: topSectionHeight + 12 },
              ]}
            >
              <TouchableOpacity
                style={styles.recenterFab}
                onPress={recenterMap}
                accessibilityLabel="Recenter to your location or route"
              >
                <CrosshairIcon size={22} color={COLORS.terracotta} />
              </TouchableOpacity>
            </View>
          )
        }

        {/* BOTTOM SECTION: ROUTE OPTIONS SELECTION -> START DRIVE */}
        {
          !navigating && result ? (
            <View style={styles.bottomSection}>
              <View style={styles.routesChoiceCard}>
                <Text style={styles.routesChoiceHeader}>Select Route Option</Text>

                {/* ROUTE CHOICES LIST (USER CAN CHOOSE ONE) */}
                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  style={styles.routeChoicesScroll}
                >
                  {result.routes.map((route, i) => {
                    const isSelected = i === selectedIndex;
                    const durationMins = Math.max(1, Math.round(route.duration_s / 60));
                    const km = (route.distance_m / 1000).toFixed(1);
                    const hazardCount = route.potholes_encountered.length;

                    return (
                      <TouchableOpacity
                        key={`route-opt-${route.label}-${i}`}
                        style={[
                          styles.routeOptionTile,
                          isSelected && styles.routeOptionTileSelected,
                        ]}
                        onPress={() => setSelectedIndex(i)}
                      >
                        <View style={styles.tileHeader}>
                          <Text
                            style={[
                              styles.tileLabel,
                              isSelected && styles.tileLabelSelected,
                            ]}
                          >
                            {{ 'Recommended': 'Best' }[route.label] || route.label}
                          </Text>
                          <View
                            style={[
                              styles.hazardBadge,
                              hazardCount > 0 ? styles.hazardBadgeWarn : styles.hazardBadgeSafe,
                            ]}
                          >
                            <Text style={styles.hazardBadgeText}>
                              {hazardCount === 0 ? 'Clear' : `${hazardCount} Pothole${hazardCount > 1 ? 's' : ''}`}
                            </Text>
                          </View>
                        </View>

                        <Text style={styles.tileDuration}>
                          {durationMins} min
                        </Text>
                        <Text style={styles.tileDistance}>
                          {km} km · {route.directions.length} turns
                        </Text>
                      </TouchableOpacity>
                    );
                  })}
                </ScrollView>

                {/* SELECTED ROUTE DETAILS (MATCHING REFERENCE IMAGE) */}
                <View style={styles.selectedRouteSummaryRow}>
                  <View style={styles.cardCircleIcon}>
                    <View style={styles.cardIconSpoke} />
                    <Text style={styles.cardIconGlyph}>✿</Text>
                  </View>
                  <View style={styles.selectedSummaryTextCol}>
                    <Text style={styles.summaryTitle}>
                      {selectedRoute?.label}: {Math.max(1, Math.round((selectedRoute?.duration_s || 0) / 60))} min
                      ({((selectedRoute?.distance_m || 0) / 1000).toFixed(1)} km)
                    </Text>
                    <Text style={styles.summarySub}>
                      {selectedRoutePotholes.length === 0
                        ? 'Smoothest Pavement · No Potholes Encountered'
                        : `${selectedRoutePotholes.length} Pothole Hazard(s) on this path`}
                    </Text>
                  </View>
                </View>
              </View>

              {/* PURE DYNAMIC HONEY WOOD BUTTON: START DRIVE */}
              <WoodButton onPress={startNavigation} style={styles.startDriveBtn}>
                Start Drive
              </WoodButton>
            </View>
          ) : (
            navigating && (
              <View style={styles.navActiveBottomBar}>
                <View style={styles.navActiveSummary}>
                  <Text style={styles.navActiveTime}>
                    ETA {Math.max(1, Math.round((selectedRoute?.duration_s || 0) / 60))} min
                  </Text>
                  <Text style={styles.navActiveSub}>
                    {((selectedRoute?.distance_m || 0) / 1000).toFixed(1)} km remaining
                  </Text>
                </View>
                <WoodButton
                  onPress={stopNavigation}
                  variant="danger"
                  small
                  style={styles.stopNavButton}
                >
                  Stop
                </WoodButton>
              </View>
            )
          )
        }
      </SafeAreaView >
    </View >
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.parchmentBg,
  },
  safeOverlay: {
    flex: 1,
    justifyContent: 'space-between',
  },

  topSection: {
    paddingHorizontal: 16,
    paddingTop: 8,
  },

  // Pill Search Bar: "to [destination]" (matching reference image)
  searchPillBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.parchmentSurface,
    borderColor: COLORS.parchmentBorder,
    borderWidth: 1.5,
    borderRadius: 26,
    paddingVertical: 11,
    paddingHorizontal: 16,
    gap: 10,
  },
  searchPillText: {
    flex: 1,
    fontSize: 15,
    fontWeight: '700',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  editPillBtn: {
    padding: 2,
  },

  // Route Planner Card
  routePlannerCard: {
    backgroundColor: COLORS.parchmentSurface,
    borderColor: COLORS.parchmentBorderDark,
    borderWidth: 1.5,
    borderRadius: 18,
    padding: 14,
  },
  plannerHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  plannerTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  closePlannerBtn: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: '#ebd9b7',
    borderWidth: 1,
    borderColor: COLORS.parchmentBorder,
    borderRadius: 12,
  },
  closePlannerText: {
    fontSize: 12,
    fontWeight: '700',
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
  },
  inputsRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  inputsColumn: {
    flex: 1,
  },
  swapBtn: {
    width: 36,
    height: 36,
    marginLeft: 6,
    backgroundColor: '#ebd9b7',
    borderColor: COLORS.parchmentBorderDark,
    borderWidth: 1.5,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Slider section inside Route Planner
  plannerSliderSection: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#ebd9b7',
  },
  sliderHeaderLine: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginBottom: 2,
  },
  riskLabelSmall: {
    fontSize: 11,
    fontWeight: '700',
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
  },
  riskPercentLarge: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.terracotta,
    fontFamily: 'serif',
  },
  sliderTrackControl: {
    width: '100%',
    height: 28,
  },
  sliderTickLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: -2,
    marginBottom: 4,
  },
  sliderTickText: {
    fontSize: 9,
    color: COLORS.inkMuted,
    fontFamily: 'serif',
  },
  findRoutesBtn: {
    marginTop: 8,
    width: '100%',
  },
  errorBox: {
    backgroundColor: '#f5dedb',
    borderColor: COLORS.hazardCritical,
    borderWidth: 1,
    borderRadius: 8,
    padding: 8,
    marginTop: 6,
  },
  errorText: {
    fontSize: 11,
    color: COLORS.hazardCritical,
    fontFamily: 'serif',
  },

  // Map Pick Row & Chips inside Route Planner
  mapPickSelectorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#ebd9b7',
  },
  mapPickSelectorLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
  },
  mapPickChip: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: '#ebd9b7',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.parchmentBorderDark,
  },
  mapPickChipActive: {
    backgroundColor: COLORS.terracotta,
    borderColor: COLORS.terracottaDark,
  },
  mapPickChipText: {
    fontSize: 10,
    fontWeight: '800',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  mapPickChipTextActive: {
    color: '#fff',
  },

  // Map Pick Floating Banner on Map
  mapPickFloatingBanner: {
    position: 'absolute',
    left: 18,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: COLORS.parchmentSurface,
    borderColor: COLORS.terracotta,
    borderWidth: 1.5,
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 7,
    zIndex: 25,
  },
  mapPickFloatingBannerText: {
    fontSize: 11,
    fontWeight: '800',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  mapPickCancelBtn: {
    padding: 2,
    marginLeft: 4,
  },

  // Right floating controls (Recenter crosshair button, positioned right below route planner)
  rightFloatingControls: {
    position: 'absolute',
    right: 18,
    alignItems: 'center',
    zIndex: 25,
  },
  recenterFab: {
    top: 55,
    left: -5,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: COLORS.parchmentSurface,
    borderColor: COLORS.parchmentBorder,
    borderWidth: 1.5,
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Bottom Section: Route Choices -> Start Drive
  bottomSection: {
    paddingHorizontal: 16,
    paddingBottom: 10,
    alignItems: 'center',
  },
  routesChoiceCard: {
    width: '100%',
    backgroundColor: COLORS.parchmentSurface,
    borderColor: COLORS.parchmentBorder,
    borderWidth: 1.5,
    borderRadius: 18,
    padding: 14,
    marginBottom: 10,
  },
  routesChoiceHeader: {
    fontSize: 12,
    fontWeight: '800',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
    marginBottom: 8,
  },
  routeChoicesScroll: {
    flexDirection: 'row',
    marginBottom: 10,
  },
  routeOptionTile: {
    width: 135,
    backgroundColor: '#ebd9b7',
    borderColor: COLORS.parchmentBorder,
    borderWidth: 1.5,
    borderRadius: 12,
    padding: 9,
    marginRight: 8,
  },
  routeOptionTileSelected: {
    backgroundColor: '#fffdf9',
    borderColor: COLORS.terracotta,
    borderWidth: 2,
  },
  tileHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  tileLabel: {
    fontSize: 11,
    fontWeight: '800',
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
  },
  tileLabelSelected: {
    color: COLORS.terracotta,
  },
  hazardBadge: {
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 4,
  },
  hazardBadgeSafe: {
    backgroundColor: COLORS.forestTint,
  },
  hazardBadgeWarn: {
    backgroundColor: '#f5dedb',
  },
  hazardBadgeText: {
    fontSize: 8,
    fontWeight: '800',
    color: COLORS.inkPrimary,
  },
  tileDuration: {
    fontSize: 14,
    fontWeight: '800',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  tileDistance: {
    fontSize: 10,
    color: COLORS.inkMuted,
    fontFamily: 'serif',
    marginTop: 2,
  },

  // Selected route summary row (matching reference image)
  selectedRouteSummaryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#ebd9b7',
  },
  cardCircleIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#ebd9b7',
    borderColor: COLORS.parchmentBorderDark,
    borderWidth: 1.5,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  cardIconSpoke: {
    position: 'absolute',
    width: 2,
    height: 32,
    backgroundColor: 'rgba(60, 35, 15, 0.22)',
  },
  cardIconGlyph: {
    fontSize: 11,
    fontWeight: '800',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  selectedSummaryTextCol: {
    flex: 1,
  },
  summaryTitle: {
    fontSize: 12,
    fontWeight: '800',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  summarySub: {
    fontSize: 10,
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
    marginTop: 1,
  },

  // Start Drive Pill Button
  startDriveBtn: {
    width: '70%',
    alignSelf: 'center',
  },

  // Navigation HUD
  navigationHud: {
    marginHorizontal: 16,
    marginTop: 8,
    backgroundColor: COLORS.forestDark,
    borderColor: COLORS.parchmentBorderDark,
    borderWidth: 2,
    borderRadius: 16,
    padding: 14,
    flexDirection: 'row',
    alignItems: 'center',
  },
  navTurnIconCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: COLORS.forestPine,
    borderColor: COLORS.forestFern,
    borderWidth: 1.5,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  navTurnTriangle: {
    width: 0,
    height: 0,
    borderLeftWidth: 6,
    borderRightWidth: 6,
    borderBottomWidth: 10,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    borderBottomColor: COLORS.parchmentCard,
  },
  navInstructionBlock: {
    flex: 1,
  },
  navDistanceText: {
    fontSize: 12,
    fontWeight: '700',
    color: COLORS.forestTint,
  },
  navInstructionText: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.parchmentCard,
    fontFamily: 'serif',
    marginTop: 2,
  },
  navActiveBottomBar: {
    marginHorizontal: 16,
    marginBottom: 12,
    backgroundColor: COLORS.parchmentSurface,
    borderColor: COLORS.parchmentBorderDark,
    borderWidth: 1.5,
    borderRadius: 16,
    padding: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  navActiveSummary: {
    flex: 1,
  },
  navActiveTime: {
    fontSize: 16,
    fontWeight: '800',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  navActiveSub: {
    fontSize: 11,
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
  },
  stopNavButton: {
    paddingHorizontal: 14,
  },

  // Markers
  originMarker: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: COLORS.forestPine,
    borderWidth: 2,
    borderColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  destMarker: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: COLORS.terracotta,
    borderWidth: 2,
    borderColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  markerText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '800',
  },
  userLocationMarker: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: 'rgba(59, 97, 56, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  userLocationDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: COLORS.forestPine,
    borderWidth: 1.5,
    borderColor: '#fff',
  },
});
