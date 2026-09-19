import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  ScrollView,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  SafeAreaView,
  TouchableOpacity,
} from 'react-native';
import * as Location from 'expo-location';

import { getPotholes, addPothole, updatePothole, deletePothole } from '../api';
import { COLORS, WoodButton, MAP_STYLE_PARCHMENT } from '../theme';
import {
  HazardPin,
  TargetMarkerPin,
  CrosshairIcon,
  CloseIcon,
  PlusCircleIcon,
} from '../components/Icons';
import { ParchmentMap, Marker } from '../components/ParchmentMap';

export default function PotholesScreen() {
  const [potholes, setPotholes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterSeverity, setFilterSeverity] = useState('ALL');

  // Form State for Add & Edit (Visible alongside the interactive map!)
  const [formOpen, setFormOpen] = useState(false);
  const [editingPothole, setEditingPothole] = useState(null); // null when adding new
  const [lat, setLat] = useState('43.4723');
  const [lng, setLng] = useState('-80.5449');
  const [severity, setSeverity] = useState('MEDIUM');
  const [submitting, setSubmitting] = useState(false);
  const [locationLoading, setLocationLoading] = useState(false);

  const mapRef = useRef(null);

  const fetchPotholes = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getPotholes();
      setPotholes(data || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPotholes();
  }, [fetchPotholes]);

  // Open add mode
  const openAddForm = () => {
    setEditingPothole(null);
    setLat('43.4723');
    setLng('-80.5449');
    setSeverity('MEDIUM');
    setFormOpen(true);
  };

  // Open edit mode for a specific record
  const openEditForm = (item) => {
    setEditingPothole(item);
    setLat(item.latitude != null ? item.latitude.toFixed(5) : '43.4723');
    setLng(item.longitude != null ? item.longitude.toFixed(5) : '-80.5449');
    setSeverity(item.severity);
    setFormOpen(true);

    if (mapRef.current?.animateToRegion) {
      mapRef.current.animateToRegion({
        latitude: item.latitude,
        longitude: item.longitude,
        latitudeDelta: 0.02,
        longitudeDelta: 0.02,
      });
    }
  };

  const handleCloseForm = () => {
    setFormOpen(false);
    setEditingPothole(null);
  };

  // Tap anywhere on the map to set / adjust coordinates
  const handleMapPress = (e) => {
    const coord = e.nativeEvent?.coordinate;
    if (!coord) return;
    setLat(coord.latitude.toFixed(5));
    setLng(coord.longitude.toFixed(5));
    if (!formOpen) {
      setEditingPothole(null);
      setSeverity('MEDIUM');
    }
    setFormOpen(true);
  };

  // GPS current location
  const handleUseCurrentLocation = async () => {
    try {
      setLocationLoading(true);
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setError('Location permission denied.');
        return;
      }
      const position = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
      });
      const newLat = position.coords.latitude.toFixed(5);
      const newLng = position.coords.longitude.toFixed(5);
      setLat(newLat);
      setLng(newLng);

      if (mapRef.current?.animateToRegion) {
        mapRef.current.animateToRegion({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          latitudeDelta: 0.02,
          longitudeDelta: 0.02,
        });
      }
    } catch {
      setError('Could not retrieve device coordinates.');
    } finally {
      setLocationLoading(false);
    }
  };

  const handleSavePothole = async () => {
    const latNum = parseFloat(lat);
    const lngNum = parseFloat(lng);

    if (isNaN(latNum) || isNaN(lngNum)) {
      setError('Please provide valid decimal coordinates.');
      return;
    }

    try {
      setSubmitting(true);
      setError(null);

      if (editingPothole) {
        // Update existing record
        await updatePothole(editingPothole.id, {
          latitude: latNum,
          longitude: lngNum,
          severity,
        });
      } else {
        // Add new record
        await addPothole({
          latitude: latNum,
          longitude: lngNum,
          severity,
        });
      }

      setFormOpen(false);
      setEditingPothole(null);
      await fetchPotholes();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (id) => {
    try {
      setError(null);
      await deletePothole(id);
      if (editingPothole && editingPothole.id === id) {
        handleCloseForm();
      }
      fetchPotholes();
    } catch (err) {
      setError(err.message);
    }
  };

  const filteredPotholes =
    filterSeverity === 'ALL'
      ? potholes
      : potholes.filter((p) => p.severity === filterSeverity);

  const getSeverityStyle = (sev) => {
    switch (sev) {
      case 'CRITICAL':
        return styles.badgeCritical;
      case 'HIGH':
        return styles.badgeHigh;
      case 'MEDIUM':
        return styles.badgeMedium;
      default:
        return styles.badgeLow;
    }
  };

  const latNum = parseFloat(lat);
  const lngNum = parseFloat(lng);
  const hasValidCoords = !isNaN(latNum) && !isNaN(lngNum);

  return (
    <SafeAreaView style={styles.container}>
      {/* HEADER */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>Hazard Ledger</Text>
          <Text style={styles.headerSub}>Survey Records · Interactive Map</Text>
        </View>
        <WoodButton small onPress={fetchPotholes} disabled={loading}>
          {loading ? 'Reading...' : 'Refresh'}
        </WoodButton>
      </View>

      {error && (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {/* TOP INTERACTIVE PARCHMENT MAP (ALWAYS VISIBLE WHILE ADDING/EDITING) */}
      <View style={styles.mapSection}>
        <ParchmentMap
          ref={mapRef}
          customMapStyle={MAP_STYLE_PARCHMENT}
          initialRegion={{
            latitude: hasValidCoords ? latNum : 43.4723,
            longitude: hasValidCoords ? lngNum : -80.5449,
            latitudeDelta: 0.05,
            longitudeDelta: 0.05,
          }}
          onPress={handleMapPress}
          style={styles.embeddedMap}
        >
          {/* All recorded potholes plotted as pins */}
          {potholes.map((item) => {
            const isEditingThis = editingPothole && editingPothole.id === item.id;
            if (isEditingThis && formOpen) return null; // Rendered as active marker below
            return (
              <Marker
                key={`ledger-pothole-${item.id}`}
                coordinate={{ latitude: item.latitude, longitude: item.longitude }}
                title={`Record #${item.id} (${item.severity})`}
                onPress={() => openEditForm(item)}
              >
                <HazardPin severity={item.severity} />
              </Marker>
            );
          })}

          {/* Active target marker when adding or editing */}
          {formOpen && hasValidCoords && (
            <Marker
              coordinate={{ latitude: latNum, longitude: lngNum }}
              title={editingPothole ? `Update #${editingPothole.id}` : 'New Hazard Pin'}
            >
              <TargetMarkerPin severity={severity} label={editingPothole ? '✎' : '+'} />
            </Marker>
          )}
        </ParchmentMap>

        {/* Floating map hint badge */}
        <View style={styles.mapHintBadge}>
          <CrosshairIcon size={14} color={COLORS.terracotta} />
          <Text style={styles.mapHintText}>
            {formOpen ? 'Tap map to move pin coordinates' : 'Tap map anywhere to add a pothole'}
          </Text>
        </View>
      </View>

      {/* LOWER SECTION: ADD/EDIT FORM OR STATS + RECORDS LIST */}
      <View style={styles.lowerSection}>
        {formOpen ? (
          /* INLINE ADD / EDIT FORM (VISIBLE WITH THE MAP) */
          <ScrollView
            style={styles.formScroll}
            contentContainerStyle={styles.formContent}
            keyboardShouldPersistTaps="handled"
          >
            <View style={styles.formCard}>
              <View style={styles.formHeaderRow}>
                <View>
                  <Text style={styles.formTitle}>
                    {editingPothole ? `Update Hazard #${editingPothole.id}` : 'Record New Hazard'}
                  </Text>
                  <Text style={styles.formSub}>
                    Selected: {lat}, {lng}
                  </Text>
                </View>
                <TouchableOpacity style={styles.closeBtn} onPress={handleCloseForm}>
                  <CloseIcon size={14} color={COLORS.inkSecondary} />
                </TouchableOpacity>
              </View>

              {/* Coordinates input row + GPS quick button */}
              <View style={styles.coordInputsRow}>
                <View style={styles.coordCol}>
                  <Text style={styles.inputLabel}>Latitude</Text>
                  <TextInput
                    style={styles.input}
                    value={lat}
                    onChangeText={setLat}
                    keyboardType="numeric"
                    placeholder="43.4723"
                    placeholderTextColor={COLORS.inkMuted}
                  />
                </View>
                <View style={styles.coordCol}>
                  <Text style={styles.inputLabel}>Longitude</Text>
                  <TextInput
                    style={styles.input}
                    value={lng}
                    onChangeText={setLng}
                    keyboardType="numeric"
                    placeholder="-80.5449"
                    placeholderTextColor={COLORS.inkMuted}
                  />
                </View>
              </View>

              <TouchableOpacity
                style={styles.gpsLocationBtn}
                onPress={handleUseCurrentLocation}
                disabled={locationLoading}
              >
                <Text style={styles.gpsLocationBtnText}>
                  {locationLoading ? 'Acquiring GPS...' : 'Use Current Device GPS'}
                </Text>
              </TouchableOpacity>

              {/* Severity Level Picker */}
              <Text style={styles.inputLabel}>Severity Rating</Text>
              <View style={styles.severityPickerRow}>
                {['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map((sev) => {
                  const isSelected = severity === sev;
                  return (
                    <TouchableOpacity
                      key={sev}
                      style={[
                        styles.sevPickerOption,
                        isSelected && styles.sevPickerOptionSelected,
                      ]}
                      onPress={() => setSeverity(sev)}
                    >
                      <Text
                        style={[
                          styles.sevPickerText,
                          isSelected && styles.sevPickerTextSelected,
                        ]}
                      >
                        {sev}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>

              {/* Action Buttons */}
              <View style={styles.formBtnRow}>
                <TouchableOpacity
                  style={styles.cancelBtn}
                  onPress={handleCloseForm}
                  disabled={submitting}
                >
                  <Text style={styles.cancelBtnText}>Cancel</Text>
                </TouchableOpacity>

                <WoodButton
                  onPress={handleSavePothole}
                  disabled={submitting}
                  style={styles.saveRecordBtn}
                >
                  {submitting ? 'Saving...' : editingPothole ? 'Update Record' : 'Save Record'}
                </WoodButton>
              </View>
            </View>
          </ScrollView>
        ) : (
          /* STATS & FILTERED POTHOLES LIST */
          <View style={styles.recordsListWrapper}>
            {/* LEDGER STATS (2 boxes) */}
            <View style={styles.statsLedger}>
              <View style={styles.statBox}>
                <Text style={styles.statNumber}>{potholes.length}</Text>
                <Text style={styles.statLabel}>Recorded Potholes</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statBox}>
                <Text style={[styles.statNumber, { color: COLORS.hazardCritical }]}>
                  {
                    potholes.filter(
                      (p) => p.severity === 'CRITICAL' || p.severity === 'HIGH'
                    ).length
                  }
                </Text>
                <Text style={styles.statLabel}>Severe Hazards</Text>
              </View>
            </View>

            {/* ACTION ROW: RECORD NEW BUTTON & FILTER CHIPS */}
            <View style={styles.actionBar}>
              <WoodButton onPress={openAddForm} small style={styles.reportBtn}>
                + Add Hazard Pin
              </WoodButton>

              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                style={styles.filterRow}
              >
                {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map((sev) => {
                  const isActive = filterSeverity === sev;
                  return (
                    <TouchableOpacity
                      key={sev}
                      style={[styles.filterChip, isActive && styles.filterChipActive]}
                      onPress={() => setFilterSeverity(sev)}
                    >
                      <Text
                        style={[
                          styles.filterChipText,
                          isActive && styles.filterChipTextActive,
                        ]}
                      >
                        {sev}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </ScrollView>
            </View>

            {/* POTHOLES FLATLIST */}
            {loading ? (
              <View style={styles.centerLoading}>
                <ActivityIndicator size="small" color={COLORS.forestPine} />
                <Text style={styles.loadingText}>Loading survey records...</Text>
              </View>
            ) : (
              <FlatList
                data={filteredPotholes}
                keyExtractor={(item) => String(item.id)}
                contentContainerStyle={styles.listContent}
                renderItem={({ item }) => (
                  <View style={styles.recordCard}>
                    <View style={styles.cardTop}>
                      <View style={[styles.sevBadge, getSeverityStyle(item.severity)]}>
                        <Text style={styles.sevBadgeText}>{item.severity}</Text>
                      </View>
                      <Text style={styles.recordIdText}>Record #{item.id}</Text>
                    </View>

                    <Text style={styles.coordText}>
                      {item.latitude.toFixed(5)}, {item.longitude.toFixed(5)}
                    </Text>

                    <View style={styles.cardActions}>
                      <TouchableOpacity
                        style={styles.cardActionBtn}
                        onPress={() => openEditForm(item)}
                      >
                        <Text style={styles.cardActionText}>Edit on Map</Text>
                      </TouchableOpacity>

                      <TouchableOpacity
                        style={[styles.cardActionBtn, styles.cardActionDelete]}
                        onPress={() => handleDelete(item.id)}
                      >
                        <Text style={[styles.cardActionText, styles.cardActionDeleteText]}>
                          Delete
                        </Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                )}
                ListEmptyComponent={
                  <View style={styles.emptyCard}>
                    <Text style={styles.emptyText}>No hazards recorded in this category.</Text>
                  </View>
                }
              />
            )}
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.parchmentBg,
  },
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
  headerTitle: {
    fontSize: 17,
    fontWeight: '800',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  headerSub: {
    fontSize: 11,
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
    marginTop: 1,
  },
  errorBanner: {
    backgroundColor: '#f5dedb',
    borderColor: COLORS.hazardCritical,
    borderBottomWidth: 1,
    paddingHorizontal: 16,
    paddingVertical: 6,
  },
  errorText: {
    fontSize: 11,
    color: COLORS.hazardCritical,
    fontFamily: 'serif',
  },

  // Map Section
  mapSection: {
    height: 220,
    marginHorizontal: 12,
    marginTop: 10,
    borderRadius: 16,
    borderWidth: 1.5,
    borderColor: COLORS.parchmentBorderDark,
    overflow: 'hidden',
    position: 'relative',
    backgroundColor: COLORS.parchmentSurface,
  },
  embeddedMap: {
    width: '100%',
    height: '100%',
  },
  mapHintBadge: {
    position: 'absolute',
    top: 10,
    left: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(251, 245, 230, 0.92)',
    borderColor: COLORS.parchmentBorderDark,
    borderWidth: 1,
    borderRadius: 14,
    paddingHorizontal: 10,
    paddingVertical: 4,
    zIndex: 20,
  },
  mapHintText: {
    fontSize: 11,
    fontWeight: '700',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },

  // Lower Section
  lowerSection: {
    flex: 1,
    marginTop: 8,
  },

  // Form (When open, sits right below the map)
  formScroll: {
    flex: 1,
    paddingHorizontal: 12,
  },
  formContent: {
    paddingBottom: 20,
  },
  formCard: {
    backgroundColor: COLORS.parchmentSurface,
    borderColor: COLORS.parchmentBorderDark,
    borderWidth: 1.5,
    borderRadius: 16,
    padding: 14,
  },
  formHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  formTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  formSub: {
    fontSize: 11,
    color: COLORS.terracotta,
    fontWeight: '700',
    fontFamily: 'serif',
    marginTop: 1,
  },
  closeBtn: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#ebd9b7',
    borderWidth: 1,
    borderColor: COLORS.parchmentBorder,
    justifyContent: 'center',
    alignItems: 'center',
  },
  coordInputsRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 8,
  },
  coordCol: {
    flex: 1,
  },
  inputLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
    marginBottom: 3,
  },
  input: {
    backgroundColor: COLORS.parchmentInput,
    borderColor: COLORS.parchmentBorder,
    borderWidth: 1.5,
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 6,
    color: COLORS.inkPrimary,
    fontSize: 12,
    fontFamily: 'serif',
  },
  gpsLocationBtn: {
    alignSelf: 'flex-start',
    backgroundColor: '#ebd9b7',
    borderWidth: 1,
    borderColor: COLORS.parchmentBorder,
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 5,
    marginBottom: 10,
  },
  gpsLocationBtnText: {
    fontSize: 10,
    fontWeight: '700',
    color: COLORS.terracotta,
    fontFamily: 'serif',
  },
  severityPickerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  sevPickerOption: {
    flex: 1,
    paddingVertical: 6,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: COLORS.parchmentBorder,
    borderRadius: 10,
    marginHorizontal: 2,
    backgroundColor: COLORS.parchmentInput,
  },
  sevPickerOptionSelected: {
    backgroundColor: COLORS.terracotta,
    borderColor: COLORS.terracottaDark,
  },
  sevPickerText: {
    fontSize: 9,
    fontWeight: '800',
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
  },
  sevPickerTextSelected: {
    color: '#fff',
  },
  formBtnRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    alignItems: 'center',
    gap: 8,
  },
  cancelBtn: {
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 18,
    borderWidth: 1.5,
    borderColor: COLORS.parchmentBorder,
    backgroundColor: '#ebd9b7',
  },
  cancelBtnText: {
    fontSize: 11,
    fontWeight: '700',
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
  },
  saveRecordBtn: {
    minWidth: 120,
  },

  // Records List View
  recordsListWrapper: {
    flex: 1,
    paddingHorizontal: 12,
  },
  statsLedger: {
    flexDirection: 'row',
    backgroundColor: COLORS.parchmentSurface,
    borderColor: COLORS.parchmentBorder,
    borderWidth: 1.5,
    borderRadius: 14,
    paddingVertical: 8,
    marginBottom: 8,
  },
  statBox: {
    flex: 1,
    alignItems: 'center',
  },
  statDivider: {
    width: 1,
    backgroundColor: COLORS.parchmentBorder,
  },
  statNumber: {
    fontSize: 18,
    fontWeight: '800',
    color: COLORS.terracotta,
    fontFamily: 'serif',
  },
  statLabel: {
    fontSize: 10,
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
    marginTop: 1,
  },

  actionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    gap: 8,
  },
  reportBtn: {
    paddingHorizontal: 10,
  },
  filterRow: {
    flexDirection: 'row',
  },
  filterChip: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 14,
    backgroundColor: COLORS.parchmentSurface,
    borderWidth: 1.5,
    borderColor: COLORS.parchmentBorder,
    marginRight: 5,
  },
  filterChipActive: {
    backgroundColor: COLORS.terracotta,
    borderColor: COLORS.terracottaDark,
  },
  filterChipText: {
    fontSize: 9,
    fontWeight: '800',
    color: COLORS.inkSecondary,
    fontFamily: 'serif',
  },
  filterChipTextActive: {
    color: '#fff',
  },

  centerLoading: {
    paddingTop: 30,
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 11,
    color: COLORS.inkMuted,
    fontFamily: 'serif',
    marginTop: 6,
  },

  listContent: {
    paddingBottom: 20,
  },
  recordCard: {
    backgroundColor: COLORS.parchmentSurface,
    borderColor: COLORS.parchmentBorder,
    borderWidth: 1.5,
    borderRadius: 14,
    padding: 10,
    marginBottom: 8,
  },
  cardTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  sevBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#fff8ee',
  },
  badgeCritical: { backgroundColor: COLORS.hazardCritical },
  badgeHigh: { backgroundColor: COLORS.hazardHigh },
  badgeMedium: { backgroundColor: COLORS.hazardMedium },
  badgeLow: { backgroundColor: COLORS.hazardLow },
  sevBadgeText: {
    color: '#fff8ee',
    fontSize: 8,
    fontWeight: '800',
    fontFamily: 'serif',
  },
  recordIdText: {
    fontSize: 10,
    color: COLORS.inkMuted,
    fontFamily: 'serif',
  },
  coordText: {
    fontSize: 11,
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
    marginBottom: 6,
  },
  cardActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    borderTopWidth: 1,
    borderTopColor: '#ebd9b7',
    paddingTop: 6,
    gap: 6,
  },
  cardActionBtn: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: '#ebd9b7',
    borderColor: COLORS.parchmentBorder,
    borderWidth: 1,
    borderRadius: 10,
  },
  cardActionText: {
    fontSize: 10,
    fontWeight: '700',
    color: COLORS.inkPrimary,
    fontFamily: 'serif',
  },
  cardActionDelete: {
    borderColor: COLORS.hazardCritical,
    backgroundColor: '#f5dedb',
  },
  cardActionDeleteText: {
    color: COLORS.hazardCritical,
  },

  emptyCard: {
    paddingVertical: 24,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 11,
    color: COLORS.inkMuted,
    fontFamily: 'serif',
    fontStyle: 'italic',
  },
});
