import React, { useState, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
  Alert,
  TextInput,
  Modal,
} from 'react-native';
import { Pothole, SeverityLevel, NewPotholePayload, UpdatePotholePayload } from './src/types/pothole';
import { COLORS, SEVERITY_COLORS } from './src/constants/theme';
import {
  fetchPotholes,
  createPothole,
  updatePothole,
  deletePothole,
  seedPotholes,
  getApiBaseUrl,
  setApiBaseUrl,
} from './src/api/client';
import { PotholeMap } from './src/components/PotholeMap';
import { PotholeList } from './src/components/PotholeList';
import { AddPotholeModal } from './src/components/AddPotholeModal';
import { EditPotholeModal } from './src/components/EditPotholeModal';

const FILTERS: (SeverityLevel | 'ALL')[] = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

export default function App() {
  const [potholes, setPotholes] = useState<Pothole[]>([]);
  const [filteredPotholes, setFilteredPotholes] = useState<Pothole[]>([]);
  const [activeFilter, setActiveFilter] = useState<SeverityLevel | 'ALL'>('ALL');
  const [selectedPothole, setSelectedPothole] = useState<Pothole | null>(null);
  const [editingPothole, setEditingPothole] = useState<Pothole | null>(null);

  // Modals
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [newCoords, setNewCoords] = useState<{ latitude: number; longitude: number } | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [serverUrlInput, setServerUrlInput] = useState(getApiBaseUrl());

  // Connection state
  const [isConnected, setIsConnected] = useState(true);

  const loadData = useCallback(async () => {
    try {
      const data = await fetchPotholes();
      setPotholes(Array.isArray(data) ? data : []);
      setIsConnected(true);
    } catch (err) {
      setIsConnected(false);
      setPotholes([]);
      console.warn('Could not connect to backend:', err);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    const list = Array.isArray(potholes) ? potholes : [];
    if (activeFilter === 'ALL') {
      setFilteredPotholes(list);
    } else {
      setFilteredPotholes(list.filter((p) => p.severity === activeFilter));
    }
  }, [potholes, activeFilter]);

  // Actions
  const handleMapPress = (coords: { latitude: number; longitude: number }) => {
    setNewCoords(coords);
    setIsAddModalOpen(true);
  };

  const handleCreatePothole = async (payload: NewPotholePayload) => {
    const created = await createPothole(payload);
    setPotholes((prev) => [created, ...prev]);
    setSelectedPothole(created);
  };

  const handleMovePothole = async (id: number, latitude: number, longitude: number) => {
    try {
      await updatePothole(id, { latitude, longitude });
      setPotholes((prev) =>
        prev.map((p) => (p.id === id ? { ...p, latitude, longitude } : p))
      );
    } catch (err: any) {
      Alert.alert('Move Failed', err.message || 'Could not move pothole');
    }
  };

  const handleUpdatePothole = async (id: number, updates: UpdatePotholePayload) => {
    await updatePothole(id, updates);
    setPotholes((prev) =>
      prev.map((p) => (p.id === id ? { ...p, ...updates } : p))
    );
  };

  const handleDeletePothole = async (id: number) => {
    try {
      await deletePothole(id);
      setPotholes((prev) => prev.filter((p) => p.id !== id));
      if (selectedPothole?.id === id) {
        setSelectedPothole(null);
      }
    } catch (err: any) {
      Alert.alert('Delete Failed', err.message || 'Could not delete pothole');
    }
  };

  const handleSeedDatabase = async () => {
    try {
      const res = await seedPotholes();
      setPotholes(res.potholes);
    } catch (err: any) {
      Alert.alert('Seed Failed', err.message || 'Could not seed database');
    }
  };

  const handleSaveSettings = () => {
    setApiBaseUrl(serverUrlInput);
    setIsSettingsOpen(false);
    loadData();
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bgDark} />

      {/* Top Navigation Bar */}
      <View style={styles.topbar}>
        <View style={styles.brandRow}>
          <View style={styles.brandIcon}>
            <Text style={{ fontSize: 16 }}>⚠️</Text>
          </View>
          <View>
            <Text style={styles.brandTitle}>Roadscope Native</Text>
            <Text style={styles.brandSubtitle}>Tiger Data TimescaleDB</Text>
          </View>
        </View>

        <View style={styles.topRightActions}>
          <View style={[styles.statusPill, isConnected ? styles.statusOnline : styles.statusOffline]}>
            <View style={[styles.statusDot, isConnected ? styles.dotOnline : styles.dotOffline]} />
            <Text style={styles.statusText}>{isConnected ? 'Connected' : 'Offline'}</Text>
          </View>

          <TouchableOpacity style={styles.settingsIconBtn} onPress={() => setIsSettingsOpen(true)}>
            <Text style={{ fontSize: 16 }}>⚙️</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Filter Chips Bar */}
      <View style={styles.filterBar}>
        {FILTERS.map((f) => {
          const isActive = activeFilter === f;
          const colorInfo = f !== 'ALL' ? SEVERITY_COLORS[f] : null;

          return (
            <TouchableOpacity
              key={f}
              style={[
                styles.filterChip,
                isActive && styles.filterChipActive,
                isActive && colorInfo && { borderColor: colorInfo.pin },
              ]}
              onPress={() => setActiveFilter(f)}
            >
              {colorInfo && (
                <View style={[styles.filterDot, { backgroundColor: colorInfo.pin }]} />
              )}
              <Text
                style={[
                  styles.filterText,
                  isActive && styles.filterTextActive,
                  isActive && colorInfo && { color: colorInfo.pin },
                ]}
              >
                {f}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* Interactive Map Area */}
      <View style={styles.mapContainer}>
        <PotholeMap
          potholes={filteredPotholes}
          selectedPothole={selectedPothole}
          onSelectPothole={setSelectedPothole}
          onMovePothole={handleMovePothole}
          onMapPress={handleMapPress}
          onDeletePothole={handleDeletePothole}
          onEditPothole={setEditingPothole}
        />

        {/* Floating Action Buttons */}
        <View style={styles.floatingActions}>
          <TouchableOpacity
            style={styles.fabSecondary}
            onPress={handleSeedDatabase}
            activeOpacity={0.8}
          >
            <Text style={styles.fabSecondaryText}>🌱 Seed DB</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.fabPrimary}
            onPress={() => {
              setNewCoords(
                potholes.length > 0
                  ? { latitude: potholes[0].latitude, longitude: potholes[0].longitude }
                  : { latitude: 43.4723, longitude: -80.5449 }
              );
              setIsAddModalOpen(true);
            }}
            activeOpacity={0.8}
          >
            <Text style={styles.fabPrimaryText}>+ Report Hazard</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Bottom Collapsible List */}
      <PotholeList
        potholes={filteredPotholes}
        selectedPothole={selectedPothole}
        onSelectPothole={setSelectedPothole}
        onEditPothole={setEditingPothole}
        onDeletePothole={handleDeletePothole}
        onSeedDatabase={handleSeedDatabase}
      />

      {/* Modals */}
      <AddPotholeModal
        visible={isAddModalOpen}
        initialCoords={newCoords}
        onClose={() => setIsAddModalOpen(false)}
        onSubmit={handleCreatePothole}
      />

      <EditPotholeModal
        visible={!!editingPothole}
        pothole={editingPothole}
        onClose={() => setEditingPothole(null)}
        onSubmit={handleUpdatePothole}
      />

      {/* Server URL Settings Modal */}
      <Modal visible={isSettingsOpen} transparent animationType="fade" onRequestClose={() => setIsSettingsOpen(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.settingsCard}>
            <Text style={styles.settingsTitle}>Backend Server Configuration</Text>
            <Text style={styles.settingsSubtitle}>
              Configure the host URL for the FastAPI backend (e.g. when connecting from a physical phone via Wi-Fi).
            </Text>

            <Text style={styles.inputLabel}>SERVER URL</Text>
            <TextInput
              style={styles.settingsInput}
              value={serverUrlInput}
              onChangeText={setServerUrlInput}
              autoCapitalize="none"
              placeholder="http://192.168.1.50:8765"
              placeholderTextColor={COLORS.textMuted}
            />

            <View style={styles.settingsBtnRow}>
              <TouchableOpacity
                style={styles.settingsCancelBtn}
                onPress={() => setIsSettingsOpen(false)}
              >
                <Text style={{ color: COLORS.textMuted, fontWeight: '600' }}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.settingsSaveBtn}
                onPress={handleSaveSettings}
              >
                <Text style={{ color: '#043828', fontWeight: '700' }}>Connect</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.bgDark,
  },
  topbar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
    backgroundColor: COLORS.bgCard,
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  brandIcon: {
    width: 32,
    height: 32,
    borderRadius: 8,
    backgroundColor: COLORS.bgDark,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: COLORS.borderLight,
  },
  brandTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.text,
    letterSpacing: -0.3,
  },
  brandSubtitle: {
    fontSize: 10,
    color: COLORS.cyan,
    fontWeight: '600',
  },
  topRightActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
  },
  statusOnline: {
    backgroundColor: '#0c271e',
    borderColor: '#1e624c',
  },
  statusOffline: {
    backgroundColor: '#301318',
    borderColor: '#62232e',
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  dotOnline: {
    backgroundColor: COLORS.emeraldGlow,
  },
  dotOffline: {
    backgroundColor: COLORS.coral,
  },
  statusText: {
    fontSize: 9,
    fontWeight: '600',
    color: COLORS.text,
  },
  settingsIconBtn: {
    padding: 6,
    borderRadius: 8,
    backgroundColor: COLORS.bgDark,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  filterBar: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 10,
    gap: 8,
    backgroundColor: COLORS.bgDark,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  filterChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 16,
    backgroundColor: COLORS.bgCard,
    borderWidth: 1,
    borderColor: COLORS.border,
    gap: 5,
  },
  filterChipActive: {
    backgroundColor: COLORS.bgCardElevated,
    borderColor: COLORS.emeraldGlow,
  },
  filterDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  filterText: {
    fontSize: 10,
    fontWeight: '700',
    color: COLORS.textMuted,
  },
  filterTextActive: {
    color: COLORS.emeraldGlow,
  },
  mapContainer: {
    flex: 1,
    position: 'relative',
  },
  floatingActions: {
    position: 'absolute',
    bottom: 16,
    left: 16,
    right: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
    pointerEvents: 'box-none',
  },
  fabPrimary: {
    backgroundColor: COLORS.emeraldGlow,
    paddingHorizontal: 16,
    paddingVertical: 11,
    borderRadius: 25,
    shadowColor: COLORS.emeraldGlow,
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 6,
  },
  fabPrimaryText: {
    color: '#043828',
    fontSize: 12,
    fontWeight: '800',
  },
  fabSecondary: {
    backgroundColor: 'rgba(15, 34, 26, 0.9)',
    borderWidth: 1,
    borderColor: COLORS.borderLight,
    paddingHorizontal: 14,
    paddingVertical: 11,
    borderRadius: 25,
    elevation: 4,
  },
  fabSecondaryText: {
    color: COLORS.text,
    fontSize: 12,
    fontWeight: '700',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    justifyContent: 'center',
    padding: 20,
  },
  settingsCard: {
    backgroundColor: COLORS.bgCardElevated,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.borderLight,
    padding: 20,
  },
  settingsTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: COLORS.text,
    marginBottom: 4,
  },
  settingsSubtitle: {
    fontSize: 11,
    color: COLORS.textMuted,
    marginBottom: 16,
    lineHeight: 16,
  },
  inputLabel: {
    fontSize: 10,
    fontWeight: '700',
    color: COLORS.emeraldGlow,
    marginBottom: 6,
  },
  settingsInput: {
    backgroundColor: COLORS.bgDark,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 9,
    color: COLORS.text,
    fontSize: 13,
    marginBottom: 16,
  },
  settingsBtnRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 10,
  },
  settingsCancelBtn: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 8,
  },
  settingsSaveBtn: {
    backgroundColor: COLORS.emeraldGlow,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
});
