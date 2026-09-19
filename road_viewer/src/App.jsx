import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  TextInput,
  ScrollView,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  Modal,
  SafeAreaView
} from 'react-native-web';

export default function App() {
  const [potholes, setPotholes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [filterSeverity, setFilterSeverity] = useState('ALL');
  
  // New Pothole Form state
  const [lat, setLat] = useState('43.4723');
  const [lng, setLng] = useState('-80.5449');
  const [severity, setSeverity] = useState('MEDIUM');

  const fetchPotholes = async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/potholes');
      const data = await res.json();
      setPotholes(data);
    } catch (err) {
      console.error('Error fetching potholes:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPotholes();
  }, []);

  const handleAddPothole = async () => {
    try {
      const payload = {
        latitude: parseFloat(lat),
        longitude: parseFloat(lng),
        severity
      };
      await fetch('/api/potholes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      setShowForm(false);
      fetchPotholes();
    } catch (err) {
      console.error('Error adding pothole:', err);
    }
  };

  const handleSeed = async () => {
    try {
      await fetch('/api/potholes/seed', { method: 'POST' });
      fetchPotholes();
    } catch (err) {
      console.error('Error seeding potholes:', err);
    }
  };

  const handleDelete = async (id) => {
    try {
      await fetch(`/api/potholes/${id}`, { method: 'DELETE' });
      fetchPotholes();
    } catch (err) {
      console.error('Error deleting pothole:', err);
    }
  };

  const filteredPotholes = filterSeverity === 'ALL'
    ? potholes
    : potholes.filter(p => p.severity === filterSeverity);

  const getSeverityStyle = (sev) => {
    switch (sev) {
      case 'CRITICAL': return styles.badgeCritical;
      case 'HIGH': return styles.badgeHigh;
      case 'MEDIUM': return styles.badgeMedium;
      default: return styles.badgeLow;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.brandContainer}>
          <Text style={styles.brandTitle}>RoughRoute RN</Text>
          <View style={styles.dbTag}>
            <View style={styles.pulseDot} />
            <Text style={styles.dbTagText}>Tiger Data DB (TimescaleDB)</Text>
          </View>
        </View>
        <TouchableOpacity style={styles.seedBtn} onPress={handleSeed}>
          <Text style={styles.seedBtnText}>Seed DB</Text>
        </TouchableOpacity>
      </View>

      {/* Main Content Area */}
      <View style={styles.content}>
        {/* Pothole Stats Card */}
        <View style={styles.statsCard}>
          <View style={styles.statBox}>
            <Text style={styles.statNumber}>{potholes.length}</Text>
            <Text style={styles.statLabel}>Total Potholes</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statNumber}>
              {potholes.filter(p => p.severity === 'CRITICAL' || p.severity === 'HIGH').length}
            </Text>
            <Text style={styles.statLabel}>High Hazards</Text>
          </View>
        </View>

        {/* Action Bar */}
        <View style={styles.actionBar}>
          <TouchableOpacity
            style={styles.addBtn}
            onPress={() => setShowForm(true)}
          >
            <Text style={styles.addBtnText}>+ Report Pothole to Tiger Data</Text>
          </TouchableOpacity>

          {/* Filter Chips */}
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterScroll}>
            {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map(sev => (
              <TouchableOpacity
                key={sev}
                style={[
                  styles.filterChip,
                  filterSeverity === sev && styles.filterChipActive
                ]}
                onPress={() => setFilterSeverity(sev)}
              >
                <Text style={[
                  styles.filterChipText,
                  filterSeverity === sev && styles.filterChipTextActive
                ]}>{sev}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Potholes List */}
        {loading ? (
          <ActivityIndicator size="large" color="#00f2fe" style={{ marginTop: 20 }} />
        ) : (
          <FlatList
            data={filteredPotholes}
            keyExtractor={(item) => item.id.toString()}
            renderItem={({ item }) => (
              <View style={styles.potholeCard}>
                <View style={styles.cardHeader}>
                  <View style={[styles.sevBadge, getSeverityStyle(item.severity)]}>
                    <Text style={styles.sevBadgeText}>{item.severity}</Text>
                  </View>
                  <Text style={styles.coordsText}>
                    📍 {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
                  </Text>
                </View>

                <View style={styles.cardFooter}>
                  <Text style={styles.statusText}>Tiger DB #{item.id}</Text>
                  <TouchableOpacity
                    style={styles.deleteBtn}
                    onPress={() => handleDelete(item.id)}
                  >
                    <Text style={styles.btnText}>Delete</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}
          />
        )}
      </View>

      {/* Add Pothole Modal */}
      <Modal visible={showForm} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Report Pothole to Tiger Data DB</Text>

            <Text style={styles.inputLabel}>Latitude</Text>
            <TextInput style={styles.input} value={lat} onChangeText={setLat} keyboardType="numeric" />

            <Text style={styles.inputLabel}>Longitude</Text>
            <TextInput style={styles.input} value={lng} onChangeText={setLng} keyboardType="numeric" />

            <View style={styles.modalButtons}>
              <TouchableOpacity style={styles.saveBtn} onPress={handleAddPothole}>
                <Text style={styles.saveBtnText}>Save Record</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setShowForm(false)}>
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0b0f19'
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    backgroundColor: '#111827',
    borderBottomWidth: 1,
    borderBottomColor: '#1f2937'
  },
  brandContainer: {
    flexDirection: 'column'
  },
  brandTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#00f2fe'
  },
  dbTag: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4
  },
  pulseDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#06d6a0',
    marginRight: 6
  },
  dbTagText: {
    fontSize: 10,
    color: '#9ca3af',
    fontWeight: '500'
  },
  seedBtn: {
    backgroundColor: '#0284c7',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6
  },
  seedBtnText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600'
  },
  content: {
    flex: 1,
    padding: 16
  },
  statsCard: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#1f2937',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16
  },
  statBox: {
    alignItems: 'center'
  },
  statNumber: {
    fontSize: 22,
    fontWeight: '700',
    color: '#00f2fe'
  },
  statLabel: {
    fontSize: 11,
    color: '#9ca3af',
    marginTop: 2
  },
  actionBar: {
    marginBottom: 16
  },
  addBtn: {
    backgroundColor: '#059669',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 12
  },
  addBtnText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 13
  },
  filterScroll: {
    flexDirection: 'row'
  },
  filterChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: '#1f2937',
    marginRight: 8
  },
  filterChipActive: {
    backgroundColor: '#00f2fe'
  },
  filterChipText: {
    color: '#9ca3af',
    fontSize: 11,
    fontWeight: '600'
  },
  filterChipTextActive: {
    color: '#0b0f19'
  },
  potholeCard: {
    backgroundColor: '#1f2937',
    borderRadius: 10,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#374151'
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8
  },
  sevBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4
  },
  badgeCritical: { backgroundColor: '#ef4444' },
  badgeHigh: { backgroundColor: '#f97316' },
  badgeMedium: { backgroundColor: '#eab308' },
  badgeLow: { backgroundColor: '#3b82f6' },
  sevBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700'
  },
  coordsText: {
    color: '#9ca3af',
    fontSize: 11,
    fontFamily: 'monospace'
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#374151',
    paddingTop: 8
  },
  statusText: {
    color: '#9ca3af',
    fontSize: 11
  },
  deleteBtn: {
    backgroundColor: '#dc2626',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4
  },
  btnText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '600'
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.75)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20
  },
  modalContent: {
    width: '100%',
    backgroundColor: '#1f2937',
    borderRadius: 12,
    padding: 20
  },
  modalTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#00f2fe',
    marginBottom: 16
  },
  inputLabel: {
    color: '#9ca3af',
    fontSize: 11,
    marginBottom: 4
  },
  input: {
    backgroundColor: '#111827',
    borderColor: '#374151',
    borderWidth: 1,
    borderRadius: 6,
    padding: 10,
    color: '#fff',
    marginBottom: 12,
    fontSize: 13
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginTop: 10
  },
  saveBtn: {
    backgroundColor: '#0284c7',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 6,
    marginRight: 8
  },
  saveBtnText: {
    color: '#fff',
    fontWeight: '600'
  },
  cancelBtn: {
    backgroundColor: '#374151',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 6
  },
  cancelBtnText: {
    color: '#9ca3af',
    fontWeight: '600'
  }
});
