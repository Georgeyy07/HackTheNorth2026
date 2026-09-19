import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import MapView, { Marker, Polyline, PROVIDER_DEFAULT } from 'react-native-maps';

import { getRoute } from '../api';

const ROUTE_COLORS = ['#ff4d6d', '#00f2fe', '#ffb454', '#a78bfa'];
const RISK_COLORS = { None: '#159c78', LOW: '#caaa50', MEDIUM: '#e78043', HIGH: '#c95268', CRITICAL: '#8b1e3f' };

function toLatLng(coords) {
  return coords.map(([lat, lon]) => ({ latitude: lat, longitude: lon }));
}

export default function RouteFinderScreen() {
  const [origin, setOrigin] = useState('University of Waterloo, Waterloo, ON');
  const [destination, setDestination] = useState('Waterloo Public Square, Waterloo, ON');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const handleFindRoutes = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getRoute(origin, destination);
      setResult(data);
      setSelectedIndex(0);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const initialRegion = result
    ? {
        latitude: result.origin.lat,
        longitude: result.origin.lon,
        latitudeDelta: 0.05,
        longitudeDelta: 0.05,
      }
    : {
        latitude: 43.4723,
        longitude: -80.5449,
        latitudeDelta: 0.05,
        longitudeDelta: 0.05,
      };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Route Finder</Text>
        <Text style={styles.subtitle}>Efficiency vs potholes</Text>
      </View>

      <View style={styles.form}>
        <TextInput
          style={styles.input}
          placeholder="Origin address"
          placeholderTextColor="#6b7280"
          value={origin}
          onChangeText={setOrigin}
        />
        <TextInput
          style={styles.input}
          placeholder="Destination address"
          placeholderTextColor="#6b7280"
          value={destination}
          onChangeText={setDestination}
        />
        <TouchableOpacity style={styles.button} onPress={handleFindRoutes} disabled={loading}>
          {loading ? <ActivityIndicator color="#0b0f19" /> : <Text style={styles.buttonText}>Find routes</Text>}
        </TouchableOpacity>
      </View>

      {error && (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      <View style={styles.mapContainer}>
        <MapView provider={PROVIDER_DEFAULT} style={styles.map} region={initialRegion}>
          {result && result.routes.map((route, i) => (
            <Polyline
              key={route.label}
              coordinates={toLatLng(route.coords)}
              strokeColor={ROUTE_COLORS[i % ROUTE_COLORS.length]}
              strokeWidth={i === selectedIndex ? 6 : 3}
              zIndex={i === selectedIndex ? 10 : 1}
            />
          ))}
          {result && (
            <>
              <Marker coordinate={{ latitude: result.origin.lat, longitude: result.origin.lon }} title="Origin" pinColor="dodgerblue" />
              <Marker coordinate={{ latitude: result.destination.lat, longitude: result.destination.lon }} title="Destination" pinColor="dodgerblue" />
            </>
          )}
          {result && result.routes[selectedIndex]?.potholes_encountered.map((p) => (
            <Marker
              key={p.id}
              coordinate={{ latitude: p.lat, longitude: p.lon }}
              title={`${p.severity} pothole`}
              pinColor="red"
            />
          ))}
        </MapView>
      </View>

      {result && (
        <ScrollView style={styles.results}>
          {result.unmatched_potholes > 0 && (
            <Text style={styles.unmatchedText}>
              {result.unmatched_potholes} pothole report(s) too far from any road to route around.
            </Text>
          )}
          {result.routes.map((route, i) => {
            const riskColor = RISK_COLORS[route.risk_rating] || '#789c8b';
            const mins = Math.floor(route.duration_s / 60);
            const secs = Math.round(route.duration_s % 60);
            const potholeList = route.potholes_encountered.length
              ? route.potholes_encountered.map((p) => `${p.severity} #${p.id.slice(0, 8)}`).join(', ')
              : 'none on this route';
            return (
              <TouchableOpacity
                key={route.label}
                style={[
                  styles.routeCard,
                  { borderLeftColor: ROUTE_COLORS[i % ROUTE_COLORS.length] },
                  i === selectedIndex && styles.routeCardSelected,
                ]}
                onPress={() => setSelectedIndex(i)}
              >
                <View style={styles.routeCardHeader}>
                  <Text style={styles.routeLabel}>{route.label}</Text>
                  <View style={[styles.riskBadge, { backgroundColor: `${riskColor}22`, borderColor: riskColor }]}>
                    <Text style={[styles.riskBadgeText, { color: riskColor }]}>{route.risk_rating} risk</Text>
                  </View>
                </View>
                <Text style={styles.routeMeta}>{Math.round(route.distance_m)}m · {mins}m {secs}s</Text>
                <Text style={styles.routePotholes}>Potholes: {potholeList}</Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0b0f19' },
  header: { padding: 16, paddingBottom: 8 },
  title: { fontSize: 20, fontWeight: '700', color: '#00f2fe' },
  subtitle: { fontSize: 11, color: '#9ca3af', marginTop: 2 },
  form: { paddingHorizontal: 16, gap: 8 },
  input: {
    backgroundColor: '#1f2937', borderColor: '#374151', borderWidth: 1, borderRadius: 8,
    padding: 12, color: '#fff', fontSize: 13, marginBottom: 8,
  },
  button: { backgroundColor: '#00f2fe', padding: 12, borderRadius: 8, alignItems: 'center', marginBottom: 8 },
  buttonText: { color: '#0b0f19', fontWeight: '700', fontSize: 13 },
  errorBanner: { marginHorizontal: 16, backgroundColor: '#7f1d1d', borderRadius: 8, padding: 10, marginBottom: 8 },
  errorText: { color: '#fecaca', fontSize: 12 },
  mapContainer: { height: 260, marginHorizontal: 16, borderRadius: 12, overflow: 'hidden' },
  map: { flex: 1 },
  results: { flex: 1, padding: 16 },
  unmatchedText: { color: '#f0c674', fontSize: 11, marginBottom: 8 },
  routeCard: {
    backgroundColor: '#152b22', borderRadius: 8, padding: 12, marginBottom: 8,
    borderLeftWidth: 3, borderWidth: 1, borderColor: '#214739',
  },
  routeCardSelected: { borderColor: '#00f2fe', backgroundColor: '#1a3a30' },
  routeCardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  routeLabel: { color: '#e6f4fe', fontWeight: '700', fontSize: 13 },
  riskBadge: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 10, borderWidth: 1 },
  riskBadgeText: { fontSize: 9, fontWeight: '700' },
  routeMeta: { color: '#9ca3af', fontSize: 11, marginBottom: 2 },
  routePotholes: { color: '#9ca3af', fontSize: 11 },
});
