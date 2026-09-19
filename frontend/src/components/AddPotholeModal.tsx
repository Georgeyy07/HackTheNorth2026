import React, { useState, useEffect } from 'react';
import {
  Modal,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SeverityLevel } from '../types/pothole';
import { COLORS, SEVERITY_COLORS } from '../constants/theme';

interface AddPotholeModalProps {
  visible: boolean;
  initialCoords: { latitude: number; longitude: number } | null;
  onClose: () => void;
  onSubmit: (payload: { latitude: number; longitude: number; severity: SeverityLevel }) => Promise<void>;
}

const SEVERITIES: SeverityLevel[] = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

export const AddPotholeModal: React.FC<AddPotholeModalProps> = ({
  visible,
  initialCoords,
  onClose,
  onSubmit,
}) => {
  const [severity, setSeverity] = useState<SeverityLevel>('HIGH');
  const [lat, setLat] = useState('');
  const [lng, setLng] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialCoords) {
      setLat(initialCoords.latitude.toFixed(6));
      setLng(initialCoords.longitude.toFixed(6));
    }
  }, [initialCoords]);

  const handleSave = async () => {
    const parsedLat = parseFloat(lat);
    const parsedLng = parseFloat(lng);

    if (isNaN(parsedLat) || isNaN(parsedLng)) {
      setError('Please enter valid numerical latitude and longitude.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      await onSubmit({
        latitude: parsedLat,
        longitude: parsedLng,
        severity,
      });
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to report pothole');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.overlay}
      >
        <View style={styles.card}>
          <View style={styles.header}>
            <Text style={styles.title}>⚠️ Report Road Hazard</Text>
            <TouchableOpacity onPress={onClose} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
              <Text style={styles.closeBtn}>✕</Text>
            </TouchableOpacity>
          </View>

          <Text style={styles.sectionLabel}>SEVERITY LEVEL</Text>
          <View style={styles.severityRow}>
            {SEVERITIES.map((s) => {
              const active = severity === s;
              const colorInfo = SEVERITY_COLORS[s];
              return (
                <TouchableOpacity
                  key={s}
                  style={[
                    styles.severityBtn,
                    active && { backgroundColor: colorInfo.pin, borderColor: colorInfo.pin },
                  ]}
                  onPress={() => setSeverity(s)}
                >
                  <Text style={[styles.severityBtnText, active && { color: colorInfo.text }]}>
                    {s}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          <Text style={styles.sectionLabel}>GPS COORDINATES</Text>
          <View style={styles.coordsRow}>
            <View style={styles.inputCol}>
              <Text style={styles.inputLabel}>Latitude</Text>
              <TextInput
                style={styles.input}
                value={lat}
                onChangeText={setLat}
                keyboardType="numeric"
                placeholder="43.4723"
                placeholderTextColor={COLORS.textMuted}
              />
            </View>
            <View style={styles.inputCol}>
              <Text style={styles.inputLabel}>Longitude</Text>
              <TextInput
                style={styles.input}
                value={lng}
                onChangeText={setLng}
                keyboardType="numeric"
                placeholder="-80.5449"
                placeholderTextColor={COLORS.textMuted}
              />
            </View>
          </View>

          {error && <Text style={styles.errorText}>{error}</Text>}

          <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.cancelBtn} onPress={onClose} disabled={loading}>
              <Text style={styles.cancelBtnText}>Cancel</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.submitBtn} onPress={handleSave} disabled={loading}>
              {loading ? (
                <ActivityIndicator color="#043828" size="small" />
              ) : (
                <Text style={styles.submitBtnText}>Save to Tiger DB</Text>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    justifyContent: 'center',
    padding: 20,
  },
  card: {
    backgroundColor: COLORS.bgCardElevated,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.borderLight,
    padding: 20,
    shadowColor: '#000',
    shadowOpacity: 0.5,
    shadowRadius: 15,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    color: COLORS.text,
  },
  closeBtn: {
    fontSize: 18,
    color: COLORS.textMuted,
    fontWeight: '600',
  },
  sectionLabel: {
    fontSize: 10,
    fontWeight: '700',
    color: COLORS.emeraldGlow,
    letterSpacing: 1,
    marginBottom: 8,
    marginTop: 6,
  },
  severityRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
  },
  severityBtn: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: COLORS.border,
    backgroundColor: COLORS.bgCard,
    alignItems: 'center',
  },
  severityBtnText: {
    fontSize: 10,
    fontWeight: '700',
    color: COLORS.textMuted,
  },
  coordsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  inputCol: {
    flex: 1,
  },
  inputLabel: {
    fontSize: 11,
    color: COLORS.textMuted,
    marginBottom: 4,
  },
  input: {
    backgroundColor: COLORS.bgDark,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    color: COLORS.text,
    fontSize: 13,
  },
  errorText: {
    color: COLORS.coral,
    fontSize: 11,
    marginBottom: 12,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  cancelBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
  },
  cancelBtnText: {
    color: COLORS.textMuted,
    fontWeight: '600',
    fontSize: 13,
  },
  submitBtn: {
    flex: 2,
    backgroundColor: COLORS.emeraldGlow,
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    shadowColor: COLORS.emeraldGlow,
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  submitBtnText: {
    color: '#043828',
    fontWeight: '700',
    fontSize: 13,
  },
});
