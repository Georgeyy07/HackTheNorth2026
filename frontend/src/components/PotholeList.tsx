import React, { useState } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native';
import { Pothole, SeverityLevel } from '../types/pothole';
import { COLORS, SEVERITY_COLORS } from '../constants/theme';

interface PotholeListProps {
  potholes: Pothole[];
  selectedPothole: Pothole | null;
  onSelectPothole: (pothole: Pothole) => void;
  onEditPothole: (pothole: Pothole) => void;
  onDeletePothole: (id: number) => void;
  onSeedDatabase: () => void;
}

export const PotholeList: React.FC<PotholeListProps> = ({
  potholes,
  selectedPothole,
  onSelectPothole,
  onEditPothole,
  onDeletePothole,
  onSeedDatabase,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const safePotholes = Array.isArray(potholes) ? potholes : [];

  const confirmDelete = (pothole: Pothole) => {
    if (typeof window !== 'undefined' && (window as any).confirm) {
      if ((window as any).confirm(`Are you sure you want to delete Hazard #${pothole.id}?`)) {
        onDeletePothole(pothole.id);
      }
    } else {
      Alert.alert(
        'Delete Pothole Record',
        `Are you sure you want to delete ${pothole.severity} hazard #${pothole.id}?`,
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Delete', style: 'destructive', onPress: () => onDeletePothole(pothole.id) },
        ]
      );
    }
  };

  const renderItem = ({ item }: { item: Pothole }) => {
    const isSelected = selectedPothole?.id === item.id;
    const colorInfo = SEVERITY_COLORS[item.severity] || SEVERITY_COLORS.MEDIUM;

    return (
      <TouchableOpacity
        style={[styles.itemCard, isSelected && styles.itemCardSelected]}
        onPress={() => onSelectPothole(item)}
      >
        <View style={styles.itemHeader}>
          <View style={[styles.badge, { backgroundColor: colorInfo.bg }]}>
            <Text style={[styles.badgeText, { color: colorInfo.text }]}>{item.severity}</Text>
          </View>
          <Text style={styles.recordId}>Tiger DB #{item.id}</Text>
        </View>

        <Text style={styles.coords}>
          📍 {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
        </Text>

        {item.timestamp ? (
          <Text style={styles.timestamp}>
            🕒 {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </Text>
        ) : null}

        <View style={styles.itemActions}>
          <TouchableOpacity
            style={[styles.actionBtn, { backgroundColor: '#133527' }]}
            onPress={() => onSelectPothole(item)}
          >
            <Text style={styles.actionBtnText}>📍 Center</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionBtn, { backgroundColor: '#1b4435' }]}
            onPress={() => onEditPothole(item)}
          >
            <Text style={styles.actionBtnText}>✏️ Edit</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionBtn, { backgroundColor: '#4a1520' }]}
            onPress={() => confirmDelete(item)}
          >
            <Text style={[styles.actionBtnText, { color: COLORS.coral }]}>🗑️ Delete</Text>
          </TouchableOpacity>
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <View style={[styles.container, !isExpanded && styles.containerCollapsed]}>
      <TouchableOpacity
        style={styles.drawerHeader}
        onPress={() => setIsExpanded(!isExpanded)}
        activeOpacity={0.8}
      >
        <View style={styles.dragHandle} />
        <View style={styles.titleRow}>
          <View style={styles.titleGroup}>
            <Text style={styles.title}>Road Hazards</Text>
            <View style={styles.countPill}>
              <Text style={styles.countText}>{safePotholes.length}</Text>
            </View>
          </View>
          <Text style={styles.chevron}>{isExpanded ? '▼' : '▲'}</Text>
        </View>
      </TouchableOpacity>

      {isExpanded && (
        <FlatList
          data={safePotholes}
          keyExtractor={(item) => item.id.toString()}
          renderItem={renderItem}
          contentContainerStyle={styles.listContent}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyTitle}>No Hazards Reported</Text>
              <Text style={styles.emptySubtitle}>
                No potholes currently recorded in this database view.
              </Text>
              <TouchableOpacity style={styles.seedBtn} onPress={onSeedDatabase}>
                <Text style={styles.seedBtnText}>Seed Sample Potholes</Text>
              </TouchableOpacity>
            </View>
          }
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.bgCardElevated,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderTopWidth: 1,
    borderColor: COLORS.borderLight,
    maxHeight: 280,
    shadowColor: '#000',
    shadowOpacity: 0.35,
    shadowRadius: 10,
    elevation: 8,
  },
  containerCollapsed: {
    maxHeight: 56,
  },
  drawerHeader: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  dragHandle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: COLORS.borderLight,
    marginBottom: 8,
  },
  titleRow: {
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  titleGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  title: {
    fontSize: 15,
    fontWeight: '700',
    color: COLORS.text,
  },
  countPill: {
    backgroundColor: COLORS.border,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 12,
  },
  countText: {
    fontSize: 11,
    fontWeight: '700',
    color: COLORS.cyan,
  },
  chevron: {
    color: COLORS.textMuted,
    fontSize: 12,
  },
  listContent: {
    padding: 14,
    gap: 10,
  },
  itemCard: {
    backgroundColor: COLORS.bgCard,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 12,
    padding: 12,
  },
  itemCardSelected: {
    borderColor: COLORS.cyan,
    backgroundColor: '#122a20',
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  badge: {
    paddingHorizontal: 7,
    paddingVertical: 3,
    borderRadius: 5,
  },
  badgeText: {
    fontSize: 9,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  recordId: {
    fontSize: 11,
    color: COLORS.cyan,
    fontWeight: '600',
  },
  coords: {
    fontSize: 12,
    color: COLORS.text,
    fontFamily: 'monospace',
    marginBottom: 2,
  },
  timestamp: {
    fontSize: 10,
    color: COLORS.textMuted,
    marginBottom: 8,
  },
  itemActions: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 4,
  },
  actionBtn: {
    flex: 1,
    paddingVertical: 6,
    borderRadius: 6,
    alignItems: 'center',
  },
  actionBtnText: {
    fontSize: 11,
    fontWeight: '600',
    color: COLORS.text,
  },
  emptyContainer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  emptyTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: COLORS.text,
    marginBottom: 4,
  },
  emptySubtitle: {
    fontSize: 11,
    color: COLORS.textMuted,
    marginBottom: 12,
    textAlign: 'center',
  },
  seedBtn: {
    backgroundColor: COLORS.emeraldGlow,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  seedBtnText: {
    color: '#043828',
    fontWeight: '700',
    fontSize: 11,
  },
});
