import React, { useState, useRef, useEffect } from 'react';
import { View, TextInput, TouchableOpacity, Text, StyleSheet, ActivityIndicator } from 'react-native';

import { suggestAddresses } from '../api';
import { COLORS } from '../theme';

const DEBOUNCE_MS = 400;
const MIN_QUERY_LENGTH = 3;

export default function AddressInput({
  value,
  onChangeText,
  onSelectAddress,
  placeholder,
  editable = true,
  style,
}) {
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const debounceRef = useRef(null);
  const requestIdRef = useRef(0);

  useEffect(() => () => clearTimeout(debounceRef.current), []);

  const handleChangeText = (text) => {
    onChangeText(text);
    if (onSelectAddress) {
      onSelectAddress(null); // Clear stale coordinates because user modified text manually
    }
    setDismissed(false);
    clearTimeout(debounceRef.current);

    if (text.trim().length < MIN_QUERY_LENGTH) {
      setSuggestions([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    debounceRef.current = setTimeout(async () => {
      const requestId = ++requestIdRef.current;
      try {
        const results = await suggestAddresses(text);
        if (requestId === requestIdRef.current) setSuggestions(results);
      } catch {
        if (requestId === requestIdRef.current) setSuggestions([]);
      } finally {
        if (requestId === requestIdRef.current) setLoading(false);
      }
    }, DEBOUNCE_MS);
  };

  const handleSelect = (item) => {
    onChangeText(item.display_name);
    if (onSelectAddress) {
      onSelectAddress({ display_name: item.display_name, lat: item.lat, lon: item.lon });
    }
    setSuggestions([]);
    setDismissed(true);
  };

  const showDropdown = !dismissed && (loading || suggestions.length > 0);

  return (
    <View style={[styles.container, style]}>
      <View style={styles.inputRow}>
        <TextInput
          style={[styles.input, !editable && styles.inputDisabled]}
          placeholder={placeholder}
          placeholderTextColor={COLORS.inkMuted}
          value={value}
          onChangeText={handleChangeText}
          editable={editable}
          autoCorrect={false}
        />
        {loading && (
          <ActivityIndicator size="small" color={COLORS.forestPine} style={styles.spinner} />
        )}
      </View>

      {showDropdown && (
        <View style={styles.dropdown}>
          {suggestions.length === 0 ? (
            <Text style={styles.emptyText}>{loading ? 'Searching roads...' : 'No addresses found'}</Text>
          ) : (
            suggestions.map((item, i) => (
              <TouchableOpacity
                key={`${item.lat},${item.lon},${i}`}
                style={[
                  styles.suggestionItem,
                  i === suggestions.length - 1 && styles.suggestionItemLast,
                ]}
                onPress={() => handleSelect(item)}
              >
                <Text style={styles.suggestionText} numberOfLines={2}>
                  {item.display_name}
                </Text>
              </TouchableOpacity>
            ))
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 6,
    zIndex: 10,
  },
  inputRow: {
    justifyContent: 'center',
  },
  input: {
    backgroundColor: COLORS.parchmentInput,
    borderColor: COLORS.parchmentBorder,
    borderWidth: 1.5,
    borderRadius: 5,
    paddingVertical: 9,
    paddingHorizontal: 12,
    paddingRight: 32,
    color: COLORS.inkPrimary,
    fontSize: 13,
    fontFamily: 'serif',
  },
  inputDisabled: {
    opacity: 0.7,
    backgroundColor: COLORS.parchmentSurface,
  },
  spinner: {
    position: 'absolute',
    right: 10,
  },
  dropdown: {
    position: 'absolute',
    top: '100%',
    left: 0,
    right: 0,
    backgroundColor: COLORS.parchmentCard,
    borderColor: COLORS.parchmentBorderDark,
    borderWidth: 1.5,
    borderRadius: 5,
    marginTop: 2,
    zIndex: 50,
    maxHeight: 180,
  },
  suggestionItem: {
    paddingHorizontal: 12,
    paddingVertical: 9,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.parchmentBorder,
  },
  suggestionItemLast: {
    borderBottomWidth: 0,
  },
  suggestionText: {
    color: COLORS.inkPrimary,
    fontSize: 12,
    fontFamily: 'serif',
  },
  emptyText: {
    color: COLORS.inkMuted,
    fontSize: 12,
    padding: 10,
    fontFamily: 'serif',
    fontStyle: 'italic',
  },
});
