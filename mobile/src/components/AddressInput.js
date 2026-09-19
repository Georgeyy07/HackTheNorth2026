import React, { useState, useRef, useEffect } from 'react';
import { View, TextInput, TouchableOpacity, Text, StyleSheet, ActivityIndicator } from 'react-native';

import { suggestAddresses } from '../api';

const DEBOUNCE_MS = 400;
const MIN_QUERY_LENGTH = 3;

// A plain text input with a dropdown of address suggestions that appears as
// you type (debounced, so it doesn't fire a network request per keystroke).
// Selecting a suggestion fills the input with its full display name.
export default function AddressInput({ value, onChangeText, placeholder, editable = true }) {
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const debounceRef = useRef(null);
  const requestIdRef = useRef(0);

  useEffect(() => () => clearTimeout(debounceRef.current), []);

  const handleChangeText = (text) => {
    onChangeText(text);
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
    setSuggestions([]);
    setDismissed(true);
  };

  const showDropdown = !dismissed && (loading || suggestions.length > 0);

  return (
    <View style={styles.container}>
      <View style={styles.inputRow}>
        <TextInput
          style={styles.input}
          placeholder={placeholder}
          placeholderTextColor="#6b7280"
          value={value}
          onChangeText={handleChangeText}
          editable={editable}
          autoCorrect={false}
        />
        {loading && <ActivityIndicator size="small" color="#00f2fe" style={styles.spinner} />}
      </View>

      {showDropdown && (
        <View style={styles.dropdown}>
          {suggestions.length === 0 ? (
            <Text style={styles.emptyText}>{loading ? 'Searching…' : 'No matches'}</Text>
          ) : (
            suggestions.map((item, i) => (
              <TouchableOpacity
                key={`${item.lat},${item.lon},${i}`}
                style={styles.suggestionItem}
                onPress={() => handleSelect(item)}
              >
                <Text style={styles.suggestionText} numberOfLines={2}>{item.display_name}</Text>
              </TouchableOpacity>
            ))
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { marginBottom: 8 },
  inputRow: { justifyContent: 'center' },
  input: {
    backgroundColor: '#1f2937', borderColor: '#374151', borderWidth: 1, borderRadius: 8,
    padding: 12, paddingRight: 36, color: '#fff', fontSize: 13,
  },
  spinner: { position: 'absolute', right: 12 },
  dropdown: {
    backgroundColor: '#1f2937', borderColor: '#374151', borderWidth: 1, borderRadius: 8,
    marginTop: 4, overflow: 'hidden',
  },
  suggestionItem: { paddingHorizontal: 12, paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: '#374151' },
  suggestionText: { color: '#e6f4fe', fontSize: 12 },
  emptyText: { color: '#6b7280', fontSize: 12, padding: 12 },
});
