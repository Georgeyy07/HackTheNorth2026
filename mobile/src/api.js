import Constants from 'expo-constants';

// The road_viewer FastAPI server (road_viewer/server.py) hosts both /api/potholes
// and /api/route. A phone can't use relative URLs like the web viewer does, so
// this derives the dev machine's LAN IP from Expo's own connection info when
// running via Expo Go -- no manual config needed for local dev. Falls back to
// localhost for the web/simulator case where that's actually reachable.
function resolveApiBaseUrl() {
  const hostUri =
    Constants.expoConfig?.hostUri ||
    Constants.expoGoConfig?.debuggerHost ||
    Constants.manifest2?.extra?.expoGo?.debuggerHost;
  if (hostUri) {
    const host = hostUri.split(':')[0];
    return `http://${host}:8765`;
  }
  return 'http://localhost:8765';
}

export const API_BASE_URL = resolveApiBaseUrl();

async function request(path, options = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE_URL}${path}`, { ...options, signal: controller.signal });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    return await res.json();
  } catch (err) {
    // Expo's native fetch (iOS via ExpoNativeResponse) doesn't throw a
    // standard DOM AbortError -- it throws its own error whose message says
    // "fetch request has been canceled". Check the signal itself rather
    // than relying on err.name/err.message shape, since that differs by
    // platform (confirmed: iOS surfaced the raw native message instead of
    // the friendly one here before this fix).
    if (controller.signal.aborted) {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s -- check your connection or try a closer address.`);
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

export function getPotholes(severity) {
  const query = severity && severity !== 'ALL' ? `?severity=${encodeURIComponent(severity)}` : '';
  return request(`/api/potholes${query}`);
}

export function addPothole({ latitude, longitude, severity }) {
  return request('/api/potholes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ latitude, longitude, severity }),
  });
}

export function deletePothole(id) {
  return request(`/api/potholes/${id}`, { method: 'DELETE' });
}

export function updatePothole(id, { latitude, longitude, severity }) {
  return request(`/api/potholes/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ latitude, longitude, severity }),
  });
}

export function getRoute(origin, destination, avoidanceWeight, originCoords = null, destCoords = null) {
  const params = new URLSearchParams({ origin, destination });
  if (avoidanceWeight !== undefined) params.set('avoidance_weight', String(avoidanceWeight));
  if (originCoords && originCoords.lat !== undefined && originCoords.lon !== undefined) {
    params.set('origin_lat', String(originCoords.lat));
    params.set('origin_lon', String(originCoords.lon));
  }
  if (destCoords && destCoords.lat !== undefined && destCoords.lon !== undefined) {
    params.set('dest_lat', String(destCoords.lat));
    params.set('dest_lon', String(destCoords.lon));
  }
  return request(`/api/route?${params.toString()}`);
}

export function suggestAddresses(query) {
  const params = new URLSearchParams({ q: query });
  return request(`/api/geocode/suggest?${params.toString()}`, {}, 8000);
}
