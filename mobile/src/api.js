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

async function request(path, options) {
  const res = await fetch(`${API_BASE_URL}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
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

export function getRoute(origin, destination) {
  const params = new URLSearchParams({ origin, destination });
  return request(`/api/route?${params.toString()}`);
}
