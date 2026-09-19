import { Platform } from 'react-native';
import { Pothole, NewPotholePayload, UpdatePotholePayload, SeverityLevel } from '../types/pothole';

// Default base URL: FastAPI backend runs on port 8765
const DEFAULT_HOST = 'http://10.37.104.116:8765';

let currentBaseUrl = DEFAULT_HOST;

export const setApiBaseUrl = (url: string) => {
  currentBaseUrl = url.trim().replace(/\/+$/, '');
};

export const getApiBaseUrl = () => currentBaseUrl;

export const fetchPotholes = async (severity?: SeverityLevel | null): Promise<Pothole[]> => {
  try {
    const query = severity ? `?severity=${encodeURIComponent(severity)}` : '';
    const res = await fetch(`${currentBaseUrl}/api/potholes${query}`);
    if (!res.ok) {
      throw new Error(`Server returned ${res.status}: ${res.statusText}`);
    }
    const data = await res.json();
    return Array.isArray(data) ? data : [];
  } catch (error) {
    console.error('Error fetching potholes:', error);
    throw error;
  }
};

export const createPothole = async (payload: NewPotholePayload): Promise<Pothole> => {
  const res = await fetch(`${currentBaseUrl}/api/potholes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Failed to create pothole (${res.status})`);
  }
  return await res.json();
};

export const updatePothole = async (
  potholeId: number,
  payload: UpdatePotholePayload
): Promise<{ status: string; id: number }> => {
  const res = await fetch(`${currentBaseUrl}/api/potholes/${potholeId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Failed to update pothole (${res.status})`);
  }
  return await res.json();
};

export const deletePothole = async (potholeId: number): Promise<{ status: string; deleted_id: number }> => {
  const res = await fetch(`${currentBaseUrl}/api/potholes/${potholeId}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    throw new Error(`Failed to delete pothole (${res.status})`);
  }
  return await res.json();
};

export const seedPotholes = async (): Promise<{ status: string; potholes: Pothole[] }> => {
  const res = await fetch(`${currentBaseUrl}/api/potholes/seed`, {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error(`Failed to seed potholes (${res.status})`);
  }
  return await res.json();
};
