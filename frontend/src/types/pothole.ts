export type SeverityLevel = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

export interface Pothole {
  id: number;
  latitude: number;
  longitude: number;
  severity: SeverityLevel;
  timestamp: string;
}

export interface NewPotholePayload {
  latitude: number;
  longitude: number;
  severity: SeverityLevel;
}

export interface UpdatePotholePayload {
  severity?: SeverityLevel;
  latitude?: number;
  longitude?: number;
}
