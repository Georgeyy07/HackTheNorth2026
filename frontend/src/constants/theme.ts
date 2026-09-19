import { SeverityLevel } from '../types/pothole';

export const COLORS = {
  bgDark: '#08120e',
  bgCard: '#0f221a',
  bgCardElevated: '#152f24',
  border: '#1d4435',
  borderLight: '#2a5a47',
  text: '#e6f3ee',
  textMuted: '#8ba69b',
  emerald: '#13795b',
  emeraldLight: '#238b68',
  emeraldGlow: '#4ef2bc',
  cyan: '#00f2fe',
  coral: '#ff4d6d',
  amber: '#ff9f1c',
  yellow: '#ffe66d',
  blue: '#4ea8de',
};

export const SEVERITY_COLORS: Record<SeverityLevel, { bg: string; text: string; pin: string }> = {
  CRITICAL: { bg: '#ff4d6d', text: '#ffffff', pin: '#ff4d6d' },
  HIGH: { bg: '#ff9f1c', text: '#000000', pin: '#ff9f1c' },
  MEDIUM: { bg: '#ffe66d', text: '#000000', pin: '#ffe66d' },
  LOW: { bg: '#4ea8de', text: '#ffffff', pin: '#4ea8de' },
};
