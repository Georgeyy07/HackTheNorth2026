import { registerRootComponent } from 'expo';
import * as Sentry from '@sentry/react-native';

import App from './App';
import { API_BASE_URL } from './src/api';

// Error monitoring + Tracing + Session Replay, set up once here before
// anything else runs. EXPO_PUBLIC_SENTRY_DSN is read at build time by Expo's
// built-in env var support; if it's unset (e.g. local dev without a DSN
// configured yet), Sentry.init just no-ops instead of throwing, so this is
// always safe to leave in.
Sentry.init({
  dsn: process.env.EXPO_PUBLIC_SENTRY_DSN,
  // Tracing: capture performance data for every session. For a hackathon
  // demo this is fine; a real production app would sample this down.
  tracesSampleRate: 1.0,
  // Distributed tracing: only attach the sentry-trace/baggage headers (which
  // link a frontend action to the backend work it caused) to requests aimed
  // at our own backend -- not every random third-party fetch the app makes.
  tracePropagationTargets: [API_BASE_URL],
  integrations: [Sentry.mobileReplayIntegration()],
  // Session Replay: most sessions are unremarkable and not worth recording,
  // but any session that hits an error gets fully captured so we can watch
  // exactly what the user did right before things broke.
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,
});

// registerRootComponent calls AppRegistry.registerComponent('main', () => App);
// It also ensures that whether you load the app in Expo Go or in a native build,
// the environment is set up appropriately.
// Sentry.wrap adds a root-level error boundary + navigation tracing.
registerRootComponent(Sentry.wrap(App));
