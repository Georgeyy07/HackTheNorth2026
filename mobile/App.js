import React from 'react';
import { View } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as Notifications from 'expo-notifications';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import PotholesScreen from './src/screens/PotholesScreen';
import RouteFinderScreen from './src/screens/RouteFinderScreen';
import { COLORS } from './src/theme';
import MotionScreen from './src/screens/MotionScreen';
import { useTiltMonitor } from './src/motion/useTiltMonitor';
import TiltWarningOverlay from './src/components/TiltWarningOverlay';
import { NavigationProvider, useNavigationStatus } from './src/context/NavigationContext';
import { ImuStreamProvider } from './src/context/ImuStreamContext';

const Tab = createBottomTabNavigator();

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldShowBanner: true,
    shouldShowList: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

function MainApp() {
  const { isNavigating, setIsNavigating } = useNavigationStatus();
  const {
    tiltAngle,
    isWarningActive,
    threshold,
    sensorAvailable,
    simulatedAngle,
    setSimulatedAngle,
  } = useTiltMonitor({ isNavigating });

  // Warning (red UI & vibration) is strictly active ONLY when isNavigating is true
  const activeWarning = Boolean(isNavigating && isWarningActive);

  return (
    <ImuStreamProvider isNavigating={isNavigating} tiltAngle={tiltAngle} maxTiltAngle={threshold || 20}>
      <View style={{ flex: 1 }}>
      <NavigationContainer>
        <StatusBar style={activeWarning ? "light" : "dark"} />
        <Tab.Navigator
          initialRouteName="RouteFinder"
          screenOptions={{
            headerShown: false,
            tabBarStyle: {
              backgroundColor: activeWarning ? '#7f1d1d' : COLORS.parchmentSurface,
              borderTopColor: activeWarning ? '#ef4444' : COLORS.parchmentBorderDark,
              borderTopWidth: 1.5,
              height: 56,
              paddingBottom: 6,
              paddingTop: 6,
            },
            tabBarLabelStyle: {
              fontFamily: 'serif',
              fontSize: 11,
              fontWeight: '700',
              letterSpacing: 0.3,
            },
            tabBarActiveTintColor: activeWarning ? '#fca5a5' : COLORS.forestPine,
            tabBarInactiveTintColor: activeWarning ? '#fecaca' : COLORS.inkMuted,
          }}
        >
          <Tab.Screen name="Potholes" component={PotholesScreen} options={{ title: 'Potholes' }} />
          <Tab.Screen name="RouteFinder" component={RouteFinderScreen} options={{ title: 'Route Finder' }} />
          <Tab.Screen name="Motion" component={MotionScreen} options={{ title: 'Motion' }} />
        </Tab.Navigator>
      </NavigationContainer>
      <TiltWarningOverlay
        tiltAngle={tiltAngle}
        isTilted={activeWarning}
        threshold={threshold}
        sensorAvailable={sensorAvailable}
        simulatedAngle={simulatedAngle}
        onSimulateTilt={setSimulatedAngle}
        isNavigating={isNavigating}
        onToggleNavigating={() => setIsNavigating((prev) => !prev)}
      />
    </View>
    </ImuStreamProvider>
  );
}

export default function App() {
  return (
    <NavigationProvider>
      <MainApp />
    </NavigationProvider>
  );
}
