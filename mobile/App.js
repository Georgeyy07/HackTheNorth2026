import { StatusBar } from 'expo-status-bar';
import * as Notifications from 'expo-notifications';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import PotholesScreen from './src/screens/PotholesScreen';
import RouteFinderScreen from './src/screens/RouteFinderScreen';
import { COLORS } from './src/theme';
import MotionScreen from './src/screens/MotionScreen';

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

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="dark" />
      <Tab.Navigator
        initialRouteName="RouteFinder"
        screenOptions={{
          headerShown: false,
          tabBarStyle: {
            backgroundColor: COLORS.parchmentSurface,
            borderTopColor: COLORS.parchmentBorderDark,
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
          tabBarActiveTintColor: COLORS.forestPine,
          tabBarInactiveTintColor: COLORS.inkMuted,
        }}
      >
        <Tab.Screen name="Potholes" component={PotholesScreen} options={{ title: 'Potholes' }} />
        <Tab.Screen name="RouteFinder" component={RouteFinderScreen} options={{ title: 'Route Finder' }} />
        <Tab.Screen name="Motion" component={MotionScreen} options={{ title: 'Motion' }} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
