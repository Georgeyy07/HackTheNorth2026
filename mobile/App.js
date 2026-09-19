import { StatusBar } from 'expo-status-bar';
import * as Notifications from 'expo-notifications';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import PotholesScreen from './src/screens/PotholesScreen';
import RouteFinderScreen from './src/screens/RouteFinderScreen';

const Tab = createBottomTabNavigator();

// Turn notifications should show even while the app is in the foreground
// (that's when you're actually driving and looking at the map).
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
      <StatusBar style="light" />
      <Tab.Navigator
        screenOptions={{
          headerShown: false,
          tabBarStyle: { backgroundColor: '#111827', borderTopColor: '#1f2937' },
          tabBarActiveTintColor: '#00f2fe',
          tabBarInactiveTintColor: '#6b7280',
        }}
      >
        <Tab.Screen name="Potholes" component={PotholesScreen} options={{ title: 'Potholes' }} />
        <Tab.Screen name="RouteFinder" component={RouteFinderScreen} options={{ title: 'Route Finder' }} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
