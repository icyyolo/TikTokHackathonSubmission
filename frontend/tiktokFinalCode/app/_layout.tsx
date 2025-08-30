import { Stack, Tabs } from 'expo-router';
import { StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

export default function Layout() {
    return (
        <SafeAreaProvider>
            {/* <StatusBar barStyle="dark-content" />
            <Stack
                screenOptions={{
                    headerShown: false,
                    contentStyle: { backgroundColor: '#fff' },
                }}
            /> */}
            <Tabs>
                <Tabs.Screen name='index'></Tabs.Screen>
                <Tabs.Screen name='calibration'></Tabs.Screen>
            </Tabs>
        </SafeAreaProvider>
    );
}
