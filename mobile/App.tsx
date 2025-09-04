/**
 * A2Z TSN/FRER Mobile Monitoring App
 * React Native application for real-time network monitoring
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
  Dimensions,
  Alert,
  RefreshControl,
  ActivityIndicator,
  Animated,
  Platform,
  Vibration,
} from 'react-native';
import {
  LineChart,
  BarChart,
  PieChart,
  ProgressChart,
} from 'react-native-chart-kit';
import LinearGradient from 'react-native-linear-gradient';
import Icon from 'react-native-vector-icons/MaterialIcons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import PushNotification from 'react-native-push-notification';
import { io, Socket } from 'socket.io-client';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// API Configuration
const API_BASE_URL = 'https://a2z-tsn.autoa2z.com/api/v1';
const WEBSOCKET_URL = 'wss://a2z-tsn.autoa2z.com';

// Color Theme
const colors = {
  primary: '#00ff88',
  secondary: '#0088ff',
  danger: '#ff4488',
  warning: '#ffcc00',
  background: '#0a0e1a',
  cardBg: '#1a1f2e',
  text: '#ffffff',
  textSecondary: '#888',
  success: '#2ecc71',
};

// Interfaces
interface NetworkStatus {
  isConnected: boolean;
  bandwidth: number;
  latency: number;
  packetLoss: number;
  availability: number;
}

interface FRERStream {
  id: number;
  name: string;
  status: 'active' | 'degraded' | 'failed';
  bandwidth: number;
  paths: number;
  activePaths: number;
  recoveryTime: number;
  lastRecovery: Date | null;
}

interface SwitchInfo {
  id: string;
  name: string;
  model: string;
  status: 'online' | 'offline' | 'warning';
  temperature: number;
  cpuUsage: number;
  memoryUsage: number;
  uptime: number;
  ports: PortInfo[];
}

interface PortInfo {
  id: number;
  status: 'up' | 'down';
  speed: number;
  errors: number;
  traffic: number;
}

interface Alert {
  id: string;
  timestamp: Date;
  level: 'critical' | 'error' | 'warning' | 'info';
  component: string;
  message: string;
  acknowledged: boolean;
}

// Main App Component
const App: React.FC = () => {
  // State Management
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus>({
    isConnected: true,
    bandwidth: 561,
    latency: 0.8,
    packetLoss: 0.00001,
    availability: 99.97,
  });

  const [frerStreams, setFrerStreams] = useState<FRERStream[]>([
    {
      id: 1001,
      name: 'LiDAR System',
      status: 'active',
      bandwidth: 100,
      paths: 2,
      activePaths: 2,
      recoveryTime: 12.3,
      lastRecovery: null,
    },
    {
      id: 1002,
      name: 'Camera Array',
      status: 'active',
      bandwidth: 400,
      paths: 2,
      activePaths: 2,
      recoveryTime: 13.5,
      lastRecovery: null,
    },
    {
      id: 1003,
      name: 'Emergency Brake',
      status: 'active',
      bandwidth: 1,
      paths: 3,
      activePaths: 3,
      recoveryTime: 8.1,
      lastRecovery: null,
    },
    {
      id: 1004,
      name: 'Steering Control',
      status: 'active',
      bandwidth: 10,
      paths: 2,
      activePaths: 2,
      recoveryTime: 9.8,
      lastRecovery: null,
    },
  ]);

  const [switches, setSwitches] = useState<SwitchInfo[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedTab, setSelectedTab] = useState<'dashboard' | 'streams' | 'switches' | 'alerts'>('dashboard');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const socket = useRef<Socket | null>(null);

  // Initialize App
  useEffect(() => {
    initializeApp();
    setupWebSocket();
    setupNotifications();
    startAnimations();

    return () => {
      if (socket.current) {
        socket.current.disconnect();
      }
    };
  }, []);

  const initializeApp = async () => {
    try {
      // Check network connectivity
      const netState = await NetInfo.fetch();
      setNetworkStatus(prev => ({ ...prev, isConnected: netState.isConnected || false }));

      // Load saved preferences
      const savedPrefs = await AsyncStorage.getItem('userPreferences');
      if (savedPrefs) {
        // Apply saved preferences
        const prefs = JSON.parse(savedPrefs);
        // Apply theme, notification settings, etc.
      }

      // Fetch initial data
      await fetchAllData();
      
      setIsLoading(false);
    } catch (error) {
      console.error('Initialization error:', error);
      Alert.alert('Error', 'Failed to initialize app');
    }
  };

  const setupWebSocket = () => {
    socket.current = io(WEBSOCKET_URL, {
      transports: ['websocket'],
      reconnectionDelay: 1000,
      reconnection: true,
      reconnectionAttempts: 5,
    });

    socket.current.on('connect', () => {
      console.log('WebSocket connected');
    });

    socket.current.on('networkUpdate', (data: any) => {
      updateNetworkStatus(data);
    });

    socket.current.on('streamUpdate', (data: any) => {
      updateFrerStreams(data);
    });

    socket.current.on('alert', (alert: Alert) => {
      handleNewAlert(alert);
    });

    socket.current.on('disconnect', () => {
      console.log('WebSocket disconnected');
    });
  };

  const setupNotifications = () => {
    PushNotification.configure({
      onRegister: (token) => {
        console.log('TOKEN:', token);
      },
      onNotification: (notification) => {
        console.log('NOTIFICATION:', notification);
        // Handle notification tap
      },
      permissions: {
        alert: true,
        badge: true,
        sound: true,
      },
      popInitialNotification: true,
      requestPermissions: Platform.OS === 'ios',
    });

    // Create notification channel for Android
    if (Platform.OS === 'android') {
      PushNotification.createChannel(
        {
          channelId: 'a2z-alerts',
          channelName: 'A2Z Network Alerts',
          channelDescription: 'Critical network and FRER alerts',
          importance: 4,
          vibrate: true,
        },
        (created) => console.log(`Channel created: ${created}`)
      );
    }
  };

  const startAnimations = () => {
    // Fade in animation
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 1000,
      useNativeDriver: true,
    }).start();

    // Pulse animation for critical indicators
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.2,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const fetchAllData = async () => {
    try {
      const [statusRes, streamsRes, switchesRes, alertsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/network/status`),
        fetch(`${API_BASE_URL}/frer/streams`),
        fetch(`${API_BASE_URL}/switches`),
        fetch(`${API_BASE_URL}/alerts?limit=10`),
      ]);

      if (statusRes.ok) {
        const statusData = await statusRes.json();
        setNetworkStatus(statusData);
      }

      if (streamsRes.ok) {
        const streamsData = await streamsRes.json();
        setFrerStreams(streamsData);
      }

      if (switchesRes.ok) {
        const switchesData = await switchesRes.json();
        setSwitches(switchesData);
      }

      if (alertsRes.ok) {
        const alertsData = await alertsRes.json();
        setAlerts(alertsData);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const updateNetworkStatus = (data: Partial<NetworkStatus>) => {
    setNetworkStatus(prev => ({ ...prev, ...data }));
  };

  const updateFrerStreams = (data: FRERStream[]) => {
    setFrerStreams(data);
  };

  const handleNewAlert = (alert: Alert) => {
    setAlerts(prev => [alert, ...prev].slice(0, 50)); // Keep last 50 alerts

    // Show notification for critical alerts
    if (alert.level === 'critical') {
      Vibration.vibrate([0, 500, 200, 500]);
      
      PushNotification.localNotification({
        channelId: 'a2z-alerts',
        title: '🚨 Critical Alert',
        message: alert.message,
        bigText: `${alert.component}: ${alert.message}`,
        color: colors.danger,
        vibrate: true,
        vibration: 300,
        priority: 'high',
        importance: 'high',
      });
    }
  };

  const onRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await fetchAllData();
    setIsRefreshing(false);
  }, []);

  // Dashboard Component
  const Dashboard = () => (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl
          refreshing={isRefreshing}
          onRefresh={onRefresh}
          tintColor={colors.primary}
        />
      }
    >
      {/* System Overview Card */}
      <LinearGradient
        colors={['#1a1f2e', '#2a2f3e']}
        style={styles.card}
      >
        <Text style={styles.cardTitle}>System Overview</Text>
        <View style={styles.statsGrid}>
          <View style={styles.statItem}>
            <Icon name="network-check" size={24} color={colors.primary} />
            <Text style={styles.statValue}>{networkStatus.availability.toFixed(2)}%</Text>
            <Text style={styles.statLabel}>Availability</Text>
          </View>
          <View style={styles.statItem}>
            <Icon name="speed" size={24} color={colors.secondary} />
            <Text style={styles.statValue}>{networkStatus.bandwidth} Mbps</Text>
            <Text style={styles.statLabel}>Bandwidth</Text>
          </View>
          <View style={styles.statItem}>
            <Icon name="timer" size={24} color={colors.warning} />
            <Text style={styles.statValue}>{networkStatus.latency.toFixed(1)} ms</Text>
            <Text style={styles.statLabel}>Latency</Text>
          </View>
          <View style={styles.statItem}>
            <Icon name="error-outline" size={24} color={colors.success} />
            <Text style={styles.statValue}>{(networkStatus.packetLoss * 100).toFixed(5)}%</Text>
            <Text style={styles.statLabel}>Packet Loss</Text>
          </View>
        </View>
      </LinearGradient>

      {/* Bandwidth Usage Chart */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Bandwidth Usage</Text>
        <LineChart
          data={{
            labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
            datasets: [{
              data: [450, 480, 520, 561, 540, 510],
              color: (opacity = 1) => colors.primary,
              strokeWidth: 2,
            }],
          }}
          width={screenWidth - 40}
          height={200}
          chartConfig={{
            backgroundColor: colors.cardBg,
            backgroundGradientFrom: colors.cardBg,
            backgroundGradientTo: colors.cardBg,
            decimalPlaces: 0,
            color: (opacity = 1) => colors.primary,
            labelColor: (opacity = 1) => colors.textSecondary,
            style: {
              borderRadius: 16,
            },
            propsForDots: {
              r: '4',
              strokeWidth: '2',
              stroke: colors.primary,
            },
          }}
          bezier
          style={styles.chart}
        />
      </View>

      {/* FRER Performance */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>FRER Recovery Performance</Text>
        <BarChart
          data={{
            labels: ['1001', '1002', '1003', '1004'],
            datasets: [{
              data: frerStreams.map(s => s.recoveryTime),
            }],
          }}
          width={screenWidth - 40}
          height={180}
          yAxisLabel=""
          yAxisSuffix="ms"
          chartConfig={{
            backgroundColor: colors.cardBg,
            backgroundGradientFrom: colors.cardBg,
            backgroundGradientTo: colors.cardBg,
            decimalPlaces: 1,
            color: (opacity = 1) => colors.secondary,
            labelColor: (opacity = 1) => colors.textSecondary,
            barPercentage: 0.7,
          }}
          style={styles.chart}
        />
      </View>

      {/* Quick Actions */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Quick Actions</Text>
        <View style={styles.actionGrid}>
          <TouchableOpacity style={styles.actionButton} onPress={() => testFailover()}>
            <Icon name="sync-problem" size={32} color={colors.warning} />
            <Text style={styles.actionText}>Test Failover</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton} onPress={() => runDiagnostics()}>
            <Icon name="analytics" size={32} color={colors.secondary} />
            <Text style={styles.actionText}>Diagnostics</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton} onPress={() => exportReport()}>
            <Icon name="file-download" size={32} color={colors.success} />
            <Text style={styles.actionText}>Export Report</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton} onPress={() => emergencyStop()}>
            <Icon name="stop-circle" size={32} color={colors.danger} />
            <Text style={styles.actionText}>Emergency</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );

  // FRER Streams Component
  const StreamsView = () => (
    <ScrollView style={styles.container}>
      {frerStreams.map((stream) => (
        <Animated.View 
          key={stream.id}
          style={[
            styles.card,
            stream.status === 'degraded' && { borderColor: colors.warning, borderWidth: 2 },
            stream.status === 'failed' && { borderColor: colors.danger, borderWidth: 2 },
          ]}
        >
          <View style={styles.streamHeader}>
            <View>
              <Text style={styles.streamId}>Stream {stream.id}</Text>
              <Text style={styles.streamName}>{stream.name}</Text>
            </View>
            <View style={[
              styles.statusBadge,
              stream.status === 'active' && styles.statusActive,
              stream.status === 'degraded' && styles.statusDegraded,
              stream.status === 'failed' && styles.statusFailed,
            ]}>
              <Text style={styles.statusText}>{stream.status.toUpperCase()}</Text>
            </View>
          </View>

          <View style={styles.streamMetrics}>
            <View style={styles.metricItem}>
              <Text style={styles.metricLabel}>Bandwidth</Text>
              <Text style={styles.metricValue}>{stream.bandwidth} Mbps</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.metricLabel}>Paths</Text>
              <Text style={styles.metricValue}>{stream.activePaths}/{stream.paths}</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.metricLabel}>Recovery</Text>
              <Text style={styles.metricValue}>{stream.recoveryTime} ms</Text>
            </View>
          </View>

          {/* Path Visualization */}
          <View style={styles.pathVisualization}>
            {Array.from({ length: stream.paths }).map((_, index) => (
              <View
                key={index}
                style={[
                  styles.pathLine,
                  index < stream.activePaths ? styles.pathActive : styles.pathInactive,
                ]}
              />
            ))}
          </View>
        </Animated.View>
      ))}
    </ScrollView>
  );

  // Switches Component
  const SwitchesView = () => (
    <ScrollView style={styles.container}>
      {switches.map((sw) => (
        <View key={sw.id} style={styles.card}>
          <View style={styles.switchHeader}>
            <View>
              <Text style={styles.switchName}>{sw.name}</Text>
              <Text style={styles.switchModel}>{sw.model}</Text>
            </View>
            <Icon
              name={sw.status === 'online' ? 'check-circle' : 'error'}
              size={32}
              color={sw.status === 'online' ? colors.success : colors.danger}
            />
          </View>

          <View style={styles.switchMetrics}>
            <View style={styles.metricRow}>
              <Icon name="thermostat" size={20} color={colors.textSecondary} />
              <Text style={styles.metricLabel}>Temperature</Text>
              <Text style={[
                styles.metricValue,
                sw.temperature > 70 && { color: colors.warning },
                sw.temperature > 80 && { color: colors.danger },
              ]}>{sw.temperature}°C</Text>
            </View>
            <View style={styles.metricRow}>
              <Icon name="memory" size={20} color={colors.textSecondary} />
              <Text style={styles.metricLabel}>CPU Usage</Text>
              <Text style={styles.metricValue}>{sw.cpuUsage}%</Text>
            </View>
            <View style={styles.metricRow}>
              <Icon name="storage" size={20} color={colors.textSecondary} />
              <Text style={styles.metricLabel}>Memory</Text>
              <Text style={styles.metricValue}>{sw.memoryUsage}%</Text>
            </View>
            <View style={styles.metricRow}>
              <Icon name="schedule" size={20} color={colors.textSecondary} />
              <Text style={styles.metricLabel}>Uptime</Text>
              <Text style={styles.metricValue}>{Math.floor(sw.uptime / 86400)}d</Text>
            </View>
          </View>

          {/* Port Status Grid */}
          <View style={styles.portGrid}>
            {sw.ports.map((port) => (
              <View
                key={port.id}
                style={[
                  styles.port,
                  port.status === 'up' ? styles.portUp : styles.portDown,
                ]}
              >
                <Text style={styles.portNumber}>{port.id}</Text>
              </View>
            ))}
          </View>
        </View>
      ))}
    </ScrollView>
  );

  // Alerts Component
  const AlertsView = () => (
    <ScrollView style={styles.container}>
      {alerts.map((alert) => (
        <TouchableOpacity
          key={alert.id}
          style={[
            styles.alertCard,
            alert.level === 'critical' && styles.alertCritical,
            alert.level === 'error' && styles.alertError,
            alert.level === 'warning' && styles.alertWarning,
          ]}
          onPress={() => acknowledgeAlert(alert.id)}
        >
          <View style={styles.alertHeader}>
            <Icon
              name={
                alert.level === 'critical' ? 'error' :
                alert.level === 'error' ? 'warning' :
                alert.level === 'warning' ? 'info' : 'check-circle'
              }
              size={24}
              color={
                alert.level === 'critical' ? colors.danger :
                alert.level === 'error' ? colors.danger :
                alert.level === 'warning' ? colors.warning : colors.success
              }
            />
            <Text style={styles.alertTime}>
              {new Date(alert.timestamp).toLocaleTimeString()}
            </Text>
          </View>
          <Text style={styles.alertComponent}>{alert.component}</Text>
          <Text style={styles.alertMessage}>{alert.message}</Text>
          {!alert.acknowledged && (
            <Text style={styles.acknowledgeButton}>Tap to acknowledge</Text>
          )}
        </TouchableOpacity>
      ))}
    </ScrollView>
  );

  // Action Functions
  const testFailover = async () => {
    Alert.alert(
      'Test Failover',
      'This will simulate a path failure. Continue?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Proceed',
          onPress: async () => {
            try {
              const response = await fetch(`${API_BASE_URL}/test/failover`, {
                method: 'POST',
              });
              if (response.ok) {
                Alert.alert('Success', 'Failover test initiated');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to initiate test');
            }
          },
        },
      ]
    );
  };

  const runDiagnostics = async () => {
    Alert.alert('Diagnostics', 'Running full system diagnostics...');
    // Implementation
  };

  const exportReport = async () => {
    Alert.alert('Export', 'Report exported to email');
    // Implementation
  };

  const emergencyStop = () => {
    Alert.alert(
      '⚠️ Emergency Stop',
      'This will stop all FRER streams. Are you sure?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'STOP',
          style: 'destructive',
          onPress: async () => {
            Vibration.vibrate(1000);
            // Implementation
          },
        },
      ]
    );
  };

  const acknowledgeAlert = async (alertId: string) => {
    setAlerts(prev =>
      prev.map(a => a.id === alertId ? { ...a, acknowledged: true } : a)
    );
    // Send acknowledgment to server
  };

  // Loading Screen
  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.primary} />
        <Text style={styles.loadingText}>Initializing A2Z Network Monitor...</Text>
      </View>
    );
  }

  // Main Render
  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor={colors.background} />
      
      {/* Header */}
      <LinearGradient
        colors={[colors.cardBg, colors.background]}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>A2Z TSN Monitor</Text>
          <View style={styles.connectionStatus}>
            <View style={[
              styles.connectionDot,
              networkStatus.isConnected ? styles.connected : styles.disconnected,
            ]} />
            <Text style={styles.connectionText}>
              {networkStatus.isConnected ? 'Connected' : 'Offline'}
            </Text>
          </View>
        </View>
      </LinearGradient>

      {/* Content */}
      <Animated.View style={{ flex: 1, opacity: fadeAnim }}>
        {selectedTab === 'dashboard' && <Dashboard />}
        {selectedTab === 'streams' && <StreamsView />}
        {selectedTab === 'switches' && <SwitchesView />}
        {selectedTab === 'alerts' && <AlertsView />}
      </Animated.View>

      {/* Tab Navigation */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, selectedTab === 'dashboard' && styles.tabActive]}
          onPress={() => setSelectedTab('dashboard')}
        >
          <Icon name="dashboard" size={24} color={selectedTab === 'dashboard' ? colors.primary : colors.textSecondary} />
          <Text style={[styles.tabText, selectedTab === 'dashboard' && styles.tabTextActive]}>Dashboard</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.tab, selectedTab === 'streams' && styles.tabActive]}
          onPress={() => setSelectedTab('streams')}
        >
          <Icon name="timeline" size={24} color={selectedTab === 'streams' ? colors.primary : colors.textSecondary} />
          <Text style={[styles.tabText, selectedTab === 'streams' && styles.tabTextActive]}>Streams</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.tab, selectedTab === 'switches' && styles.tabActive]}
          onPress={() => setSelectedTab('switches')}
        >
          <Icon name="router" size={24} color={selectedTab === 'switches' ? colors.primary : colors.textSecondary} />
          <Text style={[styles.tabText, selectedTab === 'switches' && styles.tabTextActive]}>Switches</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.tab, selectedTab === 'alerts' && styles.tabActive]}
          onPress={() => setSelectedTab('alerts')}
        >
          <Icon name="notifications" size={24} color={selectedTab === 'alerts' ? colors.primary : colors.textSecondary} />
          <Text style={[styles.tabText, selectedTab === 'alerts' && styles.tabTextActive]}>
            Alerts
            {alerts.filter(a => !a.acknowledged).length > 0 && (
              <View style={styles.badge}>
                <Text style={styles.badgeText}>{alerts.filter(a => !a.acknowledged).length}</Text>
              </View>
            )}
          </Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

// Styles
const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: colors.background,
  },
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background,
  },
  loadingText: {
    color: colors.text,
    marginTop: 20,
    fontSize: 16,
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: colors.primary + '30',
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.primary,
  },
  connectionStatus: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  connectionDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  connected: {
    backgroundColor: colors.success,
  },
  disconnected: {
    backgroundColor: colors.danger,
  },
  connectionText: {
    color: colors.textSecondary,
    fontSize: 12,
  },
  card: {
    backgroundColor: colors.cardBg,
    margin: 10,
    padding: 15,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: colors.text,
    marginBottom: 15,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  statItem: {
    width: '48%',
    alignItems: 'center',
    marginBottom: 15,
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: colors.text,
    marginTop: 5,
  },
  statLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    marginTop: 2,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  actionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  actionButton: {
    width: '48%',
    alignItems: 'center',
    padding: 15,
    backgroundColor: colors.background,
    borderRadius: 10,
    marginBottom: 10,
  },
  actionText: {
    color: colors.text,
    marginTop: 5,
    fontSize: 12,
  },
  streamHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  streamId: {
    fontSize: 16,
    fontWeight: 'bold',
    color: colors.secondary,
  },
  streamName: {
    fontSize: 14,
    color: colors.textSecondary,
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusActive: {
    backgroundColor: colors.success + '30',
  },
  statusDegraded: {
    backgroundColor: colors.warning + '30',
  },
  statusFailed: {
    backgroundColor: colors.danger + '30',
  },
  statusText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: colors.text,
  },
  streamMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 15,
  },
  metricItem: {
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 11,
    color: colors.textSecondary,
    marginBottom: 2,
  },
  metricValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: colors.text,
  },
  pathVisualization: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    height: 4,
  },
  pathLine: {
    flex: 1,
    height: 4,
    marginHorizontal: 2,
    borderRadius: 2,
  },
  pathActive: {
    backgroundColor: colors.success,
  },
  pathInactive: {
    backgroundColor: colors.textSecondary,
  },
  switchHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  switchName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: colors.text,
  },
  switchModel: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  switchMetrics: {
    marginBottom: 15,
  },
  metricRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  portGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  port: {
    width: 30,
    height: 30,
    margin: 2,
    borderRadius: 4,
    justifyContent: 'center',
    alignItems: 'center',
  },
  portUp: {
    backgroundColor: colors.success + '50',
  },
  portDown: {
    backgroundColor: colors.textSecondary + '30',
  },
  portNumber: {
    fontSize: 10,
    color: colors.text,
    fontWeight: 'bold',
  },
  alertCard: {
    backgroundColor: colors.cardBg,
    margin: 10,
    padding: 15,
    borderRadius: 10,
    borderLeftWidth: 4,
    borderLeftColor: colors.textSecondary,
  },
  alertCritical: {
    borderLeftColor: colors.danger,
    backgroundColor: colors.danger + '10',
  },
  alertError: {
    borderLeftColor: colors.danger,
  },
  alertWarning: {
    borderLeftColor: colors.warning,
  },
  alertHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  alertTime: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  alertComponent: {
    fontSize: 12,
    color: colors.secondary,
    marginBottom: 4,
  },
  alertMessage: {
    fontSize: 14,
    color: colors.text,
    lineHeight: 20,
  },
  acknowledgeButton: {
    fontSize: 12,
    color: colors.primary,
    marginTop: 8,
    fontWeight: 'bold',
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: colors.cardBg,
    borderTopWidth: 1,
    borderTopColor: colors.primary + '30',
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
  },
  tabActive: {
    backgroundColor: colors.primary + '10',
  },
  tabText: {
    fontSize: 11,
    color: colors.textSecondary,
    marginTop: 4,
  },
  tabTextActive: {
    color: colors.primary,
    fontWeight: 'bold',
  },
  badge: {
    position: 'absolute',
    top: -8,
    right: -8,
    backgroundColor: colors.danger,
    borderRadius: 10,
    minWidth: 20,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  badgeText: {
    color: colors.text,
    fontSize: 10,
    fontWeight: 'bold',
  },
});

export default App;