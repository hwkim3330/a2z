import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Alert,
  RefreshControl,
  StatusBar,
  Dimensions
} from 'react-native';
import {
  Card,
  Button,
  Badge,
  Avatar,
  ListItem,
  Header,
  LinearProgress
} from 'react-native-elements';
import Icon from 'react-native-vector-icons/MaterialIcons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { LineChart, PieChart } from 'react-native-chart-kit';
import moment from 'moment';
import 'moment/locale/ko';

// 한국어 설정
moment.locale('ko');

interface NetworkMetrics {
  timestamp: string;
  bandwidth_utilization: number;
  latency: number;
  packet_loss: number;
  frer_recoveries: number;
  availability: number;
}

interface Alert {
  id: string;
  severity: '정보' | '경고' | '오류' | '심각';
  title: string;
  description: string;
  timestamp: string;
  acknowledged: boolean;
  switch_id: string;
}

interface SwitchStatus {
  id: string;
  name: string;
  status: '정상' | '경고' | '오류' | '오프라인';
  location: string;
  uptime: number;
  temperature: number;
  port_utilization: number;
}

const DashboardScreen: React.FC = () => {
  const [metrics, setMetrics] = useState<NetworkMetrics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [switches, setSwitches] = useState<SwitchStatus[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [historicalData, setHistoricalData] = useState<number[]>([]);

  const screenWidth = Dimensions.get('window').width;

  useEffect(() => {
    loadInitialData();
    
    // 웹소켓 연결
    connectWebSocket();
    
    // 주기적 업데이트
    const interval = setInterval(loadMetrics, 5000);
    
    return () => {
      clearInterval(interval);
    };
  }, []);

  const loadInitialData = async () => {
    try {
      await Promise.all([
        loadMetrics(),
        loadAlerts(),
        loadSwitches()
      ]);
    } catch (error) {
      console.error('초기 데이터 로드 실패:', error);
      Alert.alert('오류', '초기 데이터를 불러올 수 없습니다.');
    }
  };

  const loadMetrics = async () => {
    try {
      const token = await AsyncStorage.getItem('auth_token');
      const response = await fetch('https://api.a2z-tsn.com/v2/monitoring/metrics', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept-Language': 'ko-KR'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setMetrics(data.current);
        
        // 히스토리 데이터 업데이트
        if (data.history) {
          setHistoricalData(data.history.map((item: any) => item.bandwidth_utilization));
        }
        
        setIsConnected(true);
      } else {
        throw new Error('메트릭 데이터 로드 실패');
      }
    } catch (error) {
      console.error('메트릭 로드 오류:', error);
      setIsConnected(false);
    }
  };

  const loadAlerts = async () => {
    try {
      const token = await AsyncStorage.getItem('auth_token');
      const response = await fetch('https://api.a2z-tsn.com/v2/monitoring/alerts?acknowledged=false', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept-Language': 'ko-KR'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setAlerts(data.alerts.slice(0, 5)); // 최근 5개만 표시
      }
    } catch (error) {
      console.error('알림 로드 오류:', error);
    }
  };

  const loadSwitches = async () => {
    try {
      const token = await AsyncStorage.getItem('auth_token');
      const response = await fetch('https://api.a2z-tsn.com/v2/network/switches', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept-Language': 'ko-KR'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setSwitches(data.data);
      }
    } catch (error) {
      console.error('스위치 상태 로드 오류:', error);
    }
  };

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket('wss://ws.a2z-tsn.com');
      
      ws.onopen = () => {
        console.log('웹소켓 연결됨');
        setIsConnected(true);
        
        // 실시간 업데이트 구독
        ws.send(JSON.stringify({
          type: 'subscribe',
          streams: ['metrics', 'alerts', 'switches']
        }));
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'metrics':
            setMetrics(data.data);
            break;
          case 'alert':
            handleNewAlert(data.data);
            break;
          case 'switch_status':
            updateSwitchStatus(data.data);
            break;
        }
      };
      
      ws.onerror = (error) => {
        console.error('웹소켓 오류:', error);
        setIsConnected(false);
      };
      
      ws.onclose = () => {
        console.log('웹소켓 연결 해제');
        setIsConnected(false);
        
        // 재연결 시도
        setTimeout(connectWebSocket, 5000);
      };
    } catch (error) {
      console.error('웹소켓 연결 실패:', error);
    }
  };

  const handleNewAlert = (alert: Alert) => {
    setAlerts(prev => [alert, ...prev.slice(0, 4)]);
    
    // 심각한 알림은 팝업으로 표시
    if (alert.severity === '심각') {
      Alert.alert(
        '심각한 알림',
        `${alert.title}\n\n${alert.description}`,
        [
          {
            text: '확인',
            style: 'destructive'
          },
          {
            text: '상세보기',
            onPress: () => navigateToAlertDetail(alert.id)
          }
        ]
      );
    }
  };

  const updateSwitchStatus = (switchData: SwitchStatus) => {
    setSwitches(prev => 
      prev.map(sw => sw.id === switchData.id ? switchData : sw)
    );
  };

  const navigateToAlertDetail = (alertId: string) => {
    // 알림 상세 화면으로 이동
    console.log('알림 상세 화면 이동:', alertId);
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadInitialData();
    setRefreshing(false);
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      const token = await AsyncStorage.getItem('auth_token');
      await fetch(`https://api.a2z-tsn.com/v2/monitoring/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      setAlerts(prev => prev.filter(alert => alert.id !== alertId));
    } catch (error) {
      console.error('알림 확인 실패:', error);
      Alert.alert('오류', '알림을 확인할 수 없습니다.');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case '정상': return '#4CAF50';
      case '경고': return '#FF9800';
      case '오류': return '#F44336';
      case '오프라인': return '#9E9E9E';
      default: return '#2196F3';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case '정보': return '#2196F3';
      case '경고': return '#FF9800';
      case '오류': return '#F44336';
      case '심각': return '#D32F2F';
      default: return '#757575';
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}일 ${hours}시간 ${minutes}분`;
  };

  const chartConfig = {
    backgroundColor: '#ffffff',
    backgroundGradientFrom: '#ffffff',
    backgroundGradientTo: '#ffffff',
    color: (opacity = 1) => `rgba(33, 150, 243, ${opacity})`,
    labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
    style: {
      borderRadius: 16
    },
    propsForDots: {
      r: '6',
      strokeWidth: '2',
      stroke: '#2196F3'
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar backgroundColor="#1976D2" barStyle="light-content" />
      
      <Header
        centerComponent={{
          text: 'A2Z TSN 모니터링',
          style: { color: '#fff', fontSize: 20, fontWeight: 'bold' }
        }}
        rightComponent={{
          icon: isConnected ? 'wifi' : 'wifi-off',
          color: isConnected ? '#4CAF50' : '#F44336'
        }}
        backgroundColor="#1976D2"
      />

      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* 주요 메트릭 카드 */}
        <Card containerStyle={styles.metricsCard}>
          <Text style={styles.cardTitle}>실시간 네트워크 상태</Text>
          
          <View style={styles.metricsRow}>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>
                {metrics?.bandwidth_utilization.toFixed(1) || '0.0'}%
              </Text>
              <Text style={styles.metricLabel}>대역폭 사용률</Text>
            </View>
            
            <View style={styles.metricItem}>
              <Text style={[styles.metricValue, { 
                color: (metrics?.latency || 0) > 5 ? '#F44336' : '#4CAF50' 
              }]}>
                {metrics?.latency.toFixed(2) || '0.00'}ms
              </Text>
              <Text style={styles.metricLabel}>평균 지연시간</Text>
            </View>
          </View>
          
          <View style={styles.metricsRow}>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>
                {metrics?.availability.toFixed(3) || '99.999'}%
              </Text>
              <Text style={styles.metricLabel}>가용성</Text>
            </View>
            
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>
                {metrics?.frer_recoveries || 0}
              </Text>
              <Text style={styles.metricLabel}>FRER 복구</Text>
            </View>
          </View>
          
          <LinearProgress
            style={styles.progressBar}
            value={(metrics?.bandwidth_utilization || 0) / 100}
            color={"#2196F3"}
            trackColor={"#E3F2FD"}
            variant="determinate"
          />
        </Card>

        {/* 대역폭 사용률 차트 */}
        {historicalData.length > 0 && (
          <Card containerStyle={styles.chartCard}>
            <Text style={styles.cardTitle}>대역폭 사용률 추이</Text>
            <LineChart
              data={{
                labels: ['10분전', '8분전', '6분전', '4분전', '2분전', '현재'],
                datasets: [{
                  data: historicalData.slice(-6)
                }]
              }}
              width={screenWidth - 60}
              height={220}
              chartConfig={chartConfig}
              bezier
              style={styles.chart}
            />
          </Card>
        )}

        {/* 스위치 상태 */}
        <Card containerStyle={styles.switchCard}>
          <Text style={styles.cardTitle}>TSN 스위치 상태</Text>
          
          {switches.map((sw, index) => (
            <ListItem key={sw.id} bottomDivider>
              <Avatar
                rounded
                size="small"
                source={require('../assets/switch-icon.png')}
              />
              <ListItem.Content>
                <ListItem.Title style={styles.switchName}>
                  {sw.name}
                </ListItem.Title>
                <ListItem.Subtitle style={styles.switchLocation}>
                  {sw.location} • {formatUptime(sw.uptime)}
                </ListItem.Subtitle>
                <View style={styles.switchMetrics}>
                  <Text style={styles.switchMetricText}>
                    온도: {sw.temperature}°C
                  </Text>
                  <Text style={styles.switchMetricText}>
                    포트 사용률: {sw.port_utilization}%
                  </Text>
                </View>
              </ListItem.Content>
              <Badge
                value={sw.status}
                badgeStyle={{
                  backgroundColor: getStatusColor(sw.status)
                }}
                textStyle={{ fontSize: 12 }}
              />
            </ListItem>
          ))}
        </Card>

        {/* 활성 알림 */}
        <Card containerStyle={styles.alertCard}>
          <View style={styles.alertHeader}>
            <Text style={styles.cardTitle}>활성 알림</Text>
            <Badge
              value={alerts.length}
              badgeStyle={{ backgroundColor: '#F44336' }}
            />
          </View>
          
          {alerts.length === 0 ? (
            <View style={styles.noAlertsContainer}>
              <Icon name="check-circle" size={48} color="#4CAF50" />
              <Text style={styles.noAlertsText}>활성 알림이 없습니다</Text>
            </View>
          ) : (
            alerts.map((alert, index) => (
              <ListItem key={alert.id} bottomDivider>
                <Icon
                  name={alert.severity === '심각' ? 'error' : 'warning'}
                  size={24}
                  color={getSeverityColor(alert.severity)}
                />
                <ListItem.Content>
                  <ListItem.Title style={styles.alertTitle}>
                    {alert.title}
                  </ListItem.Title>
                  <ListItem.Subtitle style={styles.alertDescription}>
                    {alert.description}
                  </ListItem.Subtitle>
                  <Text style={styles.alertTime}>
                    {moment(alert.timestamp).fromNow()} • {alert.switch_id}
                  </Text>
                </ListItem.Content>
                <Button
                  title="확인"
                  buttonStyle={styles.acknowledgeButton}
                  titleStyle={styles.acknowledgeButtonText}
                  onPress={() => acknowledgeAlert(alert.id)}
                />
              </ListItem>
            ))
          )}
        </Card>

        {/* 시스템 정보 */}
        <Card containerStyle={styles.systemCard}>
          <Text style={styles.cardTitle}>시스템 정보</Text>
          
          <View style={styles.systemInfo}>
            <View style={styles.systemInfoRow}>
              <Text style={styles.systemInfoLabel}>마지막 업데이트:</Text>
              <Text style={styles.systemInfoValue}>
                {moment().format('YYYY-MM-DD HH:mm:ss')}
              </Text>
            </View>
            
            <View style={styles.systemInfoRow}>
              <Text style={styles.systemInfoLabel}>연결 상태:</Text>
              <Text style={[
                styles.systemInfoValue,
                { color: isConnected ? '#4CAF50' : '#F44336' }
              ]}>
                {isConnected ? '연결됨' : '연결 해제됨'}
              </Text>
            </View>
            
            <View style={styles.systemInfoRow}>
              <Text style={styles.systemInfoLabel}>앱 버전:</Text>
              <Text style={styles.systemInfoValue}>2.0.0</Text>
            </View>
          </View>
        </Card>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5'
  },
  scrollView: {
    flex: 1
  },
  metricsCard: {
    margin: 10,
    borderRadius: 12,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1976D2',
    marginBottom: 15,
    textAlign: 'center'
  },
  metricsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15
  },
  metricItem: {
    alignItems: 'center',
    flex: 1
  },
  metricValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2196F3'
  },
  metricLabel: {
    fontSize: 12,
    color: '#757575',
    marginTop: 5
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
    marginTop: 10
  },
  chartCard: {
    margin: 10,
    borderRadius: 12,
    elevation: 4
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16
  },
  switchCard: {
    margin: 10,
    borderRadius: 12,
    elevation: 4
  },
  switchName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333'
  },
  switchLocation: {
    fontSize: 14,
    color: '#757575'
  },
  switchMetrics: {
    flexDirection: 'row',
    marginTop: 5
  },
  switchMetricText: {
    fontSize: 12,
    color: '#999',
    marginRight: 15
  },
  alertCard: {
    margin: 10,
    borderRadius: 12,
    elevation: 4
  },
  alertHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15
  },
  noAlertsContainer: {
    alignItems: 'center',
    paddingVertical: 30
  },
  noAlertsText: {
    fontSize: 16,
    color: '#4CAF50',
    marginTop: 10,
    fontWeight: '500'
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333'
  },
  alertDescription: {
    fontSize: 14,
    color: '#666',
    marginTop: 3
  },
  alertTime: {
    fontSize: 12,
    color: '#999',
    marginTop: 5
  },
  acknowledgeButton: {
    backgroundColor: '#4CAF50',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 8
  },
  acknowledgeButtonText: {
    fontSize: 12,
    fontWeight: 'bold'
  },
  systemCard: {
    margin: 10,
    borderRadius: 12,
    elevation: 4,
    marginBottom: 20
  },
  systemInfo: {
    paddingVertical: 10
  },
  systemInfoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8
  },
  systemInfoLabel: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500'
  },
  systemInfoValue: {
    fontSize: 14,
    color: '#333',
    fontWeight: 'bold'
  }
});

export default DashboardScreen;