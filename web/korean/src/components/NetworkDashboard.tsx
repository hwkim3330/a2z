import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Paper,
  Chip,
  Alert,
  AlertTitle,
  Button,
  Switch,
  FormControlLabel,
  LinearProgress,
  Tooltip,
  Badge,
  IconButton,
  Snackbar
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  TimelineOppositeContent
} from '@mui/lab';
import {
  NetworkCheck,
  Router,
  Speed,
  Warning,
  Error,
  CheckCircle,
  Refresh,
  Settings,
  Visibility,
  NotificationsActive,
  Security,
  Memory,
  Storage,
  CloudQueue
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { useWebSocket } from '../hooks/useWebSocket';
import { useApiCall } from '../hooks/useApiCall';
import moment from 'moment';
import 'moment/locale/ko';

// 한국어 설정
moment.locale('ko');

interface NetworkMetrics {
  timestamp: string;
  switches: {
    id: string;
    name: string;
    bandwidth: number;
    latency: number;
    packet_loss: number;
    temperature: number;
    uptime: number;
    status: 'active' | 'warning' | 'error' | 'offline';
  }[];
  frer_streams: {
    id: string;
    priority: number;
    recovery_count: number;
    active_path: 'primary' | 'secondary';
    status: 'normal' | 'recovering' | 'failed';
  }[];
  aggregate: {
    total_bandwidth: number;
    average_latency: number;
    packet_loss_rate: number;
    availability: number;
    active_alerts: number;
  };
}

interface Alert {
  id: string;
  timestamp: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  source: string;
  title: string;
  description: string;
  acknowledged: boolean;
}

interface SwitchLocation {
  id: string;
  name: string;
  zone: 'front' | 'central' | 'rear';
  location: string;
  coordinates: [number, number];
  connections: string[];
}

const NetworkDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<NetworkMetrics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [switchLocations] = useState<SwitchLocation[]>([
    {
      id: 'LAN9692-서울-강남-001',
      name: '전방 TSN 스위치',
      zone: 'front',
      location: '서울특별시 강남구 테헤란로 427',
      coordinates: [37.5665, 126.9780],
      connections: ['LAN9692-서울-강남-002']
    },
    {
      id: 'LAN9692-서울-강남-002',
      name: '중앙 TSN 스위치',
      zone: 'central',
      location: '서울특별시 강남구 역삼로 123',
      coordinates: [37.5010, 127.0374],
      connections: ['LAN9692-서울-강남-001', 'LAN9662-서울-강남-003']
    },
    {
      id: 'LAN9662-서울-강남-003',
      name: '후방 TSN 스위치',
      zone: 'rear',
      location: '서울특별시 강남구 선릉로 456',
      coordinates: [37.5045, 127.0489],
      connections: ['LAN9692-서울-강남-002']
    }
  ]);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  // WebSocket 연결
  const { connected, sendMessage } = useWebSocket('wss://ws.a2z-tsn.com', {
    onMessage: handleWebSocketMessage,
    onConnect: () => {
      console.log('대시보드 WebSocket 연결됨');
      subscribeToUpdates();
    }
  });

  // API 호출 훅
  const { loading, error, execute } = useApiCall();

  useEffect(() => {
    loadInitialData();
    
    // 자동 새로고침
    let interval: NodeJS.Timeout;
    if (autoRefresh) {
      interval = setInterval(refreshData, 30000); // 30초마다
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const loadInitialData = async () => {
    try {
      await Promise.all([
        loadCurrentMetrics(),
        loadActiveAlerts(),
        loadHistoricalData()
      ]);
    } catch (error) {
      console.error('초기 데이터 로드 실패:', error);
    }
  };

  const loadCurrentMetrics = async () => {
    const result = await execute(async () => {
      const response = await fetch('/api/v2/monitoring/metrics', {
        headers: {
          'Accept-Language': 'ko-KR'
        }
      });
      return response.json();
    });
    
    if (result) {
      setMetrics(result);
    }
  };

  const loadActiveAlerts = async () => {
    const result = await execute(async () => {
      const response = await fetch('/api/v2/monitoring/alerts?status=active');
      return response.json();
    });
    
    if (result) {
      setAlerts(result.alerts);
    }
  };

  const loadHistoricalData = async () => {
    const result = await execute(async () => {
      const response = await fetch(`/api/v2/monitoring/metrics/history?range=${selectedTimeRange}`);
      return response.json();
    });
    
    if (result) {
      setHistoricalData(result.data.map((item: any) => ({
        time: moment(item.timestamp).format('HH:mm'),
        대역폭: item.bandwidth_utilization,
        지연시간: item.latency,
        패킷손실: item.packet_loss * 100,
        FRER복구: item.frer_recoveries
      })));
    }
  };

  const handleWebSocketMessage = (data: any) => {
    switch (data.type) {
      case 'metrics':
        setMetrics(data.data);
        break;
      case 'alert':
        handleNewAlert(data.data);
        break;
      case 'frer_event':
        handleFREREvent(data.data);
        break;
    }
  };

  const subscribeToUpdates = () => {
    sendMessage({
      type: 'subscribe',
      streams: ['metrics', 'alerts', 'frer', 'switches']
    });
  };

  const handleNewAlert = (alert: Alert) => {
    setAlerts(prev => [alert, ...prev.slice(0, 9)]); // 최대 10개
    
    // 알림 스낵바 표시
    setSnackbarMessage(`새 알림: ${alert.title}`);
    setSnackbarOpen(true);
  };

  const handleFREREvent = (event: any) => {
    if (event.type === 'recovery') {
      setSnackbarMessage(`FRER 복구 완료: ${event.stream_id}`);
      setSnackbarOpen(true);
    }
  };

  const refreshData = useCallback(async () => {
    await loadCurrentMetrics();
  }, []);

  const acknowledgeAlert = async (alertId: string) => {
    try {
      await fetch(`/api/v2/monitoring/alerts/${alertId}/acknowledge`, {
        method: 'POST'
      });
      
      setAlerts(prev => prev.filter(alert => alert.id !== alertId));
    } catch (error) {
      console.error('알림 확인 실패:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return '#4CAF50';
      case 'warning': return '#FF9800';
      case 'error': return '#F44336';
      case 'offline': return '#9E9E9E';
      default: return '#2196F3';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'info': return <CheckCircle style={{ color: '#2196F3' }} />;
      case 'warning': return <Warning style={{ color: '#FF9800' }} />;
      case 'error': return <Error style={{ color: '#F44336' }} />;
      case 'critical': return <Error style={{ color: '#D32F2F' }} />;
      default: return <CheckCircle />;
    }
  };

  const getZoneColor = (zone: string) => {
    switch (zone) {
      case 'front': return '#E3F2FD';
      case 'central': return '#FFF3E0';
      case 'rear': return '#F3E5F5';
      default: return '#F5F5F5';
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}일 ${hours}시간 ${minutes}분`;
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <Box sx={{ padding: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      {/* 헤더 */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold', color: '#1976D2' }}>
          A2Z TSN 네트워크 대시보드
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                color="primary"
              />
            }
            label="자동 새로고침"
          />
          
          <Tooltip title="수동 새로고침">
            <IconButton onClick={refreshData} disabled={loading}>
              <Refresh />
            </IconButton>
          </Tooltip>
          
          <Chip
            icon={connected ? <CheckCircle /> : <Error />}
            label={connected ? '연결됨' : '연결 해제됨'}
            color={connected ? 'success' : 'error'}
            variant="outlined"
          />
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* 주요 메트릭 */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
              <NetworkCheck sx={{ mr: 1 }} />
              주요 네트워크 지표
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={3}>
                <Card sx={{ bgcolor: '#E3F2FD' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Box>
                        <Typography variant="h4" sx={{ color: '#1976D2', fontWeight: 'bold' }}>
                          {metrics?.aggregate.total_bandwidth.toFixed(1) || '0.0'}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          대역폭 사용률
                        </Typography>
                      </Box>
                      <Speed sx={{ fontSize: 40, color: '#1976D2' }} />
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={metrics?.aggregate.total_bandwidth || 0}
                      sx={{ mt: 1 }}
                    />
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card sx={{ bgcolor: '#F3E5F5' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Box>
                        <Typography variant="h4" sx={{ 
                          color: (metrics?.aggregate.average_latency || 0) > 5 ? '#F44336' : '#9C27B0',
                          fontWeight: 'bold'
                        }}>
                          {metrics?.aggregate.average_latency.toFixed(2) || '0.00'}ms
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          평균 지연시간
                        </Typography>
                      </Box>
                      <Router sx={{ fontSize: 40, color: '#9C27B0' }} />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card sx={{ bgcolor: '#E8F5E8' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Box>
                        <Typography variant="h4" sx={{ color: '#4CAF50', fontWeight: 'bold' }}>
                          {metrics?.aggregate.availability.toFixed(3) || '99.999'}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          시스템 가용성
                        </Typography>
                      </Box>
                      <CheckCircle sx={{ fontSize: 40, color: '#4CAF50' }} />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card sx={{ bgcolor: '#FFF3E0' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Box>
                        <Typography variant="h4" sx={{ color: '#FF9800', fontWeight: 'bold' }}>
                          {metrics?.aggregate.active_alerts || 0}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          활성 알림
                        </Typography>
                      </Box>
                      <NotificationsActive sx={{ fontSize: 40, color: '#FF9800' }} />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* 스위치 상태 */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              TSN 스위치 상태
            </Typography>
            
            <Grid container spacing={2}>
              {switchLocations.map((location) => {
                const switchData = metrics?.switches.find(s => s.id === location.id);
                
                return (
                  <Grid item xs={12} md={4} key={location.id}>
                    <Card sx={{ 
                      bgcolor: getZoneColor(location.zone),
                      border: switchData?.status === 'error' ? '2px solid #F44336' : 'none'
                    }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <Box>
                            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                              {location.name}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                              {location.location}
                            </Typography>
                            
                            {switchData && (
                              <Box>
                                <Typography variant="body2">
                                  대역폭: {switchData.bandwidth.toFixed(1)} Mbps
                                </Typography>
                                <Typography variant="body2">
                                  지연시간: {switchData.latency.toFixed(2)} ms
                                </Typography>
                                <Typography variant="body2">
                                  온도: {switchData.temperature}°C
                                </Typography>
                                <Typography variant="body2">
                                  업타임: {formatUptime(switchData.uptime)}
                                </Typography>
                              </Box>
                            )}
                          </Box>
                          
                          <Chip
                            label={switchData?.status === 'active' ? '정상' : 
                                   switchData?.status === 'warning' ? '경고' :
                                   switchData?.status === 'error' ? '오류' : '오프라인'}
                            color={switchData?.status === 'active' ? 'success' :
                                   switchData?.status === 'warning' ? 'warning' : 'error'}
                            size="small"
                          />
                        </Box>
                        
                        {/* 진행률 표시기 */}
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="body2" sx={{ mb: 0.5 }}>
                            포트 사용률: {((switchData?.bandwidth || 0) / 1000 * 100).toFixed(1)}%
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={(switchData?.bandwidth || 0) / 10}
                            color={switchData?.bandwidth && switchData.bandwidth > 800 ? 'error' : 'primary'}
                          />
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          </Paper>
        </Grid>

        {/* FRER 스트림 */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              FRER 스트림 상태
            </Typography>
            
            {metrics?.frer_streams && (
              <Box>
                {metrics.frer_streams.slice(0, 5).map((stream, index) => (
                  <Box key={stream.id} sx={{ mb: 2, p: 2, bgcolor: '#f8f9fa', borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                        {stream.id}
                      </Typography>
                      <Chip
                        label={stream.status === 'normal' ? '정상' :
                               stream.status === 'recovering' ? '복구중' : '실패'}
                        color={stream.status === 'normal' ? 'success' :
                               stream.status === 'recovering' ? 'warning' : 'error'}
                        size="small"
                      />
                    </Box>
                    
                    <Typography variant="body2" color="text.secondary">
                      우선순위: {stream.priority} | 복구 횟수: {stream.recovery_count}
                    </Typography>
                    
                    <Typography variant="body2" color="text.secondary">
                      활성 경로: {stream.active_path === 'primary' ? '주 경로' : '보조 경로'}
                    </Typography>
                  </Box>
                ))}
              </Box>
            )}
          </Paper>
        </Grid>

        {/* 성능 차트 */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                성능 추이
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 1 }}>
                {(['1h', '6h', '24h', '7d'] as const).map((range) => (
                  <Button
                    key={range}
                    variant={selectedTimeRange === range ? 'contained' : 'outlined'}
                    size="small"
                    onClick={() => setSelectedTimeRange(range)}
                  >
                    {range}
                  </Button>
                ))}
              </Box>
            </Box>
            
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <RechartsTooltip 
                  formatter={(value, name) => [
                    `${Number(value).toFixed(2)}${name === '대역폭' ? '%' : 
                                                   name === '지연시간' ? 'ms' : 
                                                   name === '패킷손실' ? '%' : ''}`
                  ]}
                />
                <Legend />
                <Line type="monotone" dataKey="대역펭" stroke="#2196F3" strokeWidth={2} />
                <Line type="monotone" dataKey="지연시간" stroke="#FF9800" strokeWidth={2} />
                <Line type="monotone" dataKey="FRER복구" stroke="#4CAF50" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* 알림 타임라인 */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, height: 400, overflow: 'auto' }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              최근 알림
            </Typography>
            
            <Timeline>
              {alerts.slice(0, 10).map((alert, index) => (
                <TimelineItem key={alert.id}>
                  <TimelineOppositeContent color="text.secondary">
                    {moment(alert.timestamp).format('HH:mm')}
                  </TimelineOppositeContent>
                  <TimelineSeparator>
                    <TimelineDot>
                      {getSeverityIcon(alert.severity)}
                    </TimelineDot>
                    {index < alerts.length - 1 && <TimelineConnector />}
                  </TimelineSeparator>
                  <TimelineContent>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {alert.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {alert.description}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {alert.source}
                    </Typography>
                    {!alert.acknowledged && (
                      <Button
                        size="small"
                        onClick={() => acknowledgeAlert(alert.id)}
                        sx={{ mt: 1 }}
                      >
                        확인
                      </Button>
                    )}
                  </TimelineContent>
                </TimelineItem>
              ))}
            </Timeline>
          </Paper>
        </Grid>
      </Grid>

      {/* 알림 스낵바 */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={() => setSnackbarOpen(false)}
        message={snackbarMessage}
        action={
          <Button color="inherit" size="small" onClick={() => setSnackbarOpen(false)}>
            닫기
          </Button>
        }
      />
    </Box>
  );
};

export default NetworkDashboard;