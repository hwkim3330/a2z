#!/usr/bin/env python3
"""
A2Z Digital Twin for TSN/FRER Network
디지털 트윈 기술로 네트워크 시뮬레이션 및 예측
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import deque, defaultdict
import logging
import uuid
from datetime import datetime, timedelta
import sqlite3
import redis
from concurrent.futures import ThreadPoolExecutor

# 시뮬레이션 및 모델링
import simpy
import networkx as nx
from scipy import stats, optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 3D 시각화
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 실시간 데이터 처리
import websockets
from kafka import KafkaConsumer, KafkaProducer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

class ComponentType(Enum):
    """네트워크 컴포넌트 타입"""
    TSN_SWITCH = "tsn_switch"
    ENDPOINT = "endpoint"
    LINK = "link"
    FRER_STREAM = "frer_stream"
    SENSOR = "sensor"
    ECU = "ecu"
    GATEWAY = "gateway"

class ComponentState(Enum):
    """컴포넌트 상태"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

@dataclass
class ComponentTwin:
    """네트워크 컴포넌트의 디지털 트윈"""
    id: str
    name: str
    type: ComponentType
    state: ComponentState = ComponentState.ACTIVE
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    health_score: float = 100.0
    predicted_lifetime: float = 0.0  # hours
    anomaly_score: float = 0.0
    maintenance_due: Optional[datetime] = None

@dataclass
class NetworkEvent:
    """네트워크 이벤트"""
    id: str
    timestamp: datetime
    component_id: str
    event_type: str
    severity: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class SimulationScenario:
    """시뮬레이션 시나리오"""
    id: str
    name: str
    description: str
    duration: int  # seconds
    events: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)

class DigitalTwinEngine:
    """디지털 트윈 엔진"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.components: Dict[str, ComponentTwin] = {}
        self.network_graph = nx.DiGraph()
        self.events: List[NetworkEvent] = []
        self.simulation_env = None
        self.logger = self._setup_logger()
        
        # 데이터베이스 연결
        self.db_connection = self._setup_database()
        self.redis_client = self._setup_redis()
        self.influx_client = self._setup_influxdb()
        
        # ML 모델
        self.ml_models = {
            'performance_predictor': RandomForestRegressor(n_estimators=100),
            'anomaly_detector': None,  # 사용자 정의 모델
            'failure_predictor': None
        }
        self.scaler = StandardScaler()
        
        # 실시간 데이터 수집
        self.data_collectors = {}
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        
        # 시뮬레이션 상태
        self.simulation_running = False
        self.simulation_time = 0
        self.time_acceleration = 1.0
        
        # 3D 시각화
        self.visualization_server = None
        self.vis_data = {
            'nodes': [],
            'edges': [],
            'metrics': {},
            'alerts': []
        }
        
        self.logger.info("디지털 트윈 엔진 초기화 완료")
    
    def _setup_logger(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('DigitalTwin')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_database(self) -> sqlite3.Connection:
        """데이터베이스 연결"""
        try:
            conn = sqlite3.connect('digital_twin.db', check_same_thread=False)
            
            # 테이블 생성
            cursor = conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS components (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    type TEXT,
                    state TEXT,
                    properties TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    component_id TEXT,
                    event_type TEXT,
                    severity TEXT,
                    description TEXT,
                    data TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                );
                
                CREATE TABLE IF NOT EXISTS simulations (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    scenario TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    results TEXT
                );
            """)
            conn.commit()
            
            return conn
            
        except Exception as e:
            self.logger.error(f"데이터베이스 연결 실패: {e}")
            return None
    
    def _setup_redis(self) -> Optional[redis.Redis]:
        """레디스 연결"""
        try:
            client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                decode_responses=True
            )
            client.ping()
            return client
        except Exception as e:
            self.logger.warning(f"레디스 연결 실패: {e}")
            return None
    
    def _setup_influxdb(self) -> Optional[InfluxDBClient]:
        """인플럭스DB 연결"""
        try:
            client = InfluxDBClient(
                url=self.config.get('influx_url', 'http://localhost:8086'),
                token=self.config.get('influx_token', ''),
                org=self.config.get('influx_org', 'a2z')
            )
            return client
        except Exception as e:
            self.logger.warning(f"인플럭스DB 연결 실패: {e}")
            return None
    
    def add_component(self, component: ComponentTwin) -> bool:
        """컴포넌트 추가"""
        try:
            self.components[component.id] = component
            self.network_graph.add_node(
                component.id,
                type=component.type.value,
                state=component.state.value,
                **component.properties
            )
            
            # 데이터베이스에 저장
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO components 
                    (id, name, type, state, properties) 
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    component.id,
                    component.name,
                    component.type.value,
                    component.state.value,
                    json.dumps(component.properties)
                ))
                self.db_connection.commit()
            
            self.logger.info(f"컴포넌트 '{component.name}' 추가 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"컴포넌트 추가 실패: {e}")
            return False
    
    def add_connection(self, from_component: str, to_component: str, 
                      properties: Dict[str, Any] = None) -> bool:
        """컴포넌트 간 연결 추가"""
        try:
            if from_component not in self.components or to_component not in self.components:
                raise ValueError("존재하지 않는 컴포넌트")
            
            self.network_graph.add_edge(
                from_component, 
                to_component, 
                **(properties or {})
            )
            
            # 양방향 연결 정보 업데이트
            self.components[from_component].connections.append(to_component)
            self.components[to_component].connections.append(from_component)
            
            self.logger.info(f"연결 추가: {from_component} -> {to_component}")
            return True
            
        except Exception as e:
            self.logger.error(f"연결 추가 실패: {e}")
            return False
    
    def update_component_metrics(self, component_id: str, metrics: Dict[str, float]):
        """컴포넌트 메트릭 업데이트"""
        if component_id not in self.components:
            return
        
        component = self.components[component_id]
        component.metrics.update(metrics)
        component.last_updated = datetime.now()
        
        # 메트릭 히스토리 저장
        timestamp = time.time()
        for metric_name, value in metrics.items():
            self.metrics_history[f"{component_id}.{metric_name}"].append({
                'timestamp': timestamp,
                'value': value
            })
        
        # 인플럭스DB에 저장
        if self.influx_client:
            try:
                write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
                points = []
                
                for metric_name, value in metrics.items():
                    point = Point("network_metrics") \
                        .tag("component_id", component_id) \
                        .tag("component_type", component.type.value) \
                        .field(metric_name, value) \
                        .time(datetime.now())
                    points.append(point)
                
                write_api.write(
                    bucket=self.config.get('influx_bucket', 'tsn_metrics'),
                    record=points
                )
                
            except Exception as e:
                self.logger.warning(f"인플럭스DB 저장 실패: {e}")
        
        # 이상 탐지 및 건강 점수 계산
        self._update_component_health(component_id)
    
    def _update_component_health(self, component_id: str):
        """컴포넌트 건강 상태 업데이트"""
        component = self.components[component_id]
        
        try:
            # 기본 건강 점수 계산
            health_factors = []
            
            # 온도 기반 평가
            if 'temperature' in component.metrics:
                temp = component.metrics['temperature']
                if temp < 70:  # 정상
                    health_factors.append(100)
                elif temp < 80:  # 경고
                    health_factors.append(80)
                elif temp < 90:  # 주의
                    health_factors.append(60)
                else:  # 위험
                    health_factors.append(30)
            
            # CPU 사용률 기반 평가
            if 'cpu_usage' in component.metrics:
                cpu = component.metrics['cpu_usage']
                if cpu < 70:
                    health_factors.append(100)
                elif cpu < 85:
                    health_factors.append(80)
                else:
                    health_factors.append(50)
            
            # 메모리 사용률 기반 평가
            if 'memory_usage' in component.metrics:
                mem = component.metrics['memory_usage']
                if mem < 80:
                    health_factors.append(100)
                elif mem < 90:
                    health_factors.append(75)
                else:
                    health_factors.append(40)
            
            # 업타임 기반 평가
            if 'uptime' in component.metrics:
                uptime_days = component.metrics['uptime'] / 86400
                if uptime_days > 30:  # 30일 이상
                    health_factors.append(100)
                elif uptime_days > 7:
                    health_factors.append(90)
                else:
                    health_factors.append(80)
            
            # 전체 건강 점수
            if health_factors:
                component.health_score = np.mean(health_factors)
            
            # 상태 업데이트
            if component.health_score >= 90:
                component.state = ComponentState.ACTIVE
            elif component.health_score >= 70:
                component.state = ComponentState.DEGRADED
            else:
                component.state = ComponentState.FAILED
            
            # ML 기반 이상 탐지
            anomaly_score = self._detect_anomaly(component_id)
            component.anomaly_score = anomaly_score
            
            if anomaly_score > 0.8:  # 이상 탐지
                self._create_event(
                    component_id=component_id,
                    event_type='anomaly_detected',
                    severity='warning',
                    description=f'이상 탐지: 점수 {anomaly_score:.2f}'
                )
            
        except Exception as e:
            self.logger.error(f"건강 상태 업데이트 실패: {e}")
    
    def _detect_anomaly(self, component_id: str) -> float:
        """이상 탐지 (간단한 방법)"""
        try:
            component = self.components[component_id]
            
            # 최근 10개 데이터 포인트 가져오기
            recent_metrics = []
            for metric_name in component.metrics.keys():
                key = f"{component_id}.{metric_name}"
                if key in self.metrics_history:
                    history = list(self.metrics_history[key])[-10:]
                    values = [point['value'] for point in history]
                    if len(values) >= 5:
                        recent_metrics.extend(values)
            
            if len(recent_metrics) < 10:
                return 0.0
            
            # 통계적 이상 탐지 (Z-score)
            mean_val = np.mean(recent_metrics)
            std_val = np.std(recent_metrics)
            
            if std_val == 0:
                return 0.0
            
            current_values = list(component.metrics.values())
            if not current_values:
                return 0.0
            
            z_scores = [(val - mean_val) / std_val for val in current_values]
            max_z_score = max(abs(z) for z in z_scores)
            
            # Z-score를 0-1 범위로 변환
            anomaly_score = min(max_z_score / 3.0, 1.0)
            
            return anomaly_score
            
        except Exception as e:
            self.logger.error(f"이상 탐지 실패: {e}")
            return 0.0
    
    def _create_event(self, component_id: str, event_type: str, 
                     severity: str, description: str, data: Dict[str, Any] = None):
        """이벤트 생성"""
        event = NetworkEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            component_id=component_id,
            event_type=event_type,
            severity=severity,
            description=description,
            data=data or {}
        )
        
        self.events.append(event)
        
        # 데이터베이스에 저장
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO events 
                (id, timestamp, component_id, event_type, severity, description, data) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.timestamp,
                event.component_id,
                event.event_type,
                event.severity,
                event.description,
                json.dumps(event.data)
            ))
            self.db_connection.commit()
        
        self.logger.info(f"이벤트 생성: {event_type} - {description}")
    
    def run_simulation(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """시뮬레이션 실행"""
        try:
            self.logger.info(f"시뮬레이션 시작: {scenario.name}")
            
            # SimPy 환경 설정
            self.simulation_env = simpy.Environment()
            self.simulation_running = True
            
            # 시뮬레이션 결과 저장
            results = {
                'scenario_id': scenario.id,
                'start_time': datetime.now(),
                'events_triggered': [],
                'performance_metrics': {},
                'network_state_changes': []
            }
            
            # 시나리오 이벤트 스케줄링
            for event_config in scenario.events:
                self.simulation_env.process(
                    self._simulate_event(event_config, results)
                )
            
            # 성능 모니터링 프로세스
            self.simulation_env.process(
                self._monitor_simulation_performance(results)
            )
            
            # 시뮬레이션 실행
            self.simulation_env.run(until=scenario.duration)
            
            results['end_time'] = datetime.now()
            results['duration'] = scenario.duration
            
            self.simulation_running = False
            
            # 결과 저장
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    INSERT INTO simulations 
                    (id, name, scenario, start_time, end_time, results) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    scenario.name,
                    json.dumps(asdict(scenario)),
                    results['start_time'],
                    results['end_time'],
                    json.dumps(results, default=str)
                ))
                self.db_connection.commit()
            
            self.logger.info(f"시뮬레이션 완료: {scenario.name}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"시뮬레이션 실패: {e}")
            self.simulation_running = False
            return {}
    
    def _simulate_event(self, event_config: Dict[str, Any], results: Dict[str, Any]):
        """시뮬레이션 이벤트 처리"""
        # 이벤트 발생 시간까지 대기
        yield self.simulation_env.timeout(event_config.get('delay', 0))
        
        event_type = event_config['type']
        component_id = event_config.get('component_id')
        
        if event_type == 'component_failure':
            yield self.simulation_env.process(
                self._simulate_component_failure(component_id, event_config, results)
            )
        
        elif event_type == 'traffic_surge':
            yield self.simulation_env.process(
                self._simulate_traffic_surge(event_config, results)
            )
        
        elif event_type == 'frer_recovery':
            yield self.simulation_env.process(
                self._simulate_frer_recovery(component_id, event_config, results)
            )
    
    def _simulate_component_failure(self, component_id: str, config: Dict[str, Any], 
                                   results: Dict[str, Any]):
        """컴포넌트 장애 시뮬레이션"""
        if component_id not in self.components:
            return
        
        component = self.components[component_id]
        
        # 장애 발생
        original_state = component.state
        component.state = ComponentState.FAILED
        component.health_score = 0
        
        # 이벤트 기록
        failure_event = {
            'type': 'component_failure',
            'component_id': component_id,
            'time': self.simulation_env.now,
            'original_state': original_state.value
        }
        results['events_triggered'].append(failure_event)
        
        self._create_event(
            component_id=component_id,
            event_type='simulation_failure',
            severity='critical',
            description=f'시뮬레이션에서 {component.name} 장애 발생'
        )
        
        # 복구 시간 시뮬레이션
        recovery_time = config.get('recovery_time', 300)  # 5분 기본
        yield self.simulation_env.timeout(recovery_time)
        
        # 복구
        component.state = original_state
        component.health_score = 80  # 복구 후 약간 저하
        
        recovery_event = {
            'type': 'component_recovery',
            'component_id': component_id,
            'time': self.simulation_env.now,
            'recovery_time': recovery_time
        }
        results['events_triggered'].append(recovery_event)
    
    def _simulate_traffic_surge(self, config: Dict[str, Any], results: Dict[str, Any]):
        """트래픽 급증 시뮬레이션"""
        surge_duration = config.get('duration', 60)  # 1분 기본
        surge_multiplier = config.get('multiplier', 2.0)
        
        # 모든 스위치의 대역폭 사용률 증가
        affected_components = []
        for comp_id, component in self.components.items():
            if component.type == ComponentType.TSN_SWITCH:
                if 'bandwidth_utilization' in component.metrics:
                    original_util = component.metrics['bandwidth_utilization']
                    new_util = min(original_util * surge_multiplier, 100)
                    component.metrics['bandwidth_utilization'] = new_util
                    affected_components.append({
                        'id': comp_id,
                        'original': original_util,
                        'new': new_util
                    })
        
        surge_event = {
            'type': 'traffic_surge_start',
            'time': self.simulation_env.now,
            'affected_components': affected_components,
            'multiplier': surge_multiplier
        }
        results['events_triggered'].append(surge_event)
        
        # 급증 지속 시간
        yield self.simulation_env.timeout(surge_duration)
        
        # 원래 상태로 복구
        for comp_info in affected_components:
            component = self.components[comp_info['id']]
            component.metrics['bandwidth_utilization'] = comp_info['original']
        
        recovery_event = {
            'type': 'traffic_surge_end',
            'time': self.simulation_env.now,
            'duration': surge_duration
        }
        results['events_triggered'].append(recovery_event)
    
    def _simulate_frer_recovery(self, stream_id: str, config: Dict[str, Any], 
                               results: Dict[str, Any]):
        """
FRER 복구 시뮬레이션"""
        recovery_time = config.get('recovery_time', 50)  # 50ms 기본
        
        # FRER 복구 시나리오 시뮬레이션
        frer_event = {
            'type': 'frer_recovery',
            'stream_id': stream_id,
            'time': self.simulation_env.now,
            'recovery_time_ms': recovery_time
        }
        results['events_triggered'].append(frer_event)
        
        # 복구 시간 (밀리초 단위를 초로 변환)
        yield self.simulation_env.timeout(recovery_time / 1000.0)
    
    def _monitor_simulation_performance(self, results: Dict[str, Any]):
        """시뮬레이션 성능 모니터링"""
        while self.simulation_running:
            current_time = self.simulation_env.now
            
            # 성능 메트릭 수집
            performance_snapshot = {
                'time': current_time,
                'component_states': {},
                'network_health': 0,
                'active_components': 0,
                'failed_components': 0
            }
            
            total_health = 0
            active_count = 0
            failed_count = 0
            
            for comp_id, component in self.components.items():
                performance_snapshot['component_states'][comp_id] = {
                    'state': component.state.value,
                    'health_score': component.health_score
                }
                
                total_health += component.health_score
                
                if component.state == ComponentState.ACTIVE:
                    active_count += 1
                elif component.state == ComponentState.FAILED:
                    failed_count += 1
            
            if len(self.components) > 0:
                performance_snapshot['network_health'] = total_health / len(self.components)
            
            performance_snapshot['active_components'] = active_count
            performance_snapshot['failed_components'] = failed_count
            
            results['network_state_changes'].append(performance_snapshot)
            
            # 1초마다 수집
            yield self.simulation_env.timeout(1)
    
    def predict_network_performance(self, time_horizon: int = 3600) -> Dict[str, Any]:
        """네트워크 성능 예측 (시간 단위: 초)"""
        try:
            predictions = {}
            
            for comp_id, component in self.components.items():
                # 지난 데이터 수집
                historical_data = []
                for metric_name in component.metrics.keys():
                    key = f"{comp_id}.{metric_name}"
                    if key in self.metrics_history:
                        history = list(self.metrics_history[key])
                        if len(history) >= 10:
                            historical_data.extend([point['value'] for point in history])
                
                if len(historical_data) >= 20:
                    # 시계열 예측 (간단한 선형 추세)
                    x = np.arange(len(historical_data))
                    y = np.array(historical_data)
                    
                    # 선형 회귀
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # 미래 예측
                    future_steps = time_horizon // 60  # 분 단위로 예측
                    future_x = np.arange(len(historical_data), len(historical_data) + future_steps)
                    future_y = slope * future_x + intercept
                    
                    predictions[comp_id] = {
                        'predicted_values': future_y.tolist(),
                        'confidence': r_value ** 2,
                        'trend': 'increasing' if slope > 0 else 'decreasing',
                        'slope': slope
                    }
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"성능 예측 실패: {e}")
            return {}
    
    def generate_3d_visualization(self) -> go.Figure:
        """네트워크 토폴로지 3D 시각화"""
        try:
            # 노드 데이터 준비
            node_trace = go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=[],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Health Score")
                ),
                text=[],
                textposition="middle center",
                name="Components"
            )
            
            # 엣지 데이터 준비
            edge_traces = []
            
            # 컴포넌트 노드 추가
            for comp_id, component in self.components.items():
                x, y, z = component.position
                
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
                node_trace['z'] += (z,)
                node_trace['marker']['color'] += (component.health_score,)
                node_trace['text'] += (f"{component.name}\n{component.state.value}",)
            
            # 연결 라인 추가
            for edge in self.network_graph.edges():
                comp1 = self.components[edge[0]]
                comp2 = self.components[edge[1]]
                
                edge_trace = go.Scatter3d(
                    x=[comp1.position[0], comp2.position[0], None],
                    y=[comp1.position[1], comp2.position[1], None],
                    z=[comp1.position[2], comp2.position[2], None],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    hoverinfo='none',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
            
            # 그래프 생성
            fig = go.Figure(data=[node_trace] + edge_traces)
            
            fig.update_layout(
                title='A2Z TSN Network Digital Twin - 3D Topology',
                scene=dict(
                    xaxis_title='X Position',
                    yaxis_title='Y Position',
                    zaxis_title='Z Position',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                showlegend=True,
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"3D 시각화 생성 실패: {e}")
            return go.Figure()
    
    def export_twin_data(self, format: str = 'json') -> str:
        """디지털 트윈 데이터 내보내기"""
        try:
            twin_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'total_components': len(self.components),
                    'total_events': len(self.events)
                },
                'components': {
                    comp_id: asdict(component) 
                    for comp_id, component in self.components.items()
                },
                'network_graph': {
                    'nodes': list(self.network_graph.nodes(data=True)),
                    'edges': list(self.network_graph.edges(data=True))
                },
                'recent_events': [
                    asdict(event) for event in self.events[-100:]
                ]
            }
            
            if format.lower() == 'json':
                return json.dumps(twin_data, indent=2, default=str, ensure_ascii=False)
            else:
                # 다른 형식 지원 가능
                return str(twin_data)
                
        except Exception as e:
            self.logger.error(f"데이터 내보내기 실패: {e}")
            return ""
    
    def get_system_summary(self) -> Dict[str, Any]:
        """시스템 요약 정보"""
        try:
            active_components = sum(1 for c in self.components.values() 
                                  if c.state == ComponentState.ACTIVE)
            failed_components = sum(1 for c in self.components.values() 
                                  if c.state == ComponentState.FAILED)
            
            avg_health = np.mean([c.health_score for c in self.components.values()]) \
                        if self.components else 0
            
            recent_events = len([e for e in self.events 
                               if (datetime.now() - e.timestamp).seconds < 3600])
            
            return {
                'total_components': len(self.components),
                'active_components': active_components,
                'failed_components': failed_components,
                'degraded_components': len(self.components) - active_components - failed_components,
                'average_health_score': avg_health,
                'network_connections': self.network_graph.number_of_edges(),
                'recent_events_1h': recent_events,
                'total_events': len(self.events),
                'last_updated': max(c.last_updated for c in self.components.values()) \
                              if self.components else None
            }
            
        except Exception as e:
            self.logger.error(f"요약 정보 생성 실패: {e}")
            return {}

# 사용 예시
def main():
    """메인 실행 함수"""
    print("A2Z 디지털 트윈 시스템 시작...")
    
    # 디지털 트윈 엔진 초기화
    twin_engine = DigitalTwinEngine({
        'influx_url': 'http://localhost:8086',
        'influx_token': 'your-influx-token',
        'influx_org': 'a2z',
        'influx_bucket': 'tsn_metrics'
    })
    
    # TSN 스위치 추가
    switch1 = ComponentTwin(
        id="LAN9692-001",
        name="전방 TSN 스위치",
        type=ComponentType.TSN_SWITCH,
        position=(0, 0, 0),
        properties={
            'model': 'Microchip LAN9692',
            'ports': 30,
            'location': '서울특별시 강남구 테헤란로 427'
        }
    )
    
    switch2 = ComponentTwin(
        id="LAN9692-002",
        name="중앙 TSN 스위치",
        type=ComponentType.TSN_SWITCH,
        position=(5, 0, 0),
        properties={
            'model': 'Microchip LAN9692',
            'ports': 30,
            'location': '서울특별시 강남구 역삼로 123'
        }
    )
    
    # 컴포넌트 추가
    twin_engine.add_component(switch1)
    twin_engine.add_component(switch2)
    
    # 연결 추가
    twin_engine.add_connection(
        "LAN9692-001", "LAN9692-002",
        {'bandwidth': 1000, 'latency': 0.1}
    )
    
    # 메트릭 업데이트 예시
    twin_engine.update_component_metrics("LAN9692-001", {
        'temperature': 45.2,
        'cpu_usage': 23.5,
        'memory_usage': 67.8,
        'bandwidth_utilization': 34.2,
        'uptime': 86400 * 15  # 15일
    })
    
    # 시나리오 생성 및 실행
    scenario = SimulationScenario(
        id="test_scenario_001",
        name="스위치 장애 테스트",
        description="전방 스위치 장애 및 복구 시나리오",
        duration=300,  # 5분
        events=[
            {
                'type': 'component_failure',
                'delay': 10,
                'component_id': 'LAN9692-001',
                'recovery_time': 60
            },
            {
                'type': 'traffic_surge',
                'delay': 120,
                'duration': 30,
                'multiplier': 2.5
            }
        ]
    )
    
    # 시뮬레이션 실행
    results = twin_engine.run_simulation(scenario)
    print(f"\n시뮬레이션 결과: {json.dumps(results, indent=2, default=str, ensure_ascii=False)}")
    
    # 성능 예측
    predictions = twin_engine.predict_network_performance(3600)
    print(f"\n성능 예측: {json.dumps(predictions, indent=2, ensure_ascii=False)}")
    
    # 시스템 요약
    summary = twin_engine.get_system_summary()
    print(f"\n시스템 요약: {json.dumps(summary, indent=2, default=str, ensure_ascii=False)}")
    
    # 3D 시각화 생성
    fig = twin_engine.generate_3d_visualization()
    if fig.data:
        fig.show()
        print("\n3D 시각화가 브라우저에 표시됩니다.")
    
    print("\n디지털 트윈 시스템 완료!")

if __name__ == "__main__":
    main()