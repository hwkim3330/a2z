#!/usr/bin/env python3
"""
A2Z 한국 정부 API 통합 서비스
Korean Government API Integration Services

주요 연동:
- 국토교통부 (MOLIT) - C-ITS 플랫폼
- 도로교통공단 (KOROAD) - 교통안전 정보
- 기상청 (KMA) - 날씨 및 도로상황
- 경찰청 (KNPA) - 교통단속 정보
- 한국표준과학연구원 (KRISS) - 시간동기화
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import ssl
import certifi
from dataclasses import dataclass
from enum import Enum

class APIServiceType(Enum):
    MOLIT_CITS = "molit_cits"
    KOROAD_TRAFFIC = "koroad_traffic"
    KMA_WEATHER = "kma_weather"
    KNPA_ENFORCEMENT = "knpa_enforcement"
    KRISS_TIME = "kriss_time"

@dataclass
class TrafficIncident:
    incident_id: str
    location: str
    coordinates: tuple
    incident_type: str
    severity: int  # 1-5 (1=정보, 5=심각)
    description: str
    start_time: datetime
    estimated_duration: Optional[int]  # 분 단위
    affected_lanes: List[str]

@dataclass
class WeatherCondition:
    location: str
    temperature: float
    humidity: float
    precipitation: float
    visibility: float  # km
    road_condition: str  # "건조", "습함", "결빙", "적설"
    wind_speed: float
    timestamp: datetime

class KoreanGovernmentAPIClient:
    """한국 정부기관 API 통합 클라이언트"""
    
    def __init__(self, config_path: str = "config/korea/api-credentials.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.session: Optional[aiohttp.ClientSession] = None
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # API 엔드포인트 설정
        self.endpoints = {
            APIServiceType.MOLIT_CITS: "https://cits.molit.go.kr/api/v2",
            APIServiceType.KOROAD_TRAFFIC: "https://api.koroad.or.kr/v3",
            APIServiceType.KMA_WEATHER: "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0",
            APIServiceType.KNPA_ENFORCEMENT: "https://www.police.go.kr/openapi/traffic",
            APIServiceType.KRISS_TIME: "https://time.kriss.re.kr/api"
        }
        
        # 요청 제한 (분당)
        self.rate_limits = {
            APIServiceType.MOLIT_CITS: 100,
            APIServiceType.KOROAD_TRAFFIC: 1000,
            APIServiceType.KMA_WEATHER: 1000,
            APIServiceType.KNPA_ENFORCEMENT: 100,
            APIServiceType.KRISS_TIME: 60
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """API 인증 정보 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                import yaml
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"설정 파일 없음: {config_path}")
            return {}
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        connector = aiohttp.TCPConnector(
            ssl=self.ssl_context,
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'A2Z-TSN-Platform/1.0 (Korean Government Integration)',
                'Accept': 'application/json',
                'Accept-Language': 'ko-KR,ko;q=0.9'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def get_cits_road_conditions(self, route_id: str) -> List[TrafficIncident]:
        """국토교통부 C-ITS에서 도로 상황 정보 조회"""
        endpoint = f"{self.endpoints[APIServiceType.MOLIT_CITS]}/road-conditions"
        
        params = {
            'route_id': route_id,
            'api_key': self.config.get('molit', {}).get('api_key', ''),
            'format': 'json'
        }
        
        try:
            async with self.session.get(endpoint, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    incidents = []
                    
                    for item in data.get('incidents', []):
                        incident = TrafficIncident(
                            incident_id=item['id'],
                            location=item['location'],
                            coordinates=(item['lat'], item['lng']),
                            incident_type=item['type'],
                            severity=item['severity'],
                            description=item['description'],
                            start_time=datetime.fromisoformat(item['start_time']),
                            estimated_duration=item.get('duration_minutes'),
                            affected_lanes=item.get('affected_lanes', [])
                        )
                        incidents.append(incident)
                    
                    self.logger.info(f"C-ITS 도로상황 조회 성공: {len(incidents)}건")
                    return incidents
                else:
                    self.logger.error(f"C-ITS API 오류: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"C-ITS API 호출 실패: {e}")
            return []
    
    async def get_koroad_safety_info(self, area_code: str) -> Dict[str, Any]:
        """도로교통공단 교통안전 정보 조회"""
        endpoint = f"{self.endpoints[APIServiceType.KOROAD_TRAFFIC]}/safety"
        
        params = {
            'area_code': area_code,
            'api_key': self.config.get('koroad', {}).get('api_key', ''),
            'detail': 'true'
        }
        
        try:
            async with self.session.get(endpoint, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    safety_info = {
                        'accident_prone_zones': data.get('accident_zones', []),
                        'speed_limits': data.get('speed_limits', {}),
                        'traffic_violations': data.get('violations', []),
                        'safety_recommendations': data.get('recommendations', []),
                        'real_time_alerts': data.get('alerts', [])
                    }
                    
                    self.logger.info(f"KOROAD 안전정보 조회 성공: {area_code}")
                    return safety_info
                else:
                    self.logger.error(f"KOROAD API 오류: {response.status}")
                    return {}
        except Exception as e:
            self.logger.error(f"KOROAD API 호출 실패: {e}")
            return {}
    
    async def get_kma_weather_data(self, nx: int, ny: int) -> WeatherCondition:
        """기상청 날씨 정보 조회 (격자 좌표 기준)"""
        endpoint = f"{self.endpoints[APIServiceType.KMA_WEATHER]}/getUltraSrtNcst"
        
        base_date = datetime.now().strftime('%Y%m%d')
        base_time = datetime.now().strftime('%H00')
        
        params = {
            'serviceKey': self.config.get('kma', {}).get('service_key', ''),
            'numOfRows': '10',
            'pageNo': '1',
            'base_date': base_date,
            'base_time': base_time,
            'nx': nx,
            'ny': ny,
            'dataType': 'JSON'
        }
        
        try:
            async with self.session.get(endpoint, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data['response']['body']['items']['item']
                    
                    weather_data = {}
                    for item in items:
                        category = item['category']
                        value = float(item['obsrValue'])
                        
                        if category == 'T1H':  # 기온
                            weather_data['temperature'] = value
                        elif category == 'REH':  # 습도
                            weather_data['humidity'] = value
                        elif category == 'RN1':  # 강수량
                            weather_data['precipitation'] = value
                        elif category == 'WSD':  # 풍속
                            weather_data['wind_speed'] = value
                    
                    # 도로상태 판단 로직
                    road_condition = self._determine_road_condition(
                        weather_data.get('temperature', 0),
                        weather_data.get('precipitation', 0)
                    )
                    
                    condition = WeatherCondition(
                        location=f"격자({nx},{ny})",
                        temperature=weather_data.get('temperature', 0),
                        humidity=weather_data.get('humidity', 0),
                        precipitation=weather_data.get('precipitation', 0),
                        visibility=10.0,  # 기본값
                        road_condition=road_condition,
                        wind_speed=weather_data.get('wind_speed', 0),
                        timestamp=datetime.now()
                    )
                    
                    self.logger.info(f"기상청 날씨 조회 성공: 온도 {condition.temperature}°C")
                    return condition
                else:
                    self.logger.error(f"기상청 API 오류: {response.status}")
                    return WeatherCondition("오류", 0, 0, 0, 0, "알수없음", 0, datetime.now())
        except Exception as e:
            self.logger.error(f"기상청 API 호출 실패: {e}")
            return WeatherCondition("오류", 0, 0, 0, 0, "알수없음", 0, datetime.now())
    
    def _determine_road_condition(self, temperature: float, precipitation: float) -> str:
        """온도와 강수량을 기반으로 도로 상태 판단"""
        if precipitation > 0:
            if temperature < 0:
                return "결빙위험"
            elif precipitation > 5:
                return "침수위험"
            else:
                return "습함"
        elif temperature < 0:
            return "결빙주의"
        else:
            return "건조"
    
    async def get_kriss_accurate_time(self) -> Dict[str, Any]:
        """한국표준과학연구원 정밀시각 조회"""
        endpoint = f"{self.endpoints[APIServiceType.KRISS_TIME]}/getTime"
        
        try:
            async with self.session.get(endpoint) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    time_info = {
                        'kriss_time': data.get('time'),
                        'utc_offset': data.get('utc_offset', '+09:00'),
                        'accuracy': data.get('accuracy_ms', 1),  # ms 단위
                        'synchronized': True,
                        'source': 'KRISS 원자시계'
                    }
                    
                    self.logger.info("KRISS 정밀시각 동기화 성공")
                    return time_info
                else:
                    self.logger.error(f"KRISS API 오류: {response.status}")
                    return {'synchronized': False, 'source': 'local'}
        except Exception as e:
            self.logger.error(f"KRISS API 호출 실패: {e}")
            return {'synchronized': False, 'source': 'local'}
    
    async def get_integrated_status(self, location: Dict[str, float]) -> Dict[str, Any]:
        """통합 상황 정보 조회"""
        lat, lng = location['lat'], location['lng']
        
        # 격자 좌표 변환 (간단한 근사)
        nx = int((lng + 127) * 100)
        ny = int((lat - 33) * 100)
        
        # 병렬로 모든 API 호출
        tasks = [
            self.get_cits_road_conditions("SEOUL_146"),
            self.get_koroad_safety_info("11"),  # 서울시 코드
            self.get_kma_weather_data(nx, ny),
            self.get_kriss_accurate_time()
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            integrated_status = {
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'traffic_incidents': results[0] if not isinstance(results[0], Exception) else [],
                'safety_info': results[1] if not isinstance(results[1], Exception) else {},
                'weather': results[2] if not isinstance(results[2], Exception) else None,
                'time_sync': results[3] if not isinstance(results[3], Exception) else {},
                'overall_risk_level': self._calculate_risk_level(results)
            }
            
            return integrated_status
        except Exception as e:
            self.logger.error(f"통합 상황 조회 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_risk_level(self, api_results: List[Any]) -> str:
        """API 결과를 기반으로 종합 위험도 계산"""
        risk_score = 0
        
        # 교통사고 위험도
        if api_results[0] and not isinstance(api_results[0], Exception):
            incidents = api_results[0]
            high_severity_count = sum(1 for inc in incidents if inc.severity >= 4)
            risk_score += high_severity_count * 10
        
        # 날씨 위험도
        if api_results[2] and not isinstance(api_results[2], Exception):
            weather = api_results[2]
            if weather.road_condition in ["결빙위험", "침수위험"]:
                risk_score += 30
            elif weather.road_condition in ["결빙주의", "습함"]:
                risk_score += 15
            
            if weather.precipitation > 10:
                risk_score += 20
            if weather.wind_speed > 15:
                risk_score += 10
        
        # 위험도 분류
        if risk_score >= 50:
            return "매우위험"
        elif risk_score >= 30:
            return "위험"
        elif risk_score >= 15:
            return "주의"
        else:
            return "안전"

class KoreanAPIService:
    """한국 정부 API 통합 서비스 매니저"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = KoreanGovernmentAPIClient()
        self.cache_duration = timedelta(minutes=5)
        self.cache = {}
    
    async def start_monitoring(self, locations: List[Dict[str, float]]):
        """실시간 모니터링 시작"""
        self.logger.info(f"{len(locations)}개 지점 실시간 모니터링 시작")
        
        async with self.client:
            while True:
                try:
                    for location in locations:
                        status = await self.client.get_integrated_status(location)
                        
                        # 위험 상황 알림
                        if status.get('overall_risk_level') in ['위험', '매우위험']:
                            await self._send_alert(location, status)
                        
                        # 캐시 업데이트
                        cache_key = f"{location['lat']},{location['lng']}"
                        self.cache[cache_key] = {
                            'data': status,
                            'timestamp': datetime.now()
                        }
                    
                    # 5분 간격으로 업데이트
                    await asyncio.sleep(300)
                    
                except Exception as e:
                    self.logger.error(f"모니터링 오류: {e}")
                    await asyncio.sleep(60)  # 오류 시 1분 후 재시도
    
    async def _send_alert(self, location: Dict[str, float], status: Dict[str, Any]):
        """위험 상황 알림 발송"""
        alert_message = f"""
🚨 A2Z TSN 네트워크 위험 알림

위치: {location['lat']:.4f}, {location['lng']:.4f}
위험도: {status['overall_risk_level']}
시각: {status['timestamp']}

상세 정보:
- 교통사고: {len(status.get('traffic_incidents', []))}건
- 날씨상태: {status.get('weather', {}).get('road_condition', '알수없음')}
- 기온: {status.get('weather', {}).get('temperature', 0)}°C

즉시 대응이 필요합니다.
        """
        
        self.logger.warning(alert_message)
        # 실제 구현에서는 SMS, 이메일, 웹훅 등으로 알림 발송

async def main():
    """테스트 및 데모"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 서울시 테스트 지점들
    test_locations = [
        {'lat': 37.5665, 'lng': 126.9780},  # 강남구 테헤란로
        {'lat': 37.5010, 'lng': 127.0374},  # 강남구 역삼로
        {'lat': 37.5045, 'lng': 127.0489}   # 강남구 선릉로
    ]
    
    api_service = KoreanAPIService()
    
    # 단일 조회 테스트
    async with api_service.client:
        status = await api_service.client.get_integrated_status(test_locations[0])
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    
    # 실시간 모니터링 (데모용으로 짧게)
    # await api_service.start_monitoring(test_locations)

if __name__ == "__main__":
    asyncio.run(main())