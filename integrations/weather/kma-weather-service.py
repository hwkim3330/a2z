#!/usr/bin/env python3
"""
A2Z 기상청 날씨 서비스 통합
Korea Meteorological Administration (KMA) Weather Service Integration

주요 기능:
- 실시간 날씨 정보 수집 (기상청 API)
- 도로 위험도 평가 (결빙, 침수, 시야 제한)
- 자율주행 안전 권고사항 생성
- 날씨 기반 TSN QoS 조정
"""

import asyncio
import aiohttp
import logging
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

class WeatherCondition(Enum):
    CLEAR = "맑음"
    PARTLY_CLOUDY = "구름조금"
    MOSTLY_CLOUDY = "구름많음" 
    CLOUDY = "흐림"
    RAIN = "비"
    SNOW = "눈"
    SLEET = "진눈깨비"
    FOG = "안개"
    THUNDERSTORM = "뇌우"

class RoadRiskLevel(Enum):
    SAFE = "안전"
    CAUTION = "주의"
    WARNING = "경고"
    DANGER = "위험"
    CRITICAL = "심각"

@dataclass
class WeatherData:
    location_name: str
    coordinates: Tuple[float, float]  # (lat, lng)
    grid_coordinates: Tuple[int, int]  # (nx, ny)
    
    # 기본 기상 정보
    temperature: float  # 섭씨
    humidity: float     # %
    precipitation: float  # mm/h
    wind_speed: float   # m/s
    wind_direction: int # 도 (0-360)
    visibility: float   # km
    atmospheric_pressure: float  # hPa
    
    # 도로 관련
    road_temperature: float  # 노면온도
    road_condition: str
    freeze_risk: bool
    flood_risk: bool
    visibility_risk: bool
    
    # 예보 정보
    weather_condition: WeatherCondition
    precipitation_probability: float  # %
    
    # 시간 정보
    observation_time: datetime
    forecast_valid_time: Optional[datetime] = None

@dataclass
class DrivingRecommendation:
    risk_level: RoadRiskLevel
    max_safe_speed: int  # km/h
    following_distance_multiplier: float
    headlight_required: bool
    fog_light_required: bool
    hazard_warning: bool
    advisory_message: str
    tsn_priority_adjustment: Dict[str, int]

class KMAWeatherService:
    """기상청 날씨 서비스 클라이언트"""
    
    def __init__(self, service_key: str):
        self.service_key = service_key
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://apis.data.go.kr/1360000"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 서울시 주요 지점의 격자 좌표
        self.seoul_grid_points = {
            "강남구": (61, 126),
            "서초구": (61, 125), 
            "송파구": (62, 126),
            "강동구": (62, 127),
            "영등포구": (58, 126),
            "마포구": (59, 127),
            "종로구": (60, 127),
            "중구": (60, 127),
            "용산구": (60, 126),
            "성동구": (61, 127)
        }
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'A2Z-TSN-Platform/1.0 (Weather Service)'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    def _convert_coordinates_to_grid(self, lat: float, lng: float) -> Tuple[int, int]:
        """위경도를 기상청 격자 좌표로 변환"""
        # 기상청 격자 변환 공식 (람베르트 정각원추도법)
        RE = 6371.00877     # 지구 반경 (km)
        GRID = 5.0          # 격자 간격 (km)
        SLAT1 = 30.0        # 투영 위도1
        SLAT2 = 60.0        # 투영 위도2
        OLON = 126.0        # 기준점 경도
        OLAT = 38.0         # 기준점 위도
        XO = 210 / GRID     # 기준점 X좌표
        YO = 675 / GRID     # 기준점 Y좌표
        
        DEGRAD = math.pi / 180.0
        re = RE / GRID
        slat1 = SLAT1 * DEGRAD
        slat2 = SLAT2 * DEGRAD
        olon = OLON * DEGRAD
        olat = OLAT * DEGRAD
        
        sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
        sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
        sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
        sf = math.pow(sf, sn) * math.cos(slat1) / sn
        ro = re * sf / math.pow(math.tan(math.pi * 0.25 + olat * 0.5), sn)
        
        ra = re * sf / math.pow(math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5), sn)
        theta = lng * DEGRAD - olon
        if theta > math.pi:
            theta -= 2.0 * math.pi
        if theta < -math.pi:
            theta += 2.0 * math.pi
        theta *= sn
        
        nx = int(ra * math.sin(theta) + XO + 0.5)
        ny = int(ro - ra * math.cos(theta) + YO + 0.5)
        
        return (nx, ny)
    
    async def get_current_weather(self, lat: float, lng: float, location_name: str = "") -> WeatherData:
        """현재 날씨 정보 조회"""
        nx, ny = self._convert_coordinates_to_grid(lat, lng)
        
        # 현재시각 기준으로 가장 최근 관측 시간 계산
        now = datetime.now()
        base_date = now.strftime('%Y%m%d')
        base_time = f"{now.hour:02d}00"
        
        # 초단기실황조회 API
        url = f"{self.base_url}/VilageFcstInfoService_2.0/getUltraSrtNcst"
        params = {
            'serviceKey': self.service_key,
            'numOfRows': '10',
            'pageNo': '1',
            'base_date': base_date,
            'base_time': base_time,
            'nx': nx,
            'ny': ny,
            'dataType': 'JSON'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error(f"기상청 API 오류: {response.status}")
                    return self._create_default_weather_data(lat, lng, location_name)
                
                data = await response.json()
                items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                
                # 기상 데이터 파싱
                weather_values = {}
                for item in items:
                    category = item['category']
                    value = item['obsrValue']
                    
                    if category == 'T1H':      # 기온
                        weather_values['temperature'] = float(value)
                    elif category == 'REH':    # 습도
                        weather_values['humidity'] = float(value)
                    elif category == 'RN1':    # 강수량
                        weather_values['precipitation'] = float(value)
                    elif category == 'WSD':    # 풍속
                        weather_values['wind_speed'] = float(value)
                    elif category == 'VEC':    # 풍향
                        weather_values['wind_direction'] = int(float(value))
                    elif category == 'UUU':    # 동서바람성분
                        weather_values['wind_u'] = float(value)
                    elif category == 'VVV':    # 남북바람성분
                        weather_values['wind_v'] = float(value)
                
                # 도로 상태 및 위험도 계산
                road_data = await self._calculate_road_conditions(weather_values, lat, lng)
                
                weather_data = WeatherData(
                    location_name=location_name or f"위도{lat:.3f}_경도{lng:.3f}",
                    coordinates=(lat, lng),
                    grid_coordinates=(nx, ny),
                    temperature=weather_values.get('temperature', 0),
                    humidity=weather_values.get('humidity', 50),
                    precipitation=weather_values.get('precipitation', 0),
                    wind_speed=weather_values.get('wind_speed', 0),
                    wind_direction=weather_values.get('wind_direction', 0),
                    visibility=road_data['visibility'],
                    atmospheric_pressure=1013.25,  # 기본값
                    road_temperature=road_data['road_temperature'],
                    road_condition=road_data['condition'],
                    freeze_risk=road_data['freeze_risk'],
                    flood_risk=road_data['flood_risk'],
                    visibility_risk=road_data['visibility_risk'],
                    weather_condition=self._determine_weather_condition(weather_values),
                    precipitation_probability=0,  # 현재 관측에서는 0
                    observation_time=now
                )
                
                self.logger.info(f"날씨 정보 수집 완료: {location_name}, 온도 {weather_data.temperature}°C")
                return weather_data
                
        except Exception as e:
            self.logger.error(f"날씨 정보 조회 실패: {e}")
            return self._create_default_weather_data(lat, lng, location_name)
    
    async def get_weather_forecast(self, lat: float, lng: float, hours: int = 24) -> List[WeatherData]:
        """날씨 예보 정보 조회"""
        nx, ny = self._convert_coordinates_to_grid(lat, lng)
        
        # 예보 시간 계산
        now = datetime.now()
        base_date = now.strftime('%Y%m%d')
        
        # 발표시각에 맞춰 조정 (02, 05, 08, 11, 14, 17, 20, 23시)
        forecast_hours = [2, 5, 8, 11, 14, 17, 20, 23]
        base_time = max([h for h in forecast_hours if h <= now.hour], default=23)
        if base_time == 23 and now.hour < 2:
            base_date = (now - timedelta(days=1)).strftime('%Y%m%d')
        
        base_time_str = f"{base_time:02d}00"
        
        # 단기예보조회 API
        url = f"{self.base_url}/VilageFcstInfoService_2.0/getVilageFcst"
        params = {
            'serviceKey': self.service_key,
            'numOfRows': '1000',
            'pageNo': '1',
            'base_date': base_date,
            'base_time': base_time_str,
            'nx': nx,
            'ny': ny,
            'dataType': 'JSON'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error(f"예보 API 오류: {response.status}")
                    return []
                
                data = await response.json()
                items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                
                # 시간별로 그룹화
                forecast_by_time = {}
                for item in items:
                    fcst_date = item['fcstDate']
                    fcst_time = item['fcstTime']
                    datetime_key = f"{fcst_date}_{fcst_time}"
                    
                    if datetime_key not in forecast_by_time:
                        forecast_by_time[datetime_key] = {}
                    
                    category = item['category']
                    value = item['fcstValue']
                    forecast_by_time[datetime_key][category] = value
                
                # WeatherData 객체 생성
                forecasts = []
                for datetime_key, values in forecast_by_time.items():
                    if len(forecasts) >= hours:
                        break
                    
                    fcst_datetime = datetime.strptime(datetime_key, '%Y%m%d_%H%M')
                    
                    # 필요한 데이터 추출
                    temp = float(values.get('TMP', 0))
                    humidity = float(values.get('REH', 50))
                    precip_prob = float(values.get('POP', 0))
                    precip_type = values.get('PTY', '0')
                    wind_speed = float(values.get('WSD', 0))
                    wind_dir = int(float(values.get('VEC', 0)))
                    
                    # 강수량 계산 (강수형태와 확률 기반)
                    precipitation = 0
                    if precip_type != '0' and precip_prob > 30:
                        precipitation = precip_prob / 20  # 대략적인 강수량
                    
                    # 도로 상태 계산
                    road_data = await self._calculate_road_conditions({
                        'temperature': temp,
                        'humidity': humidity,
                        'precipitation': precipitation,
                        'wind_speed': wind_speed
                    }, lat, lng)
                    
                    weather_forecast = WeatherData(
                        location_name=f"예보_위도{lat:.3f}_경도{lng:.3f}",
                        coordinates=(lat, lng),
                        grid_coordinates=(nx, ny),
                        temperature=temp,
                        humidity=humidity,
                        precipitation=precipitation,
                        wind_speed=wind_speed,
                        wind_direction=wind_dir,
                        visibility=road_data['visibility'],
                        atmospheric_pressure=1013.25,
                        road_temperature=road_data['road_temperature'],
                        road_condition=road_data['condition'],
                        freeze_risk=road_data['freeze_risk'],
                        flood_risk=road_data['flood_risk'],
                        visibility_risk=road_data['visibility_risk'],
                        weather_condition=self._determine_weather_condition_from_code(precip_type),
                        precipitation_probability=precip_prob,
                        observation_time=now,
                        forecast_valid_time=fcst_datetime
                    )
                    
                    forecasts.append(weather_forecast)
                
                self.logger.info(f"날씨 예보 수집 완료: {len(forecasts)}시간")
                return sorted(forecasts, key=lambda x: x.forecast_valid_time)
                
        except Exception as e:
            self.logger.error(f"날씨 예보 조회 실패: {e}")
            return []
    
    async def _calculate_road_conditions(self, weather_values: Dict[str, float], lat: float, lng: float) -> Dict[str, Any]:
        """기상 데이터를 기반으로 도로 상태 계산"""
        temp = weather_values.get('temperature', 0)
        humidity = weather_values.get('humidity', 50)
        precipitation = weather_values.get('precipitation', 0)
        wind_speed = weather_values.get('wind_speed', 0)
        
        # 노면온도 추정 (기온보다 약간 높거나 낮음)
        if precipitation > 0:
            road_temp = temp - 1  # 습한 노면은 기온보다 낮음
        else:
            road_temp = temp + 2  # 건조한 노면은 기온보다 높음
        
        # 결빙 위험도
        freeze_risk = road_temp <= 0 and humidity > 80
        
        # 침수 위험도 
        flood_risk = precipitation > 10  # 시간당 10mm 이상
        
        # 시야 제한 위험도
        visibility_risk = precipitation > 5 or humidity > 90 or wind_speed > 10
        
        # 가시거리 계산 
        if precipitation > 20:
            visibility = 0.5  # 호우시
        elif precipitation > 10:
            visibility = 1.0  # 강우시
        elif precipitation > 0:
            visibility = 3.0  # 약한비
        elif humidity > 95:
            visibility = 2.0  # 안개
        else:
            visibility = 10.0  # 맑음
        
        # 도로 상태 문자열
        if freeze_risk:
            condition = "결빙위험"
        elif flood_risk:
            condition = "침수위험" 
        elif precipitation > 0:
            condition = "습함"
        else:
            condition = "건조"
        
        return {
            'road_temperature': road_temp,
            'condition': condition,
            'freeze_risk': freeze_risk,
            'flood_risk': flood_risk,
            'visibility_risk': visibility_risk,
            'visibility': visibility
        }
    
    def _determine_weather_condition(self, weather_values: Dict[str, float]) -> WeatherCondition:
        """기상 관측값으로부터 날씨 상태 판정"""
        precipitation = weather_values.get('precipitation', 0)
        temp = weather_values.get('temperature', 0)
        humidity = weather_values.get('humidity', 50)
        
        if precipitation > 0:
            if temp < 0:
                return WeatherCondition.SNOW
            elif temp < 2:
                return WeatherCondition.SLEET
            else:
                return WeatherCondition.RAIN
        elif humidity > 95:
            return WeatherCondition.FOG
        elif humidity > 80:
            return WeatherCondition.CLOUDY
        elif humidity > 60:
            return WeatherCondition.MOSTLY_CLOUDY
        elif humidity > 40:
            return WeatherCondition.PARTLY_CLOUDY
        else:
            return WeatherCondition.CLEAR
    
    def _determine_weather_condition_from_code(self, precip_type: str) -> WeatherCondition:
        """기상청 강수형태 코드로부터 날씨 상태 판정"""
        if precip_type == '0':
            return WeatherCondition.CLEAR
        elif precip_type == '1':
            return WeatherCondition.RAIN
        elif precip_type == '2':
            return WeatherCondition.SLEET
        elif precip_type == '3':
            return WeatherCondition.SNOW
        elif precip_type == '4':
            return WeatherCondition.THUNDERSTORM
        else:
            return WeatherCondition.CLOUDY
    
    def _create_default_weather_data(self, lat: float, lng: float, location_name: str) -> WeatherData:
        """기본 날씨 데이터 생성 (API 실패시)"""
        nx, ny = self._convert_coordinates_to_grid(lat, lng)
        
        return WeatherData(
            location_name=location_name or "알수없음",
            coordinates=(lat, lng),
            grid_coordinates=(nx, ny),
            temperature=15.0,
            humidity=60.0,
            precipitation=0.0,
            wind_speed=2.0,
            wind_direction=180,
            visibility=10.0,
            atmospheric_pressure=1013.25,
            road_temperature=17.0,
            road_condition="건조",
            freeze_risk=False,
            flood_risk=False,
            visibility_risk=False,
            weather_condition=WeatherCondition.CLEAR,
            precipitation_probability=0,
            observation_time=datetime.now()
        )

class AutonomousDrivingAdvisor:
    """날씨 기반 자율주행 안전 권고 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_driving_recommendation(self, weather: WeatherData) -> DrivingRecommendation:
        """날씨 정보 기반 운전 권고사항 생성"""
        risk_level = self._assess_overall_risk(weather)
        
        # 기본 권고사항
        max_safe_speed = 60  # km/h
        following_distance_multiplier = 1.0
        headlight_required = False
        fog_light_required = False
        hazard_warning = False
        advisory_message = "정상 운행"
        
        # TSN 우선순위 조정 (기본값)
        tsn_priority = {
            "emergency_brake": 7,
            "steering_control": 7,
            "lidar_data": 6,
            "camera_data": 5,
            "v2x_communication": 4
        }
        
        # 날씨별 세부 조정
        if weather.freeze_risk:
            max_safe_speed = min(max_safe_speed, 30)
            following_distance_multiplier = 2.0
            advisory_message = "결빙 위험 - 서행 및 안전거리 확보"
            # 결빙시 제동 신호 우선순위 최고로
            tsn_priority["emergency_brake"] = 7
            
        if weather.flood_risk:
            max_safe_speed = min(max_safe_speed, 40) 
            following_distance_multiplier = 1.5
            advisory_message = "침수 위험 - 저속 운행"
            hazard_warning = True
            
        if weather.visibility < 1.0:  # 1km 미만
            max_safe_speed = min(max_safe_speed, 30)
            following_distance_multiplier = 2.0
            headlight_required = True
            fog_light_required = True
            advisory_message = "시야 불량 - 등화 점등 및 서행"
            # 시야 불량시 카메라/라이다 데이터 우선순위 증가
            tsn_priority["lidar_data"] = 7
            tsn_priority["camera_data"] = 6
            
        elif weather.visibility < 3.0:  # 3km 미만
            max_safe_speed = min(max_safe_speed, 50)
            following_distance_multiplier = 1.5
            headlight_required = True
            
        if weather.precipitation > 10:  # 강우
            max_safe_speed = min(max_safe_speed, 45)
            following_distance_multiplier = 1.8
            headlight_required = True
            advisory_message = "강우 - 안전거리 확보 및 서행"
            
        if weather.wind_speed > 15:  # 강풍
            max_safe_speed = min(max_safe_speed, 70)
            advisory_message = "강풍 주의 - 차량 제어 유의"
            # 강풍시 조향 제어 우선순위 증가
            tsn_priority["steering_control"] = 7
            
        # 야간 운행 (일몰 후 일출 전)
        current_hour = weather.observation_time.hour
        if current_hour < 6 or current_hour > 18:
            headlight_required = True
            if not advisory_message.startswith("정상"):
                advisory_message += " (야간)"
        
        return DrivingRecommendation(
            risk_level=risk_level,
            max_safe_speed=max_safe_speed,
            following_distance_multiplier=following_distance_multiplier,
            headlight_required=headlight_required,
            fog_light_required=fog_light_required,
            hazard_warning=hazard_warning,
            advisory_message=advisory_message,
            tsn_priority_adjustment=tsn_priority
        )
    
    def _assess_overall_risk(self, weather: WeatherData) -> RoadRiskLevel:
        """종합 도로 위험도 평가"""
        risk_score = 0
        
        # 결빙 위험
        if weather.freeze_risk:
            risk_score += 40
        elif weather.road_temperature < 2:
            risk_score += 20
            
        # 침수 위험  
        if weather.flood_risk:
            risk_score += 35
        elif weather.precipitation > 5:
            risk_score += 15
            
        # 시야 위험
        if weather.visibility < 1:
            risk_score += 30
        elif weather.visibility < 3:
            risk_score += 15
        elif weather.visibility < 5:
            risk_score += 10
            
        # 바람 위험
        if weather.wind_speed > 20:
            risk_score += 25
        elif weather.wind_speed > 15:
            risk_score += 15
        elif weather.wind_speed > 10:
            risk_score += 10
        
        # 위험도 분류
        if risk_score >= 60:
            return RoadRiskLevel.CRITICAL
        elif risk_score >= 40:
            return RoadRiskLevel.DANGER  
        elif risk_score >= 20:
            return RoadRiskLevel.WARNING
        elif risk_score >= 10:
            return RoadRiskLevel.CAUTION
        else:
            return RoadRiskLevel.SAFE

class WeatherBasedTSNController:
    """날씨 기반 TSN QoS 제어기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def adjust_tsn_parameters(self, weather: WeatherData, recommendation: DrivingRecommendation) -> Dict[str, Any]:
        """날씨 상황에 따른 TSN 파라미터 조정"""
        
        # 기본 TSN 설정
        tsn_config = {
            "time_aware_shaper": {
                "cycle_time": "125us",  # 기본 8kHz
                "gate_control_list": []
            },
            "credit_based_shaper": {
                "idle_slopes": {}
            },
            "frer_config": {
                "redundancy_level": 1  # 기본 이중화
            },
            "buffer_sizes": {},
            "latency_requirements": {}
        }
        
        # 위험 상황별 조정
        if recommendation.risk_level in [RoadRiskLevel.DANGER, RoadRiskLevel.CRITICAL]:
            # 긴급 상황 - 제어 신호 최우선
            tsn_config["time_aware_shaper"]["cycle_time"] = "62.5us"  # 16kHz로 증가
            tsn_config["frer_config"]["redundancy_level"] = 2  # 삼중 이중화
            
            # 제어 신호에 더 많은 시간 할당
            gate_control = [
                {"priority": 7, "gate_states": "oooooooo", "time_interval": "31.25us"},  # 제어신호 50%
                {"priority": 6, "gate_states": "ccoooooo", "time_interval": "15.625us"}, # 센서데이터 25% 
                {"priority": 5, "gate_states": "ccccoooo", "time_interval": "10.625us"},  # 기타 15%
                {"priority": 0, "gate_states": "ccccccco", "time_interval": "5us"}       # 일반데이터 10%
            ]
        elif recommendation.risk_level == RoadRiskLevel.WARNING:
            # 경고 상황 - 센서 데이터 우선순위 증가
            gate_control = [
                {"priority": 7, "gate_states": "oCCCCCCC", "time_interval": "15.625us"},
                {"priority": 6, "gate_states": "CooCCCCC", "time_interval": "20us"},     # 센서 증가
                {"priority": 5, "gate_states": "CCoooCCC", "time_interval": "15us"},
                {"priority": 0, "gate_states": "CCCCCooo", "time_interval": "12.5us"}
            ]
        else:
            # 정상 상황 - 기본 설정
            gate_control = [
                {"priority": 7, "gate_states": "oCCCCCCC", "time_interval": "15.625us"},
                {"priority": 6, "gate_states": "CoCCCCCC", "time_interval": "15.625us"},
                {"priority": 5, "gate_states": "CCoCCCCC", "time_interval": "15.625us"},
                {"priority": 0, "gate_states": "CCCoooooo", "time_interval": "78.125us"}
            ]
        
        tsn_config["time_aware_shaper"]["gate_control_list"] = gate_control
        
        # Credit-Based Shaper 대역폭 조정
        if weather.visibility_risk:
            # 시야 불량시 카메라/라이다 대역폭 증가
            tsn_config["credit_based_shaper"]["idle_slopes"] = {
                "lidar_data": 500000,   # 500Mbps (기본 400Mbps에서 증가)
                "camera_data": 400000   # 400Mbps (기본 300Mbps에서 증가)
            }
        
        # 버퍼 크기 조정
        if recommendation.risk_level in [RoadRiskLevel.DANGER, RoadRiskLevel.CRITICAL]:
            tsn_config["buffer_sizes"] = {
                "emergency_brake": 2048,    # 증가
                "steering_control": 2048,   # 증가 
                "lidar_data": 8192,        # 증가
                "camera_data": 8192        # 증가
            }
        
        # 지연시간 요구사항
        tsn_config["latency_requirements"] = {
            "emergency_brake": "1ms",
            "steering_control": "2ms", 
            "lidar_data": "5ms",
            "camera_data": "10ms",
            "v2x_communication": "50ms"
        }
        
        self.logger.info(f"TSN 파라미터 조정 완료 - 위험도: {recommendation.risk_level.value}")
        return tsn_config

async def main():
    """테스트 및 데모"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 기상청 API 키 (실제 사용시에는 환경변수나 설정파일에서 로드)
    service_key = "여기에_실제_API_키_입력"
    
    # 서울 강남구 테헤란로 좌표
    test_location = (37.5665, 126.9780)
    
    async with KMAWeatherService(service_key) as weather_service:
        # 현재 날씨 조회
        current_weather = await weather_service.get_current_weather(
            test_location[0], test_location[1], "강남구 테헤란로"
        )
        
        print("=== 현재 날씨 정보 ===")
        print(json.dumps(asdict(current_weather), ensure_ascii=False, indent=2, default=str))
        
        # 운전 권고사항 생성
        advisor = AutonomousDrivingAdvisor()
        recommendation = advisor.generate_driving_recommendation(current_weather)
        
        print("\n=== 자율주행 권고사항 ===")
        print(json.dumps(asdict(recommendation), ensure_ascii=False, indent=2, default=str))
        
        # TSN 파라미터 조정
        tsn_controller = WeatherBasedTSNController()
        tsn_config = tsn_controller.adjust_tsn_parameters(current_weather, recommendation)
        
        print("\n=== TSN 파라미터 조정 ===")
        print(json.dumps(tsn_config, ensure_ascii=False, indent=2))
        
        # 24시간 예보 (실제 API 키가 있을 때만 실행)
        if service_key != "여기에_실제_API_키_입력":
            forecasts = await weather_service.get_weather_forecast(
                test_location[0], test_location[1], hours=24
            )
            
            print(f"\n=== 24시간 날씨 예보 ({len(forecasts)}시간) ===")
            for i, forecast in enumerate(forecasts[:6]):  # 첫 6시간만 출력
                print(f"{i+1}시간 후: 온도 {forecast.temperature}°C, "
                      f"강수 {forecast.precipitation}mm, 위험도 {advisor._assess_overall_risk(forecast).value}")

if __name__ == "__main__":
    asyncio.run(main())