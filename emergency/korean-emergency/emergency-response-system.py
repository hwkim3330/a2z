#!/usr/bin/env python3
"""
A2Z 한국 긴급상황 대응 시스템
Korean Emergency Response System

주요 기능:
- 119 소방서 자동 연동
- 112 경찰서 자동 신고  
- 1339 응급의료정보센터 연결
- 교통정보센터 (TOPIS) 실시간 신고
- 긴급차량 우선신호 요청
- V2X 긴급상황 브로드캐스트
"""

import asyncio
import aiohttp
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import geopy.distance
import xmltodict

class EmergencyType(Enum):
    TRAFFIC_ACCIDENT = "교통사고"
    VEHICLE_FIRE = "차량화재"
    MEDICAL_EMERGENCY = "응급의료"
    MECHANICAL_FAILURE = "기계적고장"
    CYBER_ATTACK = "사이버공격"
    COLLISION = "충돌사고"
    ROLLOVER = "전복사고"
    EVACUATION = "승객대피"

class SeverityLevel(Enum):
    INFO = "정보"
    LOW = "경미"
    MEDIUM = "보통"
    HIGH = "심각"
    CRITICAL = "위험"

class ResponseStatus(Enum):
    INITIATED = "접수"
    DISPATCHED = "출동"
    ARRIVED = "현장도착"
    IN_PROGRESS = "처리중"
    RESOLVED = "해결완료"
    CANCELLED = "취소"

@dataclass
class EmergencyEvent:
    event_id: str
    event_type: EmergencyType
    severity: SeverityLevel
    location: Dict[str, float]  # lat, lng
    address: str
    description: str
    vehicle_id: str
    passenger_count: int
    timestamp: datetime
    
    # 센서 데이터
    sensor_data: Dict[str, Any]
    
    # 피해 상황
    injuries: List[Dict[str, Any]]
    damages: List[str]
    
    # 대응 상태
    response_status: ResponseStatus
    responding_agencies: List[str]
    estimated_arrival_time: Optional[datetime]

@dataclass
class EmergencyContact:
    agency_name: str
    phone_number: str
    api_endpoint: Optional[str]
    contact_person: str
    available_24h: bool
    response_time_minutes: int

class KoreanEmergencyResponse:
    """한국 긴급상황 대응 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_emergencies: Dict[str, EmergencyEvent] = {}
        
        # 한국 긴급 연락처
        self.emergency_contacts = {
            "fire_department": EmergencyContact(
                agency_name="소방서 (119)",
                phone_number="119",
                api_endpoint="https://api.fire.go.kr/emergency",
                contact_person="119상황실",
                available_24h=True,
                response_time_minutes=5
            ),
            "police": EmergencyContact(
                agency_name="경찰서 (112)",
                phone_number="112",
                api_endpoint="https://api.police.go.kr/emergency",
                contact_person="112신고센터",
                available_24h=True,
                response_time_minutes=7
            ),
            "medical": EmergencyContact(
                agency_name="응급의료정보센터 (1339)",
                phone_number="1339",
                api_endpoint="https://api.e-gen.or.kr/emergency",
                contact_person="응급의료상황실",
                available_24h=True,
                response_time_minutes=3
            ),
            "traffic_center": EmergencyContact(
                agency_name="교통정보센터 (TOPIS)",
                phone_number="02-120",
                api_endpoint="https://topis.seoul.go.kr/api/emergency",
                contact_person="교통상황실",
                available_24h=True,
                response_time_minutes=10
            ),
            "insurance": EmergencyContact(
                agency_name="보험사 24시간 접수센터",
                phone_number="1588-1234",
                api_endpoint=None,
                contact_person="사고접수팀",
                available_24h=True,
                response_time_minutes=15
            )
        }
        
        # 서울시 주요 병원 정보
        self.nearby_hospitals = [
            {
                "name": "삼성서울병원",
                "address": "서울 강남구 일원로 81",
                "coordinates": [37.4881, 127.0856],
                "emergency_phone": "02-3410-2114",
                "trauma_center": True,
                "distance_km": 0
            },
            {
                "name": "서울아산병원", 
                "address": "서울 송파구 올림픽로43길 88",
                "coordinates": [37.5265, 127.1075],
                "emergency_phone": "02-3010-3350",
                "trauma_center": True,
                "distance_km": 0
            },
            {
                "name": "강남세브란스병원",
                "address": "서울 강남구 언주로 211",
                "coordinates": [37.5192, 127.0367],
                "emergency_phone": "02-2019-4444",
                "trauma_center": False,
                "distance_km": 0
            }
        ]
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'A2Z-Emergency-System/1.0'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def detect_emergency_situation(self, vehicle_data: Dict[str, Any]) -> Optional[EmergencyEvent]:
        """차량 센서 데이터에서 긴급상황 탐지"""
        
        # 가속도 센서 분석 (급제동, 충돌 감지)
        acceleration_data = vehicle_data.get('acceleration', {})
        if acceleration_data:
            deceleration = acceleration_data.get('x', 0)  # 전후 가속도
            lateral_g = acceleration_data.get('y', 0)     # 좌우 가속도
            
            # 급제동 감지 (-0.8G 이상)
            if deceleration < -7.84:  # m/s² (-0.8G)
                return await self._create_emergency_event(
                    EmergencyType.TRAFFIC_ACCIDENT,
                    SeverityLevel.HIGH,
                    "급제동 감지 - 전방 장애물 또는 위험상황",
                    vehicle_data
                )
            
            # 측면 충격 감지 (±0.5G 이상)
            if abs(lateral_g) > 4.9:  # m/s² (±0.5G)
                return await self._create_emergency_event(
                    EmergencyType.COLLISION,
                    SeverityLevel.CRITICAL,
                    "측면 충격 감지 - 충돌사고 가능성",
                    vehicle_data
                )
        
        # 자이로 센서 분석 (전복 감지)
        gyroscope_data = vehicle_data.get('gyroscope', {})
        if gyroscope_data:
            roll_angle = gyroscope_data.get('roll', 0)
            pitch_angle = gyroscope_data.get('pitch', 0)
            
            # 전복 감지 (45도 이상 기울어짐)
            if abs(roll_angle) > 45 or abs(pitch_angle) > 30:
                return await self._create_emergency_event(
                    EmergencyType.ROLLOVER,
                    SeverityLevel.CRITICAL,
                    f"차량 전복 감지 - Roll: {roll_angle}°, Pitch: {pitch_angle}°",
                    vehicle_data
                )
        
        # 온도 센서 분석 (화재 감지)
        temperature_data = vehicle_data.get('temperature', {})
        if temperature_data:
            engine_temp = temperature_data.get('engine', 0)
            battery_temp = temperature_data.get('battery', 0)
            
            # 화재 위험 온도
            if engine_temp > 120 or battery_temp > 60:
                return await self._create_emergency_event(
                    EmergencyType.VEHICLE_FIRE,
                    SeverityLevel.CRITICAL,
                    f"과열 감지 - 엔진: {engine_temp}°C, 배터리: {battery_temp}°C",
                    vehicle_data
                )
        
        # 에어백 전개 감지
        airbag_data = vehicle_data.get('airbag_status', {})
        if airbag_data and any(airbag_data.values()):
            return await self._create_emergency_event(
                EmergencyType.COLLISION,
                SeverityLevel.CRITICAL,
                "에어백 전개 - 심각한 충돌사고",
                vehicle_data
            )
        
        # 심박수/생체신호 분석 (승객 응급상황)
        biometric_data = vehicle_data.get('passenger_biometrics', [])
        for passenger_id, biometrics in enumerate(biometric_data):
            heart_rate = biometrics.get('heart_rate', 70)
            
            # 이상 심박수 (40 미만 또는 120 초과)
            if heart_rate < 40 or heart_rate > 120:
                return await self._create_emergency_event(
                    EmergencyType.MEDICAL_EMERGENCY,
                    SeverityLevel.HIGH,
                    f"승객 {passenger_id+1} 생체신호 이상 - 심박수: {heart_rate}",
                    vehicle_data
                )
        
        # 시스템 보안 위협 감지
        security_data = vehicle_data.get('security_status', {})
        if security_data:
            intrusion_detected = security_data.get('intrusion_attempts', 0)
            if intrusion_detected > 0:
                return await self._create_emergency_event(
                    EmergencyType.CYBER_ATTACK,
                    SeverityLevel.HIGH,
                    f"사이버 공격 탐지 - {intrusion_detected}회 침입 시도",
                    vehicle_data
                )
        
        return None
    
    async def _create_emergency_event(self, event_type: EmergencyType, severity: SeverityLevel, 
                                    description: str, vehicle_data: Dict[str, Any]) -> EmergencyEvent:
        """긴급상황 이벤트 생성"""
        
        event_id = f"EMRG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        location = vehicle_data.get('location', {'lat': 37.5665, 'lng': 126.9780})
        
        # 주소 변환 (역지오코딩)
        address = await self._get_address_from_coordinates(location['lat'], location['lng'])
        
        emergency_event = EmergencyEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            location=location,
            address=address,
            description=description,
            vehicle_id=vehicle_data.get('vehicle_id', 'UNKNOWN'),
            passenger_count=vehicle_data.get('passenger_count', 0),
            timestamp=datetime.now(),
            sensor_data=vehicle_data,
            injuries=[],
            damages=[],
            response_status=ResponseStatus.INITIATED,
            responding_agencies=[],
            estimated_arrival_time=None
        )
        
        # 활성 긴급상황 목록에 추가
        self.active_emergencies[event_id] = emergency_event
        
        self.logger.critical(f"긴급상황 감지: {event_type.value} - {description}")
        return emergency_event
    
    async def _get_address_from_coordinates(self, lat: float, lng: float) -> str:
        """좌표를 주소로 변환"""
        try:
            # 카카오 맵 API를 사용한 역지오코딩 (실제 구현에서는 API 키 필요)
            # 여기서는 간단한 근사치 사용
            if 37.5 <= lat <= 37.6 and 126.9 <= lng <= 127.1:
                return f"서울특별시 강남구 (추정) {lat:.4f}, {lng:.4f}"
            else:
                return f"위치정보 {lat:.4f}, {lng:.4f}"
        except Exception as e:
            self.logger.error(f"주소 변환 실패: {e}")
            return f"위치정보 {lat:.4f}, {lng:.4f}"
    
    async def dispatch_emergency_response(self, emergency_event: EmergencyEvent) -> Dict[str, Any]:
        """긴급상황 대응 디스패치"""
        
        response_result = {
            'event_id': emergency_event.event_id,
            'dispatched_agencies': [],
            'estimated_response_times': {},
            'contact_confirmations': {},
            'errors': []
        }
        
        # 긴급상황 유형별 대응기관 결정
        required_agencies = self._determine_required_agencies(emergency_event)
        
        # 각 기관에 동시 신고
        dispatch_tasks = []
        for agency_key in required_agencies:
            if agency_key in self.emergency_contacts:
                task = self._notify_emergency_agency(agency_key, emergency_event)
                dispatch_tasks.append((agency_key, task))
        
        # 병렬 처리로 모든 기관에 동시 연락
        results = await asyncio.gather(*[task for _, task in dispatch_tasks], return_exceptions=True)
        
        for i, (agency_key, result) in enumerate(zip([key for key, _ in dispatch_tasks], results)):
            if isinstance(result, Exception):
                response_result['errors'].append(f"{agency_key}: {str(result)}")
                self.logger.error(f"{agency_key} 연락 실패: {result}")
            else:
                response_result['dispatched_agencies'].append(agency_key)
                response_result['contact_confirmations'][agency_key] = result
                response_result['estimated_response_times'][agency_key] = self.emergency_contacts[agency_key].response_time_minutes
        
        # 가장 가까운 병원 찾기 (의료 응급상황인 경우)
        if emergency_event.event_type in [EmergencyType.MEDICAL_EMERGENCY, EmergencyType.COLLISION, EmergencyType.ROLLOVER]:
            nearest_hospital = await self._find_nearest_hospital(emergency_event.location)
            response_result['nearest_hospital'] = nearest_hospital
        
        # V2X 긴급 방송
        await self._broadcast_emergency_v2x(emergency_event)
        
        # 교통신호 우선 요청 (해당하는 경우)
        if emergency_event.event_type in [EmergencyType.TRAFFIC_ACCIDENT, EmergencyType.COLLISION]:
            await self._request_traffic_priority(emergency_event)
        
        # 대응 상태 업데이트
        emergency_event.response_status = ResponseStatus.DISPATCHED
        emergency_event.responding_agencies = response_result['dispatched_agencies']
        
        return response_result
    
    def _determine_required_agencies(self, emergency_event: EmergencyEvent) -> List[str]:
        """긴급상황 유형별 필요한 대응기관 결정"""
        required = ['traffic_center']  # 모든 상황에서 교통정보센터 연락
        
        if emergency_event.event_type == EmergencyType.TRAFFIC_ACCIDENT:
            required.extend(['police', 'fire_department', 'medical'])
            
        elif emergency_event.event_type == EmergencyType.COLLISION:
            required.extend(['police', 'fire_department', 'medical'])
            if emergency_event.severity == SeverityLevel.CRITICAL:
                required.append('medical')  # 추가 의료진
                
        elif emergency_event.event_type == EmergencyType.ROLLOVER:
            required.extend(['fire_department', 'medical', 'police'])
            
        elif emergency_event.event_type == EmergencyType.VEHICLE_FIRE:
            required.extend(['fire_department', 'police'])
            if emergency_event.passenger_count > 0:
                required.append('medical')
                
        elif emergency_event.event_type == EmergencyType.MEDICAL_EMERGENCY:
            required.extend(['medical', 'fire_department'])
            
        elif emergency_event.event_type == EmergencyType.CYBER_ATTACK:
            required.extend(['police'])  # 사이버수사대
            
        elif emergency_event.event_type == EmergencyType.MECHANICAL_FAILURE:
            if emergency_event.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                required.extend(['police'])
        
        return list(set(required))  # 중복 제거
    
    async def _notify_emergency_agency(self, agency_key: str, emergency_event: EmergencyEvent) -> Dict[str, Any]:
        """개별 긴급기관 연락"""
        contact = self.emergency_contacts[agency_key]
        
        # 신고 데이터 준비
        report_data = {
            'incident_id': emergency_event.event_id,
            'incident_type': emergency_event.event_type.value,
            'severity': emergency_event.severity.value,
            'location': {
                'latitude': emergency_event.location['lat'],
                'longitude': emergency_event.location['lng'],
                'address': emergency_event.address
            },
            'description': emergency_event.description,
            'vehicle_info': {
                'vehicle_id': emergency_event.vehicle_id,
                'passenger_count': emergency_event.passenger_count,
                'vehicle_type': '12인승 자율주행 버스'
            },
            'timestamp': emergency_event.timestamp.isoformat(),
            'contact_info': {
                'reporting_system': 'A2Z 자율주행 차량',
                'emergency_phone': '1588-1234',
                'contact_person': '김한울 수석엔지니어'
            }
        }
        
        result = {'agency': contact.agency_name, 'status': 'failed', 'response': None}
        
        try:
            # API가 있는 경우 웹 신고
            if contact.api_endpoint:
                async with self.session.post(
                    contact.api_endpoint,
                    json=report_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        result['status'] = 'success'
                        result['response'] = response_data
                        self.logger.info(f"{contact.agency_name} API 신고 성공")
                    else:
                        result['error'] = f"HTTP {response.status}"
                        # API 실패시 전화 신고로 대체 시도
                        await self._make_emergency_call(contact, emergency_event)
            else:
                # API가 없는 경우 전화 신고
                call_result = await self._make_emergency_call(contact, emergency_event)
                result.update(call_result)
        
        except Exception as e:
            self.logger.error(f"{contact.agency_name} 연락 실패: {e}")
            result['error'] = str(e)
            # 최후 수단으로 전화 시도
            try:
                call_result = await self._make_emergency_call(contact, emergency_event)
                result.update(call_result)
            except:
                pass
        
        return result
    
    async def _make_emergency_call(self, contact: EmergencyContact, emergency_event: EmergencyEvent) -> Dict[str, Any]:
        """긴급 전화 신고 (실제 구현에서는 VoIP API 사용)"""
        
        # 음성 신고 메시지 생성
        call_message = f"""
        A2Z 자율주행 버스 긴급상황 신고입니다.

        사고종류: {emergency_event.event_type.value}
        심각도: {emergency_event.severity.value}
        위치: {emergency_event.address}
        좌표: 위도 {emergency_event.location['lat']:.4f}, 경도 {emergency_event.location['lng']:.4f}
        승객수: {emergency_event.passenger_count}명
        상황: {emergency_event.description}
        
        연락처: 1588-1234 (A2Z 상황실)
        담당자: 김한울 수석엔지니어
        """
        
        self.logger.critical(f"[전화신고] {contact.agency_name} ({contact.phone_number})")
        self.logger.critical(f"신고내용: {call_message.strip()}")
        
        # 실제 구현에서는 여기서 VoIP API 호출
        # 예: Twilio, Amazon Connect 등을 사용하여 자동 음성 신고
        
        return {
            'status': 'call_initiated',
            'phone_number': contact.phone_number,
            'message': call_message.strip(),
            'call_time': datetime.now().isoformat()
        }
    
    async def _find_nearest_hospital(self, location: Dict[str, float]) -> Dict[str, Any]:
        """가장 가까운 병원 찾기"""
        current_pos = (location['lat'], location['lng'])
        
        nearest_hospital = None
        min_distance = float('inf')
        
        for hospital in self.nearby_hospitals:
            hospital_pos = tuple(hospital['coordinates'])
            distance = geopy.distance.geodesic(current_pos, hospital_pos).kilometers
            
            if distance < min_distance:
                min_distance = distance
                nearest_hospital = hospital.copy()
                nearest_hospital['distance_km'] = round(distance, 2)
        
        if nearest_hospital:
            self.logger.info(f"가장 가까운 병원: {nearest_hospital['name']} ({nearest_hospital['distance_km']}km)")
        
        return nearest_hospital
    
    async def _broadcast_emergency_v2x(self, emergency_event: EmergencyEvent):
        """V2X를 통한 긴급상황 브로드캐스트"""
        
        v2x_message = {
            'message_type': 'EMERGENCY_VEHICLE_ALERT',
            'emergency_type': emergency_event.event_type.value,
            'severity': emergency_event.severity.value,
            'location': emergency_event.location,
            'radius_meters': 1000,  # 1km 반경
            'message': f"{emergency_event.event_type.value} 발생 - 주의운전 바랍니다",
            'duration_seconds': 300,  # 5분간 방송
            'timestamp': emergency_event.timestamp.isoformat()
        }
        
        self.logger.info(f"V2X 긴급방송: {v2x_message['message']}")
        
        # 실제 구현에서는 V2X 모듈로 메시지 전송
        # 예: C-V2X, DSRC를 통한 주변 차량 알림
    
    async def _request_traffic_priority(self, emergency_event: EmergencyEvent):
        """교통신호 우선신호 요청"""
        
        priority_request = {
            'request_type': 'EMERGENCY_SIGNAL_PRIORITY',
            'incident_id': emergency_event.event_id,
            'location': emergency_event.location,
            'priority_level': 'HIGH' if emergency_event.severity == SeverityLevel.CRITICAL else 'MEDIUM',
            'duration_minutes': 30,
            'reason': f"{emergency_event.event_type.value} - 응급차량 통행 우선"
        }
        
        self.logger.info(f"교통신호 우선요청: {priority_request['reason']}")
        
        # 실제 구현에서는 교통관리시스템 API 호출
        # 예: UTIS(도시교통정보시스템)와 연동
    
    async def update_emergency_status(self, event_id: str, new_status: ResponseStatus, 
                                    update_info: Optional[Dict[str, Any]] = None) -> bool:
        """긴급상황 상태 업데이트"""
        
        if event_id not in self.active_emergencies:
            self.logger.error(f"존재하지 않는 긴급상황: {event_id}")
            return False
        
        emergency_event = self.active_emergencies[event_id]
        old_status = emergency_event.response_status
        emergency_event.response_status = new_status
        
        if update_info:
            if 'estimated_arrival_time' in update_info:
                emergency_event.estimated_arrival_time = datetime.fromisoformat(update_info['estimated_arrival_time'])
            
            if 'injuries' in update_info:
                emergency_event.injuries.extend(update_info['injuries'])
            
            if 'damages' in update_info:
                emergency_event.damages.extend(update_info['damages'])
        
        self.logger.info(f"긴급상황 {event_id} 상태 변경: {old_status.value} → {new_status.value}")
        
        # 해결 완료시 활성 목록에서 제거
        if new_status == ResponseStatus.RESOLVED:
            self.active_emergencies.pop(event_id, None)
            self.logger.info(f"긴급상황 {event_id} 해결 완료 및 종료")
        
        return True
    
    def get_active_emergencies(self) -> List[Dict[str, Any]]:
        """현재 활성 긴급상황 목록"""
        active_list = []
        
        for event_id, emergency in self.active_emergencies.items():
            active_list.append({
                'event_id': event_id,
                'type': emergency.event_type.value,
                'severity': emergency.severity.value,
                'location': emergency.location,
                'address': emergency.address,
                'status': emergency.response_status.value,
                'elapsed_time_minutes': int((datetime.now() - emergency.timestamp).total_seconds() / 60),
                'responding_agencies': emergency.responding_agencies
            })
        
        return sorted(active_list, key=lambda x: x['elapsed_time_minutes'], reverse=True)
    
    async def generate_emergency_report(self, event_id: str) -> Dict[str, Any]:
        """긴급상황 상세 보고서 생성"""
        
        if event_id not in self.active_emergencies:
            return {'error': f'긴급상황 {event_id}를 찾을 수 없습니다'}
        
        emergency = self.active_emergencies[event_id]
        
        report = {
            'basic_info': {
                'event_id': emergency.event_id,
                'type': emergency.event_type.value,
                'severity': emergency.severity.value,
                'timestamp': emergency.timestamp.isoformat(),
                'location': emergency.location,
                'address': emergency.address,
                'vehicle_id': emergency.vehicle_id,
                'passenger_count': emergency.passenger_count
            },
            'incident_details': {
                'description': emergency.description,
                'sensor_data_summary': self._summarize_sensor_data(emergency.sensor_data),
                'injuries': emergency.injuries,
                'damages': emergency.damages
            },
            'response_info': {
                'status': emergency.response_status.value,
                'responding_agencies': emergency.responding_agencies,
                'estimated_arrival': emergency.estimated_arrival_time.isoformat() if emergency.estimated_arrival_time else None,
                'elapsed_time_minutes': int((datetime.now() - emergency.timestamp).total_seconds() / 60)
            },
            'generated_at': datetime.now().isoformat(),
            'report_version': '1.0'
        }
        
        return report
    
    def _summarize_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """센서 데이터 요약"""
        summary = {}
        
        if 'acceleration' in sensor_data:
            acc = sensor_data['acceleration']
            summary['최대감속도'] = f"{acc.get('x', 0):.2f} m/s²"
            summary['측면가속도'] = f"{acc.get('y', 0):.2f} m/s²"
        
        if 'gyroscope' in sensor_data:
            gyro = sensor_data['gyroscope']
            summary['롤각도'] = f"{gyro.get('roll', 0):.1f}°"
            summary['피치각도'] = f"{gyro.get('pitch', 0):.1f}°"
        
        if 'temperature' in sensor_data:
            temp = sensor_data['temperature']
            summary['엔진온도'] = f"{temp.get('engine', 0):.1f}°C"
            summary['배터리온도'] = f"{temp.get('battery', 0):.1f}°C"
        
        return summary

async def main():
    """테스트 및 데모"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 테스트 차량 데이터 (충돌 시나리오)
    test_vehicle_data = {
        'vehicle_id': 'A2Z_BUS_001',
        'location': {'lat': 37.5665, 'lng': 126.9780},
        'passenger_count': 8,
        'acceleration': {
            'x': -9.5,  # 급제동 (-0.97G)
            'y': 0.2,
            'z': -9.8
        },
        'gyroscope': {
            'roll': 5.0,
            'pitch': -8.0,
            'yaw': 0.0
        },
        'temperature': {
            'engine': 95.0,
            'battery': 45.0
        },
        'airbag_status': {
            'front_left': True,  # 에어백 전개
            'front_right': False,
            'side_left': False,
            'side_right': False
        },
        'passenger_biometrics': [
            {'heart_rate': 95},  # 정상
            {'heart_rate': 130}, # 높음 (스트레스)
            {'heart_rate': 88}   # 정상
        ]
    }
    
    emergency_system = KoreanEmergencyResponse()
    
    async with emergency_system:
        # 1. 긴급상황 탐지
        emergency_event = await emergency_system.detect_emergency_situation(test_vehicle_data)
        
        if emergency_event:
            print("=== 긴급상황 탐지 ===")
            print(json.dumps(asdict(emergency_event), ensure_ascii=False, indent=2, default=str))
            
            # 2. 긴급 대응 디스패치
            dispatch_result = await emergency_system.dispatch_emergency_response(emergency_event)
            print("\n=== 긴급 대응 결과 ===")
            print(json.dumps(dispatch_result, ensure_ascii=False, indent=2, default=str))
            
            # 3. 상태 업데이트 시뮬레이션
            await emergency_system.update_emergency_status(
                emergency_event.event_id,
                ResponseStatus.ARRIVED,
                {'estimated_arrival_time': (datetime.now() + timedelta(minutes=5)).isoformat()}
            )
            
            # 4. 상세 보고서 생성
            report = await emergency_system.generate_emergency_report(emergency_event.event_id)
            print("\n=== 상세 보고서 ===")
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            
            # 5. 활성 긴급상황 목록
            active_emergencies = emergency_system.get_active_emergencies()
            print("\n=== 활성 긴급상황 목록 ===")
            print(json.dumps(active_emergencies, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())