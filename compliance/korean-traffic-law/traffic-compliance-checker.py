#!/usr/bin/env python3
"""
A2Z 한국 교통법규 준수 검증 시스템
Korean Traffic Law Compliance Checker

주요 법규:
- 도로교통법 (Road Traffic Act)
- 자동차관리법 (Motor Vehicle Management Act)  
- 자율주행자동차 상용화 촉진 및 지원에 관한 법률
- 개인정보보호법 (Personal Information Protection Act)
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import math

class ViolationType(Enum):
    SPEED_VIOLATION = "speed_violation"
    LANE_VIOLATION = "lane_violation"  
    SIGNAL_VIOLATION = "signal_violation"
    FOLLOWING_DISTANCE = "following_distance"
    OVERTAKING_VIOLATION = "overtaking_violation"
    PEDESTRIAN_VIOLATION = "pedestrian_violation"
    SCHOOL_ZONE = "school_zone"
    PRIVACY_VIOLATION = "privacy_violation"
    DATA_RETENTION = "data_retention"

class SeverityLevel(Enum):
    INFO = "정보"
    WARNING = "경고" 
    VIOLATION = "위반"
    CRITICAL = "심각"

@dataclass
class TrafficLawViolation:
    violation_id: str
    vehicle_id: str
    violation_type: ViolationType
    severity: SeverityLevel
    location: Dict[str, float]
    timestamp: datetime
    description: str
    legal_basis: str
    penalty_points: int
    fine_amount: int  # 원
    evidence: Dict[str, Any]
    resolution_required: bool

@dataclass
class SpeedLimit:
    road_type: str
    speed_limit: int  # km/h
    school_zone_limit: int  # km/h
    weather_reduction: int  # 악천후시 감속
    night_reduction: int  # 야간 감속

@dataclass
class KoreanTrafficRules:
    speed_limits: Dict[str, SpeedLimit]
    following_distances: Dict[int, float]  # 속도별 안전거리 (m)
    school_zone_hours: Tuple[int, int]  # (시작시간, 종료시간)
    data_retention_days: int
    privacy_blur_distance: float  # 개인정보 블러 처리 거리 (m)

class KoreanTrafficLawChecker:
    """한국 교통법규 준수 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules = self._initialize_korean_rules()
        self.violation_history: List[TrafficLawViolation] = []
        
    def _initialize_korean_rules(self) -> KoreanTrafficRules:
        """한국 교통법규 초기화"""
        speed_limits = {
            "고속도로": SpeedLimit("고속도로", 100, 30, 20, 0),
            "자동차전용도로": SpeedLimit("자동차전용도로", 80, 30, 20, 0),
            "일반도로": SpeedLimit("일반도로", 60, 30, 10, 0), 
            "주거지역": SpeedLimit("주거지역", 50, 30, 10, 0),
            "어린이보호구역": SpeedLimit("어린이보호구역", 30, 30, 10, 0),
            "실버구역": SpeedLimit("실버구역", 30, 30, 10, 0)
        }
        
        # 속도별 안전거리 (도로교통법 제19조)
        following_distances = {
            30: 9,   # 30km/h -> 9m
            40: 16,  # 40km/h -> 16m  
            50: 25,  # 50km/h -> 25m
            60: 36,  # 60km/h -> 36m
            70: 49,  # 70km/h -> 49m
            80: 64,  # 80km/h -> 64m
            90: 81,  # 90km/h -> 81m
            100: 100 # 100km/h -> 100m
        }
        
        return KoreanTrafficRules(
            speed_limits=speed_limits,
            following_distances=following_distances,
            school_zone_hours=(8, 20),  # 08:00-20:00
            data_retention_days=30,     # 개인정보보호법
            privacy_blur_distance=50.0   # 50m 이내 개인정보 블러
        )
    
    async def check_speed_compliance(self, vehicle_data: Dict[str, Any]) -> Optional[TrafficLawViolation]:
        """속도 위반 검사"""
        current_speed = vehicle_data.get('speed', 0)  # km/h
        road_type = vehicle_data.get('road_type', '일반도로')
        location = vehicle_data.get('location', {})
        weather_condition = vehicle_data.get('weather', '맑음')
        is_school_zone = vehicle_data.get('is_school_zone', False)
        current_time = datetime.now()
        
        # 제한속도 결정
        speed_rule = self.rules.speed_limits.get(road_type, self.rules.speed_limits['일반도로'])
        
        if is_school_zone:
            limit = speed_rule.school_zone_limit
            # 어린이보호구역 시간대 확인 (08:00-20:00)
            current_hour = current_time.hour
            if self.rules.school_zone_hours[0] <= current_hour <= self.rules.school_zone_hours[1]:
                legal_basis = "도로교통법 제12조 (어린이보호구역)"
            else:
                limit = speed_rule.speed_limit
                legal_basis = f"도로교통법 제17조 ({road_type})"
        else:
            limit = speed_rule.speed_limit
            legal_basis = f"도로교통법 제17조 ({road_type})"
        
        # 악천후 감속 적용
        if weather_condition in ['비', '눈', '안개', '빙판']:
            limit -= speed_rule.weather_reduction
        
        # 위반 확인
        speed_excess = current_speed - limit
        if speed_excess > 0:
            # 벌점 및 벌금 계산 (도로교통법 시행령)
            if speed_excess <= 20:
                penalty_points = 15
                fine_amount = 60000
                severity = SeverityLevel.WARNING
            elif speed_excess <= 40:
                penalty_points = 30
                fine_amount = 90000
                severity = SeverityLevel.VIOLATION
            elif speed_excess <= 60:
                penalty_points = 60
                fine_amount = 120000
                severity = SeverityLevel.CRITICAL
            else:
                penalty_points = 120
                fine_amount = 200000
                severity = SeverityLevel.CRITICAL
            
            violation = TrafficLawViolation(
                violation_id=f"SPEED_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                vehicle_id=vehicle_data.get('vehicle_id', 'UNKNOWN'),
                violation_type=ViolationType.SPEED_VIOLATION,
                severity=severity,
                location=location,
                timestamp=current_time,
                description=f"제한속도 {limit}km/h 구간에서 {current_speed}km/h 주행 ({speed_excess}km/h 초과)",
                legal_basis=legal_basis,
                penalty_points=penalty_points,
                fine_amount=fine_amount,
                evidence={
                    'measured_speed': current_speed,
                    'speed_limit': limit,
                    'road_type': road_type,
                    'weather_condition': weather_condition,
                    'is_school_zone': is_school_zone,
                    'measurement_accuracy': '±3%'
                },
                resolution_required=True
            )
            
            self.logger.warning(f"속도위반 감지: {violation.description}")
            return violation
        
        return None
    
    async def check_following_distance(self, vehicle_data: Dict[str, Any]) -> Optional[TrafficLawViolation]:
        """안전거리 위반 검사"""
        current_speed = vehicle_data.get('speed', 0)
        following_distance = vehicle_data.get('following_distance', float('inf'))
        location = vehicle_data.get('location', {})
        
        # 속도에 따른 최소 안전거리 계산
        required_distance = self._calculate_safe_following_distance(current_speed)
        
        if following_distance < required_distance:
            shortage = required_distance - following_distance
            
            violation = TrafficLawViolation(
                violation_id=f"FOLLOW_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                vehicle_id=vehicle_data.get('vehicle_id', 'UNKNOWN'),
                violation_type=ViolationType.FOLLOWING_DISTANCE,
                severity=SeverityLevel.VIOLATION if shortage > 10 else SeverityLevel.WARNING,
                location=location,
                timestamp=datetime.now(),
                description=f"안전거리 부족: {following_distance:.1f}m (최소 {required_distance:.1f}m 필요)",
                legal_basis="도로교통법 제19조 (안전거리 확보)",
                penalty_points=10,
                fine_amount=40000,
                evidence={
                    'measured_distance': following_distance,
                    'required_distance': required_distance,
                    'current_speed': current_speed,
                    'shortage': shortage
                },
                resolution_required=True
            )
            
            self.logger.warning(f"안전거리 위반: {violation.description}")
            return violation
        
        return None
    
    def _calculate_safe_following_distance(self, speed_kmh: int) -> float:
        """속도에 따른 안전거리 계산"""
        # 가장 가까운 속도 구간 찾기
        speed_ranges = sorted(self.rules.following_distances.keys())
        
        for speed in speed_ranges:
            if speed_kmh <= speed:
                return self.rules.following_distances[speed]
        
        # 최고속도 초과시 공식 적용: 속도(km/h) / 10 * 3 (m)
        return max(speed_kmh / 10 * 3, self.rules.following_distances[100])
    
    async def check_lane_compliance(self, vehicle_data: Dict[str, Any]) -> Optional[TrafficLawViolation]:
        """차선 준수 검사"""
        lane_changes = vehicle_data.get('recent_lane_changes', [])
        current_lane = vehicle_data.get('current_lane', 'center')
        road_markings = vehicle_data.get('road_markings', {})
        location = vehicle_data.get('location', {})
        
        # 실선 침범 검사
        if road_markings.get('marking_type') == '실선' and lane_changes:
            last_change = lane_changes[-1]
            if datetime.now() - datetime.fromisoformat(last_change['timestamp']) < timedelta(seconds=5):
                violation = TrafficLawViolation(
                    violation_id=f"LANE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    vehicle_id=vehicle_data.get('vehicle_id', 'UNKNOWN'),
                    violation_type=ViolationType.LANE_VIOLATION,
                    severity=SeverityLevel.VIOLATION,
                    location=location,
                    timestamp=datetime.now(),
                    description="실선 구간 차선 변경",
                    legal_basis="도로교통법 제15조 (차로의 통행)",
                    penalty_points=10,
                    fine_amount=30000,
                    evidence={
                        'marking_type': '실선',
                        'lane_change_time': last_change['timestamp'],
                        'from_lane': last_change['from'],
                        'to_lane': last_change['to']
                    },
                    resolution_required=True
                )
                
                self.logger.warning(f"차선위반: {violation.description}")
                return violation
        
        return None
    
    async def check_privacy_compliance(self, sensor_data: Dict[str, Any]) -> List[TrafficLawViolation]:
        """개인정보보호법 준수 검사"""
        violations = []
        camera_data = sensor_data.get('cameras', [])
        data_storage = sensor_data.get('data_storage', {})
        
        # 데이터 보존기간 검사
        stored_data = data_storage.get('personal_data', [])
        for data_item in stored_data:
            storage_date = datetime.fromisoformat(data_item['stored_at'])
            days_stored = (datetime.now() - storage_date).days
            
            if days_stored > self.rules.data_retention_days:
                violation = TrafficLawViolation(
                    violation_id=f"PRIVACY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    vehicle_id=sensor_data.get('vehicle_id', 'UNKNOWN'),
                    violation_type=ViolationType.DATA_RETENTION,
                    severity=SeverityLevel.CRITICAL,
                    location=sensor_data.get('location', {}),
                    timestamp=datetime.now(),
                    description=f"개인정보 보존기간 초과: {days_stored}일 (최대 {self.rules.data_retention_days}일)",
                    legal_basis="개인정보보호법 제21조 (개인정보의 파기)",
                    penalty_points=0,
                    fine_amount=0,  # 행정처분
                    evidence={
                        'data_type': data_item['type'],
                        'stored_date': data_item['stored_at'],
                        'retention_period': self.rules.data_retention_days,
                        'excess_days': days_stored - self.rules.data_retention_days
                    },
                    resolution_required=True
                )
                violations.append(violation)
        
        # 개인정보 블러 처리 검사
        for camera in camera_data:
            detected_faces = camera.get('detected_faces', [])
            for face in detected_faces:
                distance = face.get('distance_to_vehicle', 0)
                is_blurred = face.get('is_blurred', False)
                
                if distance <= self.rules.privacy_blur_distance and not is_blurred:
                    violation = TrafficLawViolation(
                        violation_id=f"BLUR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        vehicle_id=sensor_data.get('vehicle_id', 'UNKNOWN'),
                        violation_type=ViolationType.PRIVACY_VIOLATION,
                        severity=SeverityLevel.WARNING,
                        location=sensor_data.get('location', {}),
                        timestamp=datetime.now(),
                        description=f"개인정보 블러 처리 누락 (거리: {distance:.1f}m)",
                        legal_basis="개인정보보호법 제15조 (개인정보의 수집·이용)",
                        penalty_points=0,
                        fine_amount=0,
                        evidence={
                            'face_distance': distance,
                            'blur_threshold': self.rules.privacy_blur_distance,
                            'camera_id': camera['id']
                        },
                        resolution_required=True
                    )
                    violations.append(violation)
        
        return violations
    
    async def check_comprehensive_compliance(self, vehicle_data: Dict[str, Any]) -> Dict[str, Any]:
        """종합 법규 준수 검사"""
        violations = []
        
        # 모든 검사 병렬 실행
        checks = [
            self.check_speed_compliance(vehicle_data),
            self.check_following_distance(vehicle_data), 
            self.check_lane_compliance(vehicle_data),
            self.check_privacy_compliance(vehicle_data.get('sensors', {}))
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        # 결과 처리
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"검사 {i} 실패: {result}")
                continue
                
            if isinstance(result, list):
                violations.extend(result)
            elif result is not None:
                violations.append(result)
        
        # 위반사항 저장
        self.violation_history.extend(violations)
        
        # 종합 평가
        critical_count = sum(1 for v in violations if v.severity == SeverityLevel.CRITICAL)
        violation_count = sum(1 for v in violations if v.severity == SeverityLevel.VIOLATION)
        warning_count = sum(1 for v in violations if v.severity == SeverityLevel.WARNING)
        
        compliance_score = max(0, 100 - (critical_count * 30 + violation_count * 15 + warning_count * 5))
        
        compliance_report = {
            'timestamp': datetime.now().isoformat(),
            'vehicle_id': vehicle_data.get('vehicle_id', 'UNKNOWN'),
            'compliance_score': compliance_score,
            'violations': [self._violation_to_dict(v) for v in violations],
            'summary': {
                'total_violations': len(violations),
                'critical': critical_count,
                'violations': violation_count, 
                'warnings': warning_count,
                'actions_required': sum(1 for v in violations if v.resolution_required)
            },
            'recommendations': self._generate_recommendations(violations)
        }
        
        self.logger.info(f"법규 준수 검사 완료 - 점수: {compliance_score}/100, 위반: {len(violations)}건")
        return compliance_report
    
    def _violation_to_dict(self, violation: TrafficLawViolation) -> Dict[str, Any]:
        """위반사항을 딕셔너리로 변환"""
        return {
            'violation_id': violation.violation_id,
            'type': violation.violation_type.value,
            'severity': violation.severity.value,
            'timestamp': violation.timestamp.isoformat(),
            'description': violation.description,
            'legal_basis': violation.legal_basis,
            'penalty_points': violation.penalty_points,
            'fine_amount': violation.fine_amount,
            'evidence': violation.evidence,
            'resolution_required': violation.resolution_required
        }
    
    def _generate_recommendations(self, violations: List[TrafficLawViolation]) -> List[str]:
        """위반사항 기반 개선 권고사항 생성"""
        recommendations = []
        
        violation_types = [v.violation_type for v in violations]
        
        if ViolationType.SPEED_VIOLATION in violation_types:
            recommendations.append("속도 제한 알림 시스템 강화 필요")
            recommendations.append("어린이보호구역 진입시 자동 감속 기능 활성화")
        
        if ViolationType.FOLLOWING_DISTANCE in violation_types:
            recommendations.append("적응형 순항 제어(ACC) 시스템 점검")
            recommendations.append("전방 충돌 경고 시스템 민감도 조정")
        
        if ViolationType.LANE_VIOLATION in violation_types:
            recommendations.append("차선 이탈 방지 시스템(LDWS) 재보정")
            recommendations.append("실선/점선 구분 인식 알고리즘 개선")
        
        if ViolationType.PRIVACY_VIOLATION in violation_types:
            recommendations.append("개인정보 자동 블러 처리 시스템 점검")
            recommendations.append("데이터 자동 삭제 정책 적용")
        
        return recommendations
    
    def get_violation_statistics(self, days: int = 30) -> Dict[str, Any]:
        """최근 위반 통계"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_violations = [v for v in self.violation_history if v.timestamp >= cutoff_date]
        
        stats = {
            'period_days': days,
            'total_violations': len(recent_violations),
            'by_type': {},
            'by_severity': {},
            'total_fines': 0,
            'total_penalty_points': 0
        }
        
        for violation in recent_violations:
            # 유형별 통계
            v_type = violation.violation_type.value
            stats['by_type'][v_type] = stats['by_type'].get(v_type, 0) + 1
            
            # 심각도별 통계
            severity = violation.severity.value
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
            
            # 벌금/벌점 합계
            stats['total_fines'] += violation.fine_amount
            stats['total_penalty_points'] += violation.penalty_points
        
        return stats

async def main():
    """테스트 및 데모"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    checker = KoreanTrafficLawChecker()
    
    # 테스트 차량 데이터
    test_vehicle_data = {
        'vehicle_id': 'A2Z_BUS_001',
        'speed': 45,  # 어린이보호구역에서 45km/h
        'road_type': '어린이보호구역',
        'is_school_zone': True,
        'weather': '맑음',
        'following_distance': 8.0,  # 부족한 안전거리
        'location': {'lat': 37.5665, 'lng': 126.9780},
        'recent_lane_changes': [],
        'road_markings': {'marking_type': '점선'},
        'sensors': {
            'vehicle_id': 'A2Z_BUS_001',
            'location': {'lat': 37.5665, 'lng': 126.9780},
            'cameras': [{
                'id': 'front_cam_1',
                'detected_faces': [{
                    'distance_to_vehicle': 30.0,  # 블러 처리 필요
                    'is_blurred': False
                }]
            }],
            'data_storage': {
                'personal_data': [{
                    'type': 'facial_recognition',
                    'stored_at': (datetime.now() - timedelta(days=35)).isoformat()  # 보존기간 초과
                }]
            }
        }
    }
    
    # 종합 검사
    report = await checker.check_comprehensive_compliance(test_vehicle_data)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    
    # 통계 조회
    stats = checker.get_violation_statistics(30)
    print(f"\n=== 최근 30일 위반 통계 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())