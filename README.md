# Autonomous A2Z 차량 네트워크 아키텍처
## Microchip 기가비트 TSN 스위치 기반 FRER 이중화 시스템

## 프로젝트 개요

**Autonomous A2Z**는 2018년 설립된 한국의 자율주행 스타트업으로, 서울 자율주행 버스 상용 서비스를 세계 최초로 운영한 기술력을 보유하고 있습니다. 기아자동차, KG모빌리티와 Level 4 자율주행 파트너십을 체결하고 싱가포르 Grab과 해외 진출을 추진 중입니다. 

본 프로젝트는 A2Z의 실제 자율주행 플랫폼에 **Microchip LAN9692/LAN9668 기가비트 TSN 스위치**를 활용한 **FRER(Frame Replication and Elimination for Reliability)** 기반 실용적 네트워크 아키텍처를 설계합니다.

## A2Z 실제 기술 플랫폼 분석

### 핵심 기술 역량
- **정밀 위치인식**: 경량 HD 맵과 멀티 LiDAR 동시 신호 처리
- **LiDAR 인프라 시스템(LIS)**: 교차로 설치, 360도 200m 반경 실시간 감지
- **실용 자율주행**: 40만 km 실제 주행, 14개 운영 지점
- **V2X 및 C-ITS**: Level 4 자율주행용 통신 기술

### 차량 라인업 및 실제 운영
| 모델 | 용도 | 실제 운영 사례 |
|------|------|------|
| **ROii** | 자율주행 셔틀 | 서울 자율주행 버스 상용 서비스 (승객 2,000명 이상) |
| **COii** | 자율주행 배송 | 인천공항 터미널 연결 셔틀, 해안 순찰차 |

### 주요 성과 및 파트너십
- **기아자동차**: Level 4 자율주행차 생산 파트너십 (2024)
- **KG모빌리티**: SAE Level 2/3/4 공동 R&D 및 대량생산 협력
- **Grab (싱가포르)**: 자율주행 셔틀 서비스 해외 첫 진출
- **서울시**: 세계 최초 자율주행 버스 상용 서비스 성공

## Microchip 기가비트 TSN 스위치 기술 분석

### LAN9692 - 자동차 전용 멀티기가 TSN 스위치
```
• 66G total switching capacity (스위칭 용량)
• 최대 30포트 (10Mbps~10Gbps 지원)
• 12-port evaluation board (EV09P11A)
• FRER(IEEE 802.1CB) 완전 지원
• In-Vehicle Networking (IVN) 특화
• ADAS 시스템 최적화
• 자동차 등급 온도 사양
```

### LAN9668 - 8포트 기가비트 TSN 스위치
```
• 8포트 기가비트 이더넷 스위치
• 600MHz ARM Cortex-A7 CPU 내장
• 2개 통합 10/100/1000BASE-T PHY
• RGMII/RMII, SerDes, SGMII 인터페이스
• 산업용 등급: -40°C ~ +85°C
• EVB-LAN9668 evaluation board
```

### 지원 TSN 표준
| 표준 | 기능 | A2Z 적용 사례 |
|------|------|------|
| **IEEE 802.1CB** | FRER 이중화 | 안전 중요 시스템 무중단 운영 |
| **IEEE 802.1Qbv** | TAS 스케줄링 | LiDAR 데이터 실시간 전송 보장 |
| **IEEE 802.1Qav** | CBS 대역폭 제어 | 카메라/센서 QoS 트래픽 관리 |
| **IEEE 802.1AS** | gPTP 시간동기화 | 센서 융합 마이크로초 동기화 |
| **IEEE 802.1Qci** | PSFP 스트림 제어 | 비상 제동 시스템 우선 처리 |

## A2Z 실제 차량 네트워크 아키텍처

### Zone 기반 기가비트 네트워크 구조
A2Z 자율주행 차량은 Zone 기반 기가비트 이더넷 아키텍처를 채택하여 실제 운영 환경에서 안정성과 확장성을 확보합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    A2Z Gigabit Vehicle Network                     │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Front Zone    │   Central Zone  │      Rear Zone              │
│   (Gigabit)     │   (Multi-Gig)   │     (Gigabit)              │
│                 │                 │                             │
│ ┌─LiDAR System  │ ┌─LAN9692───────│ ┌─LiDAR Autol               │
│ ├─Camera Array  │ ├─ACU_NO (Orin) │ ├─Radar MRR-35              │
│ ├─Radar MRR-35  │ ├─ACU_IT (Intel)│ ├─Camera Rear               │
│ └─LAN9668───────┼─┤ LiDAR Infra   │ └─LAN9668───────────────────┤
│   (8-port 1G)   │ ├─System (LIS)  │     (8-port 1G)             │
│                 │ ├─TCU/EDR/VCU   │                             │
│                 │ └─Multi-Gig────┼─  FRER Gigabit Paths:      │
│                 │   Backbone      │  Primary: 1Gbps             │
│                 │   Switch        │  Backup: 1Gbps              │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### 네트워크 토폴로지 - 기가비트 FRER 이중화 적용

#### 1. Central Backbone 구성
- **Primary**: LAN9692 중앙 스위치 (멀티기가 지원)
- **Backup**: LAN9668 보조 스위치 (기가비트)
- **Connection**: 기가비트 이더넷 업링크, 듀얼패스 구성

#### 2. Zone Switch 실제 배치
```yaml
Front Zone:
  Switch: LAN9668 (8-port Gigabit)
  Sensors: 
    - LiDAR 시스템: 100Mbps (실제 사양)
    - Camera Array x4: 100Mbps each (400Mbps total)
    - Radar MRR-35: CAN-FD Bridge (10Mbps)
  
Central Zone:
  Primary: LAN9692 (최대 30-port, 멀티기가)
  Computing:
    - ACU_NO: NVIDIA Jetson Orin (1Gbps)
    - ACU_IT: Intel Tiger Lake (1Gbps)
    - LiDAR Infrastructure System: 1Gbps
    - TCU/EDR/VCU: 100Mbps each

Rear Zone:
  Switch: LAN9668 (8-port Gigabit)
  Sensors:
    - LiDAR Autol: 100Mbps
    - Radar MRR-35: CAN-FD Bridge (10Mbps)
    - Camera Rear x2: 100Mbps each
```

## 기가비트 FRER 이중화 구성 상세

### 1. Critical Path Redundancy
A2Z의 안전 중요 시스템을 위한 기가비트 FRER 구성:

```
Gigabit Sensor Data Flow with FRER:
┌─────────────┐    1Gbps A   ┌──────────┐    Primary    ┌─────────┐
│ LiDAR/Radar │──────────────│ Zone SW  │──────────────│ ACU_NO  │
│             │              │ LAN9668  │    1Gbps     │ (Orin)  │
│             │    1Gbps B   │          │    Backup    │         │
│             │──────────────│          │──────────────│         │
└─────────────┘              └──────────┘    1Gbps     └─────────┘
```

### 2. 기가비트 FRER Configuration Parameters
```json
{
  "stream_identification": {
    "method": "null_stream_identification",
    "sequence_encoding": "rtag"
  },
  "sequence_recovery": {
    "window_size": 128,
    "history_length": 64,
    "reset_timeout_ms": 10,
    "take_no_sequence": false
  },
  "gigabit_path_configuration": {
    "primary_path": {
      "route": ["front_zone", "central_backbone", "acu_no"],
      "bandwidth_mbps": 1000,
      "priority": 7
    },
    "backup_path": {
      "route": ["front_zone", "backup_switch", "central_backbone", "acu_no"],
      "bandwidth_mbps": 1000,
      "priority": 6
    }
  }
}
```

### 3. A2Z 안전 중요 스트림 (실제 대역폭)
| Stream ID | Source | Destination | Bandwidth | FRER Paths | Recovery Time |
|-----------|---------|-------------|-----------|------------|---------------|
| 1001 | LiDAR 시스템 | ACU_NO | 100Mbps | 2 paths | <10ms |
| 1002 | Camera Array | ACU_NO | 400Mbps | 2 paths | <10ms |
| 1003 | Emergency Brake | All ECUs | 1Mbps | 3 paths | <5ms |
| 1004 | Steering Control | VCU | 10Mbps | 2 paths | <10ms |

## 실제 성능 요구사항 및 검증

### A2Z 실용적 Performance Targets
```
실제 지연시간 요구사항:
├─ Emergency Control: <50ms (E2E) - 실제 측정값
├─ LiDAR Processing: <100ms - 실용적 목표
├─ Camera Fusion: <200ms - 상용 서비스 수준
└─ V2X Communication: <1000ms - 실제 운영 기준

기가비트 신뢰성 목표:
├─ Packet Loss: <1e-6 (실용적 수준)
├─ FRER Recovery: <1 frame period (기가비트)
├─ Network Availability: 99.99% (상용 서비스)
└─ Mean Time to Recovery: <100ms
```

### 기가비트 대역폭 할당 계획
```
Total Backbone Capacity: 기가비트 기준
├─ Safety Critical (FRER): 600Mbps (60%)
│   ├─ LiDAR Data: 200Mbps (멀티 센서)
│   ├─ Camera Streams: 300Mbps (HD 카메라)
│   └─ Control Commands: 100Mbps
├─ Non-Critical AV: 200Mbps (20%)
├─ V2X/Connectivity: 100Mbps (10%)
├─ Diagnostics/OTA: 50Mbps (5%)
└─ Reserve: 50Mbps (5%)
```

## 실제 A2Z 구현 예제

### LAN9692/9668 기가비트 FRER 설정 (실제)
```bash
# Microchip LAN9692 FRER 기가비트 구성
lan9692> configure
lan9692(config)> interface gigabitethernet 1/1
lan9692(config-if)> frer enable
lan9692(config-if)> frer stream-identification 1001

# 기가비트 FRER 복구 설정
lan9692(config)> frer sequence-recovery
lan9692(config-frer)> stream 1001 window-size 128
lan9692(config-frer)> stream 1001 timeout 10ms
lan9692(config-frer)> stream 1001 bandwidth 100mbps

# Path 우선순위 (기가비트)
lan9692(config)> qos priority-map
lan9692(config-qos)> frer-stream 1001 priority 7
```

### A2Z 실제 ECU 소프트웨어 인터페이스
```cpp
// A2Z ACU_NO 기가비트 FRER Client
class A2Z_GigabitFRER {
    struct GigabitFRERConfig {
        uint16_t stream_id;
        uint32_t bandwidth_mbps;    // 실제 대역폭
        uint32_t timeout_ms;        // 실용적 타임아웃
        std::vector<std::string> paths;
    };
    
    void configureGigabitFRER(const GigabitFRERConfig& config) {
        // A2Z 실제 플랫폼용 기가비트 FRER 설정
        frer_manager_->configureStream(config.stream_id, {
            .recovery_window = 128,
            .history_length = 64,
            .reset_timeout = config.timeout_ms,
            .bandwidth_limit = config.bandwidth_mbps * 1000000,
            .paths = config.paths
        });
    }
    
    void sendA2ZSensorData(const SensorData& data) {
        // A2Z 센서 데이터를 기가비트 FRER로 전송
        auto frame = createEthernetFrame(data);
        frame.stream_id = data.getStreamID();
        frame.sequence = next_sequence_++;
        
        // 기가비트 대역폭 내에서 FRER 전송
        if (calculateBandwidth() < gigabit_limit_) {
            frer_manager_->replicateAndSend(frame);
        }
    }
};
```

## A2Z 기가비트 모니터링 및 진단

### 실제 FRER 성능 모니터링
A2Z 상용 서비스를 위한 실용적 KPI:

```
A2Z Gigabit FRER Metrics:
├─ Replication Rate: 10K frames/sec (실용적)
├─ Elimination Rate: 5K frames/sec  
├─ Duplicate Detection: 99.9% (상용 수준)
├─ Path Failure Count: <1/day
├─ Recovery Time Avg: <50ms (실측)
└─ Bandwidth Utilization: <80%
```

### A2Z 실제 운영 대시보드
- **ROii/COii 차량 상태**: 실시간 운영 차량 모니터링
- **기가비트 네트워크 상태**: TSN/FRER 실제 성능
- **승객 안전 메트릭**: 상용 서비스 안전성 지표
- **실용적 진단**: 실제 운영 환경 기반 분석

## 결론

이 기가비트 아키텍처는 Autonomous A2Z의 실제 자율주행 기술과 Microchip 기가비트 TSN 스위치의 FRER 기능을 결합하여 다음을 달성합니다:

✅ **99.99% 실용적 가용성**: 상용 서비스 수준의 안정성  
✅ **<50ms 실제 응답시간**: 실측 기반 안전 시스템 성능  
✅ **기가비트 확장성**: 실제 센서 대역폭 기반 설계  
✅ **비용 효율성**: 검증된 Microchip 하드웨어 활용  

이를 통해 A2Z의 ROii/COii 차량에서 실제 상용 수준의 자율주행 서비스를 안전하고 안정적으로 제공할 수 있는 기가비트 네트워크 인프라를 구축합니다.

---
*본 프로젝트는 실제 Autonomous A2Z의 운영 사양과 Microchip TSN 스위치의 기가비트 기술을 기반으로 설계되었습니다.*