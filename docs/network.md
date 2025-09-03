# 네트워크 구성 상세

## TSN 네트워크 아키텍처

### 네트워크 토폴로지 옵션

#### 옵션 A: 단일 스위치 구성
```
        [Sensors]
            |
     [TSN Switch]
            |
        [ACU_IT]
```

**특징**
- 최소 홉 수: 1-2
- 최소 지연 구성
- 단순한 구조

**장점**
- 구현 간단
- 지연 시간 최소화
- 비용 효율적

**단점**
- 단일 장애점(SPOF)
- FRER 경로 중복 불가
- 확장성 제한

**적용 시나리오**
- 데모/프로토타입
- 저비용 구현
- FRER 미적용 시스템

#### 옵션 B: 듀얼 엣지 + 중앙 집선
```
    [Left Sensors]    [Right Sensors]
          |                 |
    [Edge SW L]      [Edge SW R]
          \               /
           [Central SW]
                |
            [ACU_IT]
```

**특징**
- 홉 수: 2-3
- 포트 수 유연성
- 계층적 구조

**장점**
- 센서 케이블 최적화
- 포트 확장 용이
- 부분 이중화 가능

**단점**
- 중앙 집선 SPOF
- 홉 증가로 지연 증가

**보완 방안**
- 중앙 집선 업링크 2.5G/10G
- 집선 스위치 이중화

#### 옵션 C: 3스위치 메시 + ACU 듀얼홈
```
    [Front Sensors]    [Rear Sensors]
          |                 |
     [SW Front]        [SW Rear]
          \               /
           X             X
          /               \
    [SW Central]     [ACU Dual-homed]
```

**특징**
- 완전 메시 토폴로지
- ACU 듀얼 홈 연결
- FRER 완전 지원

**장점**
- 진정한 경로 이중화
- 높은 가용성
- FRER 효과 극대화

**단점**
- 구성 복잡도 높음
- 비용 증가
- 관리 오버헤드

**권장 사용**
- 고가용성 요구
- 안전 필수 시스템
- FRER 필수 구현

## TSN 기능 구성

### IEEE 802.1Qbv (TAS - Time Aware Shaper)
**게이트 제어 목록 설정**
```
시간 슬롯 | Q7 | Q6 | Q5 | Q4 | Q3 | Q2 | Q1 | Q0 |
---------|----|----|----|----|----|----|----|----|
0-2ms    | O  | C  | C  | C  | C  | C  | C  | C  |
2-4ms    | C  | O  | C  | C  | C  | C  | C  | C  |
4-6ms    | C  | C  | O  | O  | C  | C  | C  | C  |
6-10ms   | C  | C  | C  | C  | O  | O  | O  | O  |
```
O: Open, C: Closed

### IEEE 802.1Qav (CBS - Credit Based Shaper)
**큐별 대역폭 할당**
| 큐 | 용도 | IdleSlope | SendSlope | 대역폭 |
|----|------|-----------|-----------|---------|
| Q7 | 안전 중요 (LiDAR/Radar) | 300 Mbps | -700 Mbps | 30% |
| Q6 | 카메라 스트림 | 150 Mbps | -850 Mbps | 15% |
| Q5 | 제어 메시지 | 50 Mbps | -950 Mbps | 5% |
| Q4-Q0 | 기타/베스트에포트 | - | - | 50% |

### IEEE 802.1CB (FRER - Frame Replication and Elimination)
**FRER 스트림 구성**
```yaml
Stream_ID: 0x0001
  Source: Pandar40P_Front
  Destination: ACU_IT
  Paths:
    - Primary: SW_Front -> SW_Central -> ACU_IT
    - Secondary: SW_Front -> SW_Rear -> SW_Central -> ACU_IT
  Sequence_Window: 64
  History_Length: 128
  Recovery_Timeout: 100ms
```

### IEEE 802.1AS (gPTP - Generalized PTP)
**시간 동기화 계층**
```
    [GPS Grand Master]
           |
      [SW_Central]
       /    |    \
   [SW_F] [SW_R] [SW_L]
     |      |      |
  [Sensors] ... [ECUs]
```

**동기화 요구사항**
- Grand Master 정확도: ±100 ns
- 스위치 간 동기: ±500 ns
- 엔드포인트 동기: ±1 μs

## VLAN 및 우선순위 매핑

### VLAN 구성
| VLAN ID | 이름 | 용도 | 장비 |
|---------|------|------|------|
| 10 | SENSOR_CRITICAL | 안전 중요 센서 | LiDAR, Radar |
| 20 | SENSOR_VIDEO | 비디오 스트림 | Camera |
| 30 | CONTROL | 제어 메시지 | ACU, VCU |
| 40 | TELEMETRY | 원격측정/진단 | TCU, EDR |
| 50 | MANAGEMENT | 관리 트래픽 | OAM |

### PCP 매핑
| PCP | 트래픽 클래스 | 큐 | DSCP | 용도 |
|-----|--------------|-----|------|------|
| 7 | Network Control | Q7 | CS7 | PTP, 네트워크 제어 |
| 6 | Critical Safety | Q7 | CS6 | LiDAR, Radar |
| 5 | Video | Q6 | AF41 | Camera 스트림 |
| 4 | Control | Q5 | AF31 | 제어 메시지 |
| 3 | Telemetry | Q4 | AF21 | 원격측정 |
| 2 | Management | Q3 | CS2 | OAM |
| 1 | Background | Q2 | CS1 | 로그, 업데이트 |
| 0 | Best Effort | Q1 | BE | 기본 트래픽 |

## 스위치 구성 예시

### Microchip LAN9662 설정
```bash
# 포트 설정
port 1-4 mode 1000baseT1
port 5-6 mode 2500baseT
port 7-8 mode sgmii

# VLAN 설정
vlan 10 name SENSOR_CRITICAL
vlan 10 port 1-4 untagged
vlan 10 port 5-6 tagged

# QoS 설정
qos port 1-4 default-pcp 6
qos pcp-queue-map 7:7 6:7 5:6 4:5 3:4 2:3 1:2 0:1

# CBS 설정
cbs queue 7 idleslope 300000
cbs queue 6 idleslope 150000

# FRER 설정
frer stream 1 sequence-recovery enable
frer stream 1 window-size 64
frer stream 1 member-port 5,6

# PTP 설정
ptp mode boundary-clock
ptp domain 0
ptp priority1 128
```

## 링크 예산 계산

### 존별 링크 대역폭 요구사항
| 링크 | 센서 트래픽 | FRER 오버헤드 | 관리 트래픽 | 여유율 | 총 요구 대역폭 |
|------|------------|---------------|------------|--------|---------------|
| Front Edge → Central | 600 Mbps | +100% (복제) | 10 Mbps | 30% | 1.43 Gbps |
| Rear Edge → Central | 200 Mbps | +100% (복제) | 10 Mbps | 30% | 546 Mbps |
| Left Edge → Central | 500 Mbps | +100% (복제) | 10 Mbps | 30% | 1.33 Gbps |
| Right Edge → Central | 500 Mbps | +100% (복제) | 10 Mbps | 30% | 1.33 Gbps |

**권장 링크 속도**
- Edge → Central: 2.5 Gbps
- Central → ACU: 10 Gbps (듀얼)
- Inter-switch mesh: 2.5 Gbps

## 모니터링 및 진단

### 주요 모니터링 지표
```yaml
Network KPIs:
  - Frame Loss Rate: < 1e-7
  - E2E Latency (P99.999): < 10ms
  - Jitter: < 1ms
  - FRER Recovery Rate: > 99.999%
  
Switch Metrics:
  - Queue Occupancy: < 80%
  - CBS Credit: > 0
  - TAS Gate Violations: 0
  - PTP Offset: < 1μs
  
Link Metrics:
  - Utilization: < 70%
  - CRC Errors: 0
  - Link Flaps: 0
  - PHY Errors: 0
```

### 로그 수집 항목
- FRER: framesRecovered, duplicatesDiscarded, lateArrival
- QoS: Q7 delay/drop 통계
- PTP: state, offset, drift
- 장애: fault_code, res_code, timestamp