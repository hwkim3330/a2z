# A2Z 자율주행차 네트워크 요구도 정의

## 프로젝트 개요
A2Z 자율주행차의 고장 시나리오 기반 네트워크 요구도 정의 및 TSN(Time-Sensitive Networking) 구현 방안

## 시스템 구성

### 1. 센서 구성
| 센서 | 모델명 | 주요 사양 | 인터페이스 | 대역폭 |
|------|--------|-----------|------------|---------|
| **LiDAR** | Hesai Pandar40P | 40채널, 360° 회전형, 200m 범위 | 100BASE-TX | 18.84-37.68 Mbps |
| **LiDAR** | Autol G32 | 32채널, 솔리드스테이트, 250m 범위 | 1000Base-T1 | 122.9-405.5 Mbps |
| **Radar** | MRR-35 | 중거리 레이더, 76-77 GHz | CAN-FD/Ethernet | 1-5 Mbps |
| **Camera** | STURDeCAM31 (추정) | 1920x1536, 60°/110° 화각 | MJPEG/Ethernet | 44.2-353.9 Mbps |

### 2. ECU/모듈 구성
| 약어 | 전체 이름 | 역할 | 플랫폼 |
|------|-----------|------|--------|
| ACU_NO | Autonomous Control Unit (Nvidia) | 센서 데이터 융합, 경로 계획 | NVIDIA Jetson Orin |
| ACU_IT | Autonomous Control Unit (Intel) | 자율주행 제어 | Intel Tiger Lake |
| TCU | Telematics Control Unit | 차량-클라우드 통신, OTA | - |
| EDR | Event Data Recorder | 이벤트 데이터 기록 | - |
| DSSAD | Data Storage System for Automated Driving | 자율주행 데이터 저장 | - |
| CMU | Connectivity Management Unit | 네트워크 연결 관리 | - |
| VCU | Vehicle Control Unit | 차량 제어 | - |

### 3. 네트워크 구성
- **TSN 스위치**: Microchip Automotive Ethernet Switch
  - IEEE 802.1Qbv (TAS), CBS, FRER 지원
  - Multi-GigE (1G/2.5G/10G) 포트
  - VLAN, QoS, PSFP 지원

## 네트워크 아키텍처 옵션

### 옵션 A: 단일 스위치 구성
- 구조: 센서 → TSN 스위치 → ACU_IT
- 장점: 최소 홉(1-2), 최소 지연
- 단점: 단일 장애점(SPOF)

### 옵션 B: 듀얼 엣지 + 중앙 집선
- 구조: 좌/우 엣지 스위치 + 중앙 집선 → ACU_IT
- 장점: 포트 수 유연, 케이블 최적화
- 단점: 홉 증가(2-3)

### 옵션 C: 3스위치 메시 + ACU 듀얼홈
- 구조: 상단 2 + 중간 1 메시 구성, ACU 이중경로
- 장점: FRER 완전 지원, 고가용성
- 단점: 구성 복잡도 증가

## 고장 시나리오 및 대응

### 코드 체계
`[SCN]-[LOC]-[DEV]-[FLT] (+ [RES])`

#### SCN (시나리오 대분류)
- A: 손실 (링크단선, 포워딩오류, 오버런)
- B: 과도 지연/지터
- C: 순서/복제 문제
- D: 프레임 손상
- E: 동기 문제
- F: 장비/스위치 고장
- G: 전원/환경 문제
- H: 구성/펌웨어 문제

#### LOC (위치)
- ZF: Front, ZB: Rear, ZL: Left, ZR: Right

#### DEV (장비)
- 센서: LP40, LG32, MR35, CAMV
- 스위치: SWF, SWB, SWL, SWR
- ECU: ACI, ACN, TCU, EDR, DSS, CMU, VCU

### 해결 방안(RES)
| 코드 | 의미 | 적용 시나리오 |
|------|------|---------------|
| FRR | FRER 기반 무중단 복구 | 경로 단절 시 대체 경로 활용 |
| RER | 즉시 경로 전환 | LAG/ECMP/듀얼업링크 |
| CBS+ | CBS/우선순위 재설정 | Q7 지연 한도 확보 |
| PSFP | 스트림 폴리싱/차단 | 폭주 트래픽 억제 |
| PTP | 동기 복구 | 타임 필터링/GM 전환 |
| RST | 재시작/재부팅 | 단독 장애 도메인 |
| DGR | 성능저하모드 | 안전기능 유지 |
| ISO | 격리/바이패스 | 불량 포트/장비 격리 |
| RBK | 롤백 | 안정버전 복귀 |
| SSC | 안전정지 | 해결불가 시 |
| ALM | 경보/로그 | 모든 이상 기록 |

## 정량적 요구사항

### 성능 목표
- **지연**: Q7 E2E ≤ 10 ms (P99.999)
- **손실률**: ≤ 1e-7
- **Failover**: ≤ 1 프레임 주기
- **장애검출(FDI)**: L2/L3 ≤ 50 ms, 장비 Hang ≤ 200 ms

### TSN 설정
- **PCP → 큐 매핑**
  - 안전 스트림(LiDAR/Radar): PCP 6-7 → Q7
  - 카메라: PCP 5 → Q6
- **CBS 프로파일**
  - Q7: 25-30%
  - Q6: 10-15%
  - 기타: ≤ 10%
- **FRER 윈도**: 초기 64 프레임

## 대역폭 계산

### LiDAR G32 (추정)
| FPS | 1-echo | 2-echo | 3-echo |
|-----|--------|--------|--------|
| 10 | 54.1 Mbps | 108.1 Mbps | 162.2 Mbps |
| 20 | 108.1 Mbps | 216.3 Mbps | 324.4 Mbps |
| 25 | 135.2 Mbps | 270.3 Mbps | 405.5 Mbps |

### Camera (MJPEG)
| FPS | 0.5 bpp | 0.75 bpp | 1.0 bpp | 1.5 bpp | 2.0 bpp |
|-----|---------|----------|---------|---------|---------|
| 30 | 44.2 Mbps | 66.4 Mbps | 88.5 Mbps | 132.7 Mbps | 176.9 Mbps |
| 60 | 88.5 Mbps | 132.7 Mbps | 176.9 Mbps | 265.4 Mbps | 353.9 Mbps |

## 참고 링크
- [A2Z 자율주행 기술](https://autoa2z.co.kr/Coretech)
- [관련 YouTube 영상](https://www.youtube.com/watch?v=kRHf5jUnVAg)

## 문서
- [시나리오 상세](docs/scenarios.md)
- [센서 스펙 상세](docs/sensors.md)
- [네트워크 구성도](docs/network.md)
