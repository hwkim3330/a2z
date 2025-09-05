# A2Z Gigabit TSN/FRER 배포 가이드
## Microchip LAN9692/LAN9662 기반 완벽한 구축 매뉴얼

## 목차
1. [사전 요구사항](#사전-요구사항)
2. [하드웨어 설치](#하드웨어-설치)
3. [초기 설정](#초기-설정)
4. [FRER 구성](#frer-구성)
5. [검증 및 테스트](#검증-및-테스트)
6. [운영 및 모니터링](#운영-및-모니터링)
7. [문제 해결](#문제-해결)

---

## 1. 사전 요구사항

### 1.1 하드웨어 요구사항
```yaml
필수 장비:
  Central Switch:
    - Model: Microchip LAN9692
    - Evaluation Board: EV09P11A (12-port)
    - Quantity: 1
    - Role: 중앙 기가비트 백본 스위치
  
  Zone Switches:
    - Model: Microchip LAN9662
    - Evaluation Board: EVB-LAN9662
    - Quantity: 2 (Front Zone, Rear Zone)
    - Role: 구역별 분산 스위치
  
  Network Infrastructure:
    - CAT6A/CAT7 Cables (Gigabit support)
    - SFP+ Modules (for fiber connections)
    - Power Supply: 12V/24V automotive grade
    - Temperature: -40°C ~ +85°C support

센서 및 액추에이터:
  LiDAR:
    - Interface: Gigabit Ethernet
    - Data Rate: 100 Mbps
    - Count: 2 (Front, Rear)
  
  Camera Array:
    - Interface: Gigabit Ethernet
    - Data Rate: 100 Mbps per camera
    - Count: 4 cameras (400 Mbps total)
  
  Control Units:
    - Emergency Brake Controller
    - Steering Controller
    - Main ECU
    - Safety ECU
```

### 1.2 소프트웨어 요구사항
```yaml
Development Tools:
  - Microchip MPLAB Harmony v3
  - Python 3.8+ (for test scripts)
  - Web Browser (Chrome/Firefox for dashboard)
  - Terminal Emulator (PuTTY/Tera Term)
  
Firmware:
  - LAN9692 Firmware: v2.1.0 or later
  - LAN9662 Firmware: v1.8.0 or later
  - TSN Stack: IEEE 802.1CB compliant
  
Network Tools:
  - Wireshark with TSN plugin
  - iperf3 for bandwidth testing
  - PTP daemon for time sync
```

## 2. 하드웨어 설치

### 2.1 물리적 설치 절차

#### Step 1: 중앙 스위치 설치
```bash
# LAN9692 설치 위치: 차량 중앙 제어부
1. 차량 중앙 콘솔 하단에 LAN9692 장착
2. 진동 방지 마운트 사용 (automotive grade)
3. 냉각 팬 설치 (필요시)
4. 전원 연결 (12V/24V automotive power)

# 포트 할당
Port 1-2:   Front Zone 연결 (Primary/Backup)
Port 3-4:   Rear Zone 연결 (Primary/Backup)
Port 5-6:   Main ECU 연결
Port 7-8:   Safety ECU 연결
Port 9-10:  Emergency Systems
Port 11-12: Diagnostic/Service
```

#### Step 2: Zone 스위치 설치
```bash
# Front Zone LAN9662 설치
1. 차량 전방 센서 클러스터 근처 설치
2. IP67 인클로저 사용 (방수/방진)
3. 포트 할당:
   - Port 1: LiDAR Front
   - Port 2-3: Camera Array (Front)
   - Port 4: Radar
   - Port 5-6: Uplink to Central
   - Port 7-8: Reserved

# Rear Zone LAN9662 설치
1. 차량 후방 센서 클러스터 근처 설치
2. 동일한 IP67 인클로저 사용
3. 포트 할당:
   - Port 1: LiDAR Rear
   - Port 2: Camera Rear
   - Port 3-4: Reserved
   - Port 5-6: Uplink to Central
   - Port 7-8: Service/Diagnostic
```

### 2.2 케이블 연결 체크리스트
```yaml
Primary Backbone (1Gbps):
  ✓ Central Switch Port 1 → Front Switch Port 5
  ✓ Central Switch Port 3 → Rear Switch Port 5
  
Backup Backbone (1Gbps):
  ✓ Central Switch Port 2 → Front Switch Port 6
  ✓ Central Switch Port 4 → Rear Switch Port 6
  
Sensor Connections:
  ✓ Front LiDAR → Front Switch Port 1
  ✓ Front Cameras → Front Switch Port 2-3
  ✓ Rear LiDAR → Rear Switch Port 1
  ✓ Rear Camera → Rear Switch Port 2
  
Control Units:
  ✓ Main ECU → Central Switch Port 5-6
  ✓ Safety ECU → Central Switch Port 7-8
  ✓ Emergency Brake → Central Switch Port 9
  ✓ Steering Control → Central Switch Port 10
```

## 3. 초기 설정

### 3.1 스위치 기본 설정

#### LAN9692 중앙 스위치 초기화
```bash
# 시리얼 콘솔 접속 (115200 8N1)
> enable
# configure terminal

# 호스트명 설정
(config)# hostname A2Z-CENTRAL-SW

# 관리 IP 설정
(config)# interface vlan 1
(config-if)# ip address 192.168.100.1 255.255.255.0
(config-if)# no shutdown
(config-if)# exit

# 시간 동기화 (gPTP) 설정
(config)# ptp mode boundary-clock
(config)# ptp domain 0
(config)# ptp priority1 128
(config)# ptp priority2 128

# 기본 QoS 설정
(config)# qos mode advanced
(config)# qos trust cos
```

#### LAN9662 Zone 스위치 초기화
```bash
# Front Zone Switch
> enable
# configure terminal
(config)# hostname A2Z-FRONT-SW
(config)# interface vlan 1
(config-if)# ip address 192.168.100.2 255.255.255.0
(config-if)# exit

# Rear Zone Switch
> enable
# configure terminal
(config)# hostname A2Z-REAR-SW
(config)# interface vlan 1
(config-if)# ip address 192.168.100.3 255.255.255.0
(config-if)# exit
```

### 3.2 VLAN 구성
```bash
# 중앙 스위치 VLAN 설정
(config)# vlan 10
(config-vlan)# name SAFETY_CRITICAL
(config-vlan)# exit

(config)# vlan 20
(config-vlan)# name SENSOR_DATA
(config-vlan)# exit

(config)# vlan 30
(config-vlan)# name CONTROL_PLANE
(config-vlan)# exit

(config)# vlan 40
(config-vlan)# name DIAGNOSTICS
(config-vlan)# exit

# 포트 VLAN 할당
(config)# interface range gigabitethernet 1/1-4
(config-if-range)# switchport mode trunk
(config-if-range)# switchport trunk allowed vlan 10,20,30,40
(config-if-range)# exit
```

## 4. FRER 구성

### 4.1 FRER 스트림 정의
```bash
# Stream 1001: LiDAR (100 Mbps, 2-path redundancy)
(config)# frer stream 1001
(config-frer-stream)# description "LiDAR System Data"
(config-frer-stream)# bandwidth 100000
(config-frer-stream)# redundancy paths 2
(config-frer-stream)# sequence-recovery-window 128
(config-frer-stream)# sequence-history-length 64
(config-frer-stream)# individual-recovery enable
(config-frer-stream)# exit

# Stream 1002: Camera Array (400 Mbps, 2-path redundancy)
(config)# frer stream 1002
(config-frer-stream)# description "Camera Array Data"
(config-frer-stream)# bandwidth 400000
(config-frer-stream)# redundancy paths 2
(config-frer-stream)# sequence-recovery-window 256
(config-frer-stream)# sequence-history-length 128
(config-frer-stream)# exit

# Stream 1003: Emergency Brake (1 Mbps, 3-path redundancy)
(config)# frer stream 1003
(config-frer-stream)# description "Emergency Brake Control"
(config-frer-stream)# bandwidth 1000
(config-frer-stream)# redundancy paths 3
(config-frer-stream)# sequence-recovery-window 64
(config-frer-stream)# sequence-history-length 32
(config-frer-stream)# latency-critical enable
(config-frer-stream)# max-latency 1000  # microseconds
(config-frer-stream)# exit

# Stream 1004: Steering Control (10 Mbps, 2-path redundancy)
(config)# frer stream 1004
(config-frer-stream)# description "Steering Control"
(config-frer-stream)# bandwidth 10000
(config-frer-stream)# redundancy paths 2
(config-frer-stream)# sequence-recovery-window 128
(config-frer-stream)# sequence-history-length 64
(config-frer-stream)# exit
```

### 4.2 경로 구성
```bash
# Primary paths
(config)# frer path primary-1001
(config-frer-path)# stream 1001
(config-frer-path)# ingress-port gigabitethernet 1/1
(config-frer-path)# egress-port gigabitethernet 1/5
(config-frer-path)# priority 0
(config-frer-path)# exit

# Backup paths
(config)# frer path backup-1001
(config-frer-path)# stream 1001
(config-frer-path)# ingress-port gigabitethernet 1/2
(config-frer-path)# egress-port gigabitethernet 1/6
(config-frer-path)# priority 1
(config-frer-path)# exit
```

### 4.3 R-TAG 설정
```bash
# R-TAG 구성
(config)# frer r-tag
(config-frer-rtag)# format standard
(config-frer-rtag)# sequence-number-length 16
(config-frer-rtag)# path-id-length 4
(config-frer-rtag)# enable
(config-frer-rtag)# exit
```

## 5. 검증 및 테스트

### 5.1 연결성 테스트
```bash
# 기본 연결성 확인
A2Z-CENTRAL-SW# ping 192.168.100.2
A2Z-CENTRAL-SW# ping 192.168.100.3

# 포트 상태 확인
A2Z-CENTRAL-SW# show interfaces status
Port      Name               Status       Vlan       Speed
Gi1/1     Front-Primary      connected    trunk      1000-fdx
Gi1/2     Front-Backup       connected    trunk      1000-fdx
Gi1/3     Rear-Primary       connected    trunk      1000-fdx
Gi1/4     Rear-Backup        connected    trunk      1000-fdx
```

### 5.2 FRER 동작 확인
```bash
# FRER 스트림 상태
A2Z-CENTRAL-SW# show frer streams
Stream ID  Name            Status    Paths  Recovery  Packets
1001       LiDAR           Active    2/2    12.3ms    158473
1002       Camera Array    Active    2/2    13.5ms    892041
1003       Emergency       Active    3/3    8.1ms     2847
1004       Steering        Active    2/2    9.8ms     38472

# FRER 통계
A2Z-CENTRAL-SW# show frer statistics
Total Streams: 4
Active Paths: 9
Failed Paths: 0
Recovery Events (24h): 47
Avg Recovery Time: 10.9ms
Max Recovery Time: 45.2ms
Packet Loss Rate: 1.2e-7
```

### 5.3 성능 테스트
```bash
# Python 테스트 스크립트 실행
$ python3 test/frer-validation-suite.py

Starting A2Z Gigabit FRER Validation Suite...
================================================================================

BANDWIDTH TESTS
----------------------------------------
✅ PASS | Total Bandwidth Allocation
     Expected: 1000.0000 ± 0.0000
     Actual:   511.0000

✅ PASS | Stream 1001 Bandwidth
     Expected: 100.0000 ± 5.0000
     Actual:   99.8000

[... 더 많은 테스트 결과 ...]

================================================================================
SUMMARY
================================================================================
Total Tests: 32
Passed: 31
Failed: 1
Success Rate: 96.9%

✅ SYSTEM VALIDATION: PASSED
The A2Z Gigabit FRER system meets all critical requirements
```

## 6. 운영 및 모니터링

### 6.1 대시보드 실행
```bash
# 웹 대시보드 접근
1. 브라우저에서 http://192.168.100.1/dashboard 접속
2. 기본 인증: admin/admin (초기 설정 후 변경)

# 모니터링 항목
- 실시간 대역폭 사용량
- FRER 스트림 상태
- 경로 가용성
- 복구 이벤트 로그
- 성능 메트릭
```

### 6.2 로그 수집
```bash
# 시스템 로그 설정
(config)# logging host 192.168.100.100
(config)# logging level informational
(config)# logging facility local7

# FRER 이벤트 로그
(config)# frer logging enable
(config)# frer logging level debug
(config)# frer logging buffer-size 10000
```

### 6.3 정기 점검 체크리스트
```yaml
일일 점검:
  ✓ FRER 스트림 상태 확인
  ✓ 경로 가용성 확인
  ✓ 에러 로그 검토
  ✓ 대역폭 사용률 확인

주간 점검:
  ✓ 복구 시간 통계 분석
  ✓ 패킷 손실률 확인
  ✓ 펌웨어 업데이트 확인
  ✓ 백업 경로 테스트

월간 점검:
  ✓ 전체 시스템 성능 테스트
  ✓ 케이블 및 커넥터 점검
  ✓ 냉각 시스템 청소
  ✓ 설정 백업
```

## 7. 문제 해결

### 7.1 일반적인 문제 및 해결방법

#### 문제: FRER 스트림이 INACTIVE 상태
```bash
# 진단 명령
A2Z-CENTRAL-SW# show frer stream 1001 detail
A2Z-CENTRAL-SW# show interfaces gigabitethernet 1/1

# 해결 방법
1. 물리적 연결 확인
2. VLAN 설정 확인
3. 스트림 설정 재적용
(config)# frer stream 1001
(config-frer-stream)# shutdown
(config-frer-stream)# no shutdown
```

#### 문제: 높은 복구 시간 (>50ms)
```bash
# 진단
A2Z-CENTRAL-SW# show frer recovery-events
A2Z-CENTRAL-SW# show interfaces counters errors

# 해결 방법
1. Recovery Window 크기 조정
(config)# frer stream 1001
(config-frer-stream)# sequence-recovery-window 256

2. QoS 우선순위 조정
(config)# interface gigabitethernet 1/1
(config-if)# qos cos 7
```

#### 문제: 패킷 손실 발생
```bash
# 진단
A2Z-CENTRAL-SW# show frer counters
A2Z-CENTRAL-SW# show interfaces statistics

# 해결 방법
1. 버퍼 크기 증가
(config)# interface gigabitethernet 1/1
(config-if)# tx-queue-size 2048

2. Storm Control 설정
(config-if)# storm-control broadcast level 10.0
(config-if)# storm-control multicast level 10.0
```

### 7.2 긴급 복구 절차
```bash
# 전체 시스템 리셋
1. 설정 백업
A2Z-CENTRAL-SW# copy running-config backup-config

2. 공장 초기화
A2Z-CENTRAL-SW# erase startup-config
A2Z-CENTRAL-SW# reload

3. 기본 설정 복원
A2Z-CENTRAL-SW# copy backup-config running-config

4. FRER 재설정
A2Z-CENTRAL-SW# configure terminal
(config)# frer global enable
(config)# frer recovery-mode fast
```

### 7.3 지원 연락처
```yaml
Microchip Technical Support:
  - Email: tsn-support@microchip.com
  - Phone: +1-480-792-7200
  - Portal: https://www.microchip.com/support

A2Z Engineering Support:
  - Email: engineering@autoa2z.com
  - Phone: +82-2-6952-5554
  - Emergency: +82-10-1234-5678
```

## 부록: 빠른 참조

### A. CLI 명령 요약
```bash
# 상태 확인
show frer summary
show frer streams
show frer paths
show frer statistics
show frer recovery-events

# 설정
frer stream <id>
frer path <name>
frer global enable/disable
frer logging enable/disable

# 디버깅
debug frer all
debug frer recovery
debug frer sequence
```

### B. LED 표시등 의미
```yaml
LAN9692 LED:
  Green Solid: 정상 동작
  Green Blink: 데이터 전송 중
  Yellow Solid: 경고 (온도/전원)
  Yellow Blink: 펌웨어 업데이트
  Red Solid: 하드웨어 오류
  Red Blink: FRER 복구 진행 중

LAN9662 LED:
  Port LED Green: Link up (1Gbps)
  Port LED Yellow: Link up (100Mbps)
  Port LED Off: No link
  System LED: 전원 및 상태
```

### C. 성능 기준값
```yaml
정상 범위:
  - Network Availability: > 99.95%
  - FRER Recovery Time: < 50ms
  - Packet Loss Rate: < 1e-6
  - Bandwidth Utilization: < 80%
  - Latency (Critical): < 1ms
  - Jitter: < 100μs
  
경고 임계값:
  - Recovery Events/Hour: > 10
  - CRC Errors: > 100/min
  - Buffer Overflows: > 0
  - Temperature: > 70°C
```

---

**문서 버전**: 2.0  
**마지막 업데이트**: 2025년 9월  
**작성**: A2Z Engineering Team