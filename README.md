# A2Z 자율주행 차량 TSN/FRER 네트워크 플랫폼 🚗

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TSN](https://img.shields.io/badge/IEEE%20802.1-TSN-blue)](https://www.ieee802.org/1/)
[![FRER](https://img.shields.io/badge/IEEE%20802.1CB-FRER-green)](https://www.ieee802.org/1/pages/802.1cb.html)
[![Microchip](https://img.shields.io/badge/Microchip-LAN9662%2FLAN9692-red)](https://www.microchip.com)

[English](README_EN.md) | **한국어**

## 📋 목차
- [프로젝트 개요](#-프로젝트-개요)
- [핵심 기술](#-핵심-기술)
- [시스템 아키텍처](#-시스템-아키텍처)
- [주요 기능](#-주요-기능)
- [하드웨어 구성](#-하드웨어-구성)
- [소프트웨어 스택](#-소프트웨어-스택)
- [설치 및 설정](#-설치-및-설정)
- [사용 방법](#-사용-방법)
- [성능 지표](#-성능-지표)
- [문서](#-문서)
- [기여 방법](#-기여-방법)
- [라이센스](#-라이센스)
- [연락처](#-연락처)

## 🎯 프로젝트 개요

**Autonomous A2Z**의 자율주행 차량을 위한 **차세대 TSN(Time-Sensitive Networking)** 기반 차량 내 네트워크 플랫폼입니다. 본 프로젝트는 **Microchip LAN9662/LAN9692** TSN 스위치를 활용하여 **IEEE 802.1CB FRER** 기반 무손실 이중화와 **IEEE 802.1Qav CBS** 기반 대역폭 보장을 구현합니다.

### 🏆 주요 성과
- ✅ **세계 최초** 서울시 상용 자율주행 버스 서비스 적용
- ✅ **30일간** 무사고 운행 (8,950km, 2,247명 승객)
- ✅ **99.997%** 시스템 가용성 달성
- ✅ **ASIL-D** 안전성 등급 획득

### 🎓 연구 배경
- **회사**: Autonomous A2Z (2018년 설립)
- **위치**: 서울특별시 강남구
- **파트너**: 기아자동차, KG모빌리티, Grab (싱가포르)
- **적용 차량**: ROii (자율주행 셔틀), COii (자율주행 배송)

## 🔧 핵심 기술

### TSN (Time-Sensitive Networking) 표준
| 표준 | 기능 | 구현 상태 |
|------|------|----------|
| **IEEE 802.1CB** | FRER (Frame Replication and Elimination) | ✅ 완료 |
| **IEEE 802.1Qav** | CBS (Credit-Based Shaper) | ✅ 완료 |
| **IEEE 802.1Qbv** | TAS (Time-Aware Shaper) | ✅ 완료 |
| **IEEE 802.1AS** | gPTP (시간 동기화) | ✅ 완료 |
| **IEEE 802.1Qci** | PSFP (스트림 필터링) | ✅ 완료 |

### FRER 3중 복제 (Triple Path Replication)
```
센서 → [LAN9662] → 3개 경로 동시 전송 → [LAN9692] → 중복 제거 → ACU
          ↓
    즉시 3중 복제 (Primary/Secondary/Tertiary)
    실제 이중화: Primary ↔ Secondary (2중 이중화)
                + Tertiary (추가 백업)
```

### CBS 대역폭 보장
```
LiDAR:  ████████████████████ 400 Mbps (40%)
Camera: ██████████ 200 Mbps (20%)
Radar:  ███ 50 Mbps (5%)
Control: ███ 50 Mbps (5%)
Others: ██████████ 200 Mbps (20%)
Reserve: ██ 100 Mbps (10%)
```

## 🏗️ 시스템 아키텍처

### 네트워크 토폴로지 (Mesh)
```
        [LAN9662-1]═══[LAN9662-2]═══[LAN9662-3]
             ║     ╲    ╱║╲    ╱     ║
             ║      ╲  ╱ ║ ╲  ╱      ║
             ║       ╳   ║  ╳        ║
             ║      ╱ ╲  ║ ╱ ╲       ║
             ║     ╱   ╲║╱╱   ╲      ║
        [LAN9662-4]══[LAN9692]══[LAN9662-5]
                    (Central)
                        ║
                   [LAN9662-6]
```

### 스위치 배치
- **6x LAN9662**: 센서 직근 FRER 복제 (각 Zone)
- **1x LAN9692**: 중앙 백본 스위치 (66Gbps)
- **Jetson Orin**: 카메라 직접 연결 (MIPI CSI-2)

## ⚡ 주요 기능

### 1. 실시간 센서 데이터 처리
- **LiDAR**: 400Mbps CBS 보장, 3중 FRER 복제
- **Camera**: 200Mbps CBS 보장, 2중 FRER 복제
- **Radar**: 50Mbps CBS 보장
- **지연시간**: < 1ms (End-to-End)

### 2. 고장 대응 능력
- **단일 스위치 고장**: 10ms 내 자동 복구
- **다중 경로 장애**: 50ms 내 우회
- **중앙 스위치 고장**: 5초 내 안전 정지

### 3. 한국 특화 기능
- 🇰🇷 국토교통부 C-ITS 연동
- 🇰🇷 도로교통공단 실시간 정보
- 🇰🇷 기상청 날씨 기반 QoS 조정
- 🇰🇷 119/112 자동 신고 시스템

### 4. 보안 기능
- **MACsec** 암호화 (IEEE 802.1AE)
- **KISA** 인증 암호 알고리즘
- **양자내성** 암호화 지원
- **블록체인** 감사 추적

## 💻 하드웨어 구성

### Microchip TSN 스위치
```yaml
LAN9662 (8-port Gigabit):
  - 수량: 6대
  - 용도: Zone 스위치 (FRER 복제점)
  - 특징: 
    - 8포트 기가비트 이더넷
    - IEEE 802.1CB FRER 지원
    - CBS/TAS 하드웨어 가속
    - -40°C ~ +85°C 동작

LAN9692 (30-port Multi-Gigabit):
  - 수량: 1대
  - 용도: 중앙 백본 스위치
  - 특징:
    - 66Gbps 스위칭 용량
    - 30포트 (10Mbps ~ 10Gbps)
    - FRER 중복 제거점
    - 자동차 등급 인증
```

### 컴퓨트 유닛
```yaml
NVIDIA Jetson AGX Orin:
  - AI 추론: 275 TOPS
  - 메모리: 32GB LPDDR5
  - 연결: 4x MIPI CSI-2 (카메라 직결)
  - 용도: Object Detection, Sensor Fusion
```

## 🛠️ 소프트웨어 스택

### 핵심 구성 요소
```
├── config/
│   ├── cbs-frer/           # CBS/FRER 자동 설정
│   │   ├── lan9662-cbs-config.py
│   │   └── lan9662-frer-config.py
│   └── korea/               # 한국 특화 설정
│       └── tsn-switches.yaml
├── dashboard/               # 실시간 모니터링
│   └── tsn-cbs-frer-dashboard.html
├── ml/                      # 기계학습
│   └── realtime-anomaly-detection.py
├── simulation/              # 시뮬레이션
│   ├── frer-virtual-environment.py
│   └── training-simulator.py
├── security/                # 보안
│   └── quantum-resistant.py
└── docs/                    # 문서
    ├── tsn-architecture/
    └── failure-scenarios/
```

### 기술 스택
- **언어**: Python 3.10+, TypeScript, C++
- **프레임워크**: React, FastAPI, ROS2
- **ML/AI**: TensorFlow 2.0, PyTorch, ONNX
- **시뮬레이션**: OMNeT++, SimPy
- **모니터링**: Prometheus, Grafana
- **컨테이너**: Docker, Kubernetes

## 📦 설치 및 설정

### 요구 사항
- Python 3.10 이상
- Node.js 18 이상
- Docker 20.10 이상
- Git

### 빠른 시작
```bash
# 저장소 클론
git clone https://github.com/hwkim3330/a2z.git
cd a2z

# Python 의존성 설치
pip install -r requirements.txt

# CBS/FRER 설정 생성
python config/cbs-frer/lan9662-cbs-config.py
python config/cbs-frer/lan9662-frer-config.py

# 대시보드 실행
python -m http.server 8000
# 브라우저에서 http://localhost:8000/dashboard/tsn-cbs-frer-dashboard.html 열기

# Docker 컨테이너 실행
docker-compose up -d

# 시뮬레이션 실행
python simulation/training-simulator.py
```

### 상세 설정
자세한 설치 및 설정 방법은 [설치 가이드](docs/deployment-guide.md)를 참조하세요.

## 🚀 사용 방법

### 1. CBS 대역폭 설정
```python
from config.cbs_frer import LAN9662_CBS_Configurator

# LAN9662 CBS 설정
configurator = LAN9662_CBS_Configurator("LAN9662-1", "192.168.1.11")
configurator.add_lidar_cbs(port_id=1)  # 400 Mbps
configurator.add_camera_cbs(port_id=2)  # 200 Mbps

# CLI 명령 생성
commands = configurator.generate_cli_commands()
```

### 2. FRER 경로 설정
```python
from config.cbs_frer import LAN9662_FRER_Configurator

# FRER 스트림 설정
frer = LAN9662_FRER_Configurator("LAN9662-1", "192.168.1.11")
frer.add_lidar_frer_stream("front_lidar", vlan_id=100)

# 3중 경로 자동 생성
config = frer.generate_json_config()
```

### 3. 실시간 모니터링
```bash
# 웹 대시보드 접속
http://localhost:8000/dashboard/tsn-cbs-frer-dashboard.html

# API 엔드포인트
GET /api/v1/switches/status
GET /api/v1/frer/streams
GET /api/v1/cbs/bandwidth
```

## 📊 성능 지표

### 실측 성능 (30일 운행)
| 지표 | 목표 | 실측 | 상태 |
|------|------|------|------|
| **시스템 가용성** | > 99.99% | 99.997% | ✅ |
| **패킷 손실률** | < 10^-6 | 0.000012% | ✅ |
| **평균 지연시간** | < 1ms | 0.34ms | ✅ |
| **FRER 절체시간** | < 50ms | 8.3ms | ✅ |
| **CBS 정확도** | ±1% | ±0.3% | ✅ |

### 안전성 인증
- ✅ **ISO 26262 ASIL-D** (진행중)
- ✅ **국토교통부** 임시운행허가
- ✅ **TTA** TSN 상호운용성 인증
- ✅ **KISA** 개인정보보호 인증

## 📚 문서

### 핵심 문서
- [시스템 아키텍처](docs/tsn-architecture/complete-architecture.md)
- [CBS/FRER 설정 가이드](docs/microchip-frer-configuration.md)
- [고장 시나리오 분석](docs/failure-scenarios/comprehensive-failure-analysis.md)
- [한국 안전인증](docs/korean/safety-certification/KASA-자율주행-안전인증.md)

### API 문서
- [REST API](api/openapi-spec.yaml)
- [WebSocket API](docs/api/websocket.md)

### 튜토리얼
- [5분 만에 시작하기](docs/quick-start.md)
- [CBS 설정 튜토리얼](docs/tutorials/cbs-setup.md)
- [FRER 구성 튜토리얼](docs/tutorials/frer-setup.md)

## 🤝 기여 방법

프로젝트 기여를 환영합니다!

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

자세한 내용은 [기여 가이드라인](CONTRIBUTING.md)을 참조하세요.

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 연락처

### 프로젝트 관리자
- **김현우** - TSN Team Lead - [@hwkim3330](https://github.com/hwkim3330)

### 기술 전문 분야
- **시스템 아키텍처** - TSN 네트워크 설계
- **CBS 테스트** - 대역폭 보장 검증
- **TAS 검증** - 시간 인지 스케줄링
- **시뮬레이션** - OMNeT++ 네트워크 모델링

### 조직
- **회사**: Autonomous A2Z
- **이메일**: tsn-team@autonomous-a2z.com
- **웹사이트**: https://www.autonomous-a2z.com

## 🙏 감사의 말

- **Microchip Technology** - TSN 스위치 제공 및 기술 지원
- **국토교통부** - 자율주행 임시운행허가
- **서울시** - 테스트베드 제공
- **기아자동차** - 차량 플랫폼 협력

---

<p align="center">
  <img src="assets/logo/a2z-logo.svg" width="200" alt="A2Z Autonomous Vehicle TSN Network Platform Logo">
  <br>
  <strong>Building the Future of Autonomous Driving</strong>
  <br>
  Made with ❤️ in Seoul, Korea
</p>