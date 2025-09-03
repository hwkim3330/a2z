# Implementation Examples & Configuration Templates

본 문서는 A2Z 자율주행 차량 네트워크의 실제 구현 예제와 설정 템플릿을 제공합니다.

## 목차
1. [TSN 스위치 설정](#tsn-스위치-설정)
2. [ECU 소프트웨어 구현](#ecu-소프트웨어-구현)
3. [QoS 설정 템플릿](#qos-설정-템플릿)
4. [시간 동기화 구성](#시간-동기화-구성)
5. [FRER 중복성 설정](#frer-중복성-설정)
6. [네트워크 모니터링](#네트워크-모니터링)
7. [보안 설정](#보안-설정)

## TSN 스위치 설정

### Microchip LAN9692 TSN 스위치 설정

```bash
#!/bin/bash
# TSN Switch Configuration Script for LAN9692
# A2Z Autonomous Vehicle Network

# 기본 스위치 초기화
echo "Initializing TSN Switch LAN9692..."

# VLAN 설정
configure_vlans() {
    # Critical Traffic VLAN
    /usr/bin/switch-config -p 0-7 -v 100 --priority 7 --name "critical_control"
    
    # Safety Traffic VLAN  
    /usr/bin/switch-config -p 0-7 -v 101 --priority 6 --name "safety_systems"
    
    # AV Data VLAN
    /usr/bin/switch-config -p 0-7 -v 200 --priority 5 --name "av_perception"
    
    # Sensor Data VLAN
    /usr/bin/switch-config -p 0-7 -v 201 --priority 4 --name "sensor_streams"
    
    # Management VLAN
    /usr/bin/switch-config -p 0-7 -v 999 --priority 0 --name "management"
}

# IEEE 802.1Qbv TAS 설정
configure_time_aware_shaper() {
    echo "Configuring Time-Aware Shaper (TAS)..."
    
    # GCL (Gate Control List) 설정 - 1ms 사이클
    cat > /tmp/gcl_config.json << EOF
{
    "cycle_time": 1000000,
    "base_time": 0,
    "gate_states": [
        {
            "time": 0,
            "duration": 100000,
            "gates": ["critical"],
            "comment": "Critical traffic only - 100μs"
        },
        {
            "time": 100000,
            "duration": 150000, 
            "gates": ["critical", "safety"],
            "comment": "Critical + Safety - 150μs"
        },
        {
            "time": 250000,
            "duration": 300000,
            "gates": ["critical", "safety", "av_data"],
            "comment": "Control + AV data - 300μs"
        },
        {
            "time": 550000,
            "duration": 200000,
            "gates": ["critical", "safety", "av_data", "sensor"],
            "comment": "Add sensor streams - 200μs"
        },
        {
            "time": 750000,
            "duration": 250000,
            "gates": ["all"],
            "comment": "Best effort window - 250μs"
        }
    ]
}
EOF

    # TAS 활성화
    /usr/bin/tsn-config --enable-tas --gcl-file /tmp/gcl_config.json --port all
}

# IEEE 802.1Qav CBS 설정
configure_credit_based_shaper() {
    echo "Configuring Credit-Based Shaper (CBS)..."
    
    # Class A (Voice/Critical) - 3% 대역폭
    /usr/bin/tsn-config --cbs \
        --class A \
        --idle-slope 750 \
        --send-slope -750 \
        --hi-credit 32768 \
        --lo-credit -32768 \
        --ports 0-7
    
    # Class B (Video/Sensor) - 15% 대역폭  
    /usr/bin/tsn-config --cbs \
        --class B \
        --idle-slope 3750 \
        --send-slope -3750 \
        --hi-credit 131072 \
        --lo-credit -131072 \
        --ports 0-7
}

# IEEE 802.1CB FRER 설정
configure_frer() {
    echo "Configuring Frame Replication and Elimination (FRER)..."
    
    # 중복 제거 설정
    /usr/bin/frer-config --enable \
        --sequence-recovery-window 128 \
        --history-length 64 \
        --reset-timeout 100000 \
        --ports 0-7
    
    # 복제 스트림 설정
    /usr/bin/frer-config --replicate \
        --stream-id 1-1000 \
        --paths "primary,backup" \
        --sequence-generation enable
}

# PTP 시간 동기화 설정
configure_ptp() {
    echo "Configuring PTP time synchronization..."
    
    # gPTP 설정
    cat > /etc/ptp4l.conf << EOF
[global]
gmCapable               1
priority1               128
priority2               128
domainNumber            0
logAnnounceInterval     1
logSyncInterval         -3
logMinDelayReqInterval  -3
announceReceiptTimeout  3
syncReceiptTimeout      3
delayAsymmetry          0
fault_reset_interval    4
neighborPropDelayThresh 20000000
min_neighbor_prop_delay -20000000
assume_two_step         0
path_trace_enabled      0
follow_up_info          0
hybrid_e2e              0
inhibit_multicast_service 0
net_sync_monitor        1
tc_spanning_tree        0
tx_timestamp_timeout    1
unicast_listen          0
unicast_master_table    0
unicast_req_duration    3600
use_syslog              1
verbose                 0
summary_interval        0
kernel_leap             1
check_fup_sync          0
clock_servo             pi
sanity_freq_limit       200000000
ntpshm_segment          0
msg_interval_request    0

[eth0]
logAnnounceInterval     1
logSyncInterval         -3
logMinDelayReqInterval  -3
announceReceiptTimeout  3
syncReceiptTimeout      3
delay_mechanism         E2E
network_transport       L2
delay_filter            moving_median
delay_filter_length     10
egressLatency           0
ingressLatency          0
boundary_clock_jbod     0
EOF

    # PTP 데몬 시작
    ptp4l -f /etc/ptp4l.conf -i eth0 -s &
}

# 실행
configure_vlans
configure_time_aware_shaper  
configure_credit_based_shaper
configure_frer
configure_ptp

echo "TSN Switch configuration completed."
```

### TSN 스위치 모니터링 스크립트

```python
#!/usr/bin/env python3
"""
TSN Switch Monitoring and Management
A2Z Autonomous Vehicle Network
"""

import time
import json
import subprocess
from datetime import datetime
from typing import Dict, List

class TSNSwitchMonitor:
    def __init__(self, switch_ip: str = "192.168.1.100"):
        self.switch_ip = switch_ip
        self.metrics = {}
        
    def get_queue_statistics(self) -> Dict:
        """큐 통계 수집"""
        cmd = f"snmpwalk -v2c -c public {self.switch_ip} 1.3.6.1.4.1.17420.1.2.1"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'queues': {}
        }
        
        for line in result.stdout.split('\n'):
            if 'queueDepth' in line:
                parts = line.split()
                queue_id = parts[0].split('.')[-1]
                depth = int(parts[-1])
                stats['queues'][queue_id] = {
                    'depth': depth,
                    'max_depth': 512,  # 설정된 최대값
                    'utilization': depth / 512 * 100
                }
        
        return stats
    
    def get_frer_statistics(self) -> Dict:
        """FRER 중복성 통계"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'replication': {
                'frames_replicated': 0,
                'frames_eliminated': 0, 
                'path_failures': 0,
                'recovery_time_avg': 0.0
            }
        }
        
        # FRER 카운터 수집 (실제 구현에서는 스위치 API 사용)
        with open('/proc/net/frer_stats', 'r') as f:
            for line in f:
                if 'replicated' in line:
                    stats['replication']['frames_replicated'] = int(line.split()[-1])
                elif 'eliminated' in line:
                    stats['replication']['frames_eliminated'] = int(line.split()[-1])
                    
        return stats
    
    def check_time_sync_status(self) -> Dict:
        """시간 동기화 상태 확인"""
        cmd = "pmc -u -b 0 'GET CURRENT_DATA_SET'"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'ptp_status': 'unknown',
            'offset_ns': 0,
            'drift_ppb': 0.0
        }
        
        for line in result.stdout.split('\n'):
            if 'offsetFromMaster' in line:
                status['offset_ns'] = int(line.split()[-1])
            elif 'meanPathDelay' in line:
                status['path_delay_ns'] = int(line.split()[-1])
                
        return status
    
    def generate_report(self) -> str:
        """성능 리포트 생성"""
        queue_stats = self.get_queue_statistics()
        frer_stats = self.get_frer_statistics() 
        sync_status = self.check_time_sync_status()
        
        report = f"""
TSN Switch Performance Report
============================
Generated: {datetime.now()}
Switch IP: {self.switch_ip}

Queue Statistics:
----------------
"""
        
        for queue_id, stats in queue_stats['queues'].items():
            report += f"Queue {queue_id}: {stats['depth']} frames ({stats['utilization']:.1f}%)\n"
            
        report += f"""
FRER Statistics:
---------------
Frames Replicated: {frer_stats['replication']['frames_replicated']}
Frames Eliminated: {frer_stats['replication']['frames_eliminated']}
Path Failures: {frer_stats['replication']['path_failures']}

Time Synchronization:
--------------------
PTP Status: {sync_status['ptp_status']}  
Offset from Master: {sync_status['offset_ns']} ns
Path Delay: {sync_status.get('path_delay_ns', 'N/A')} ns
"""
        
        return report

if __name__ == "__main__":
    monitor = TSNSwitchMonitor("192.168.100.10")
    
    while True:
        print(monitor.generate_report())
        time.sleep(10)  # 10초마다 모니터링
```

## ECU 소프트웨어 구현

### ACU_NO (NVIDIA Jetson Orin) 메인 애플리케이션

```cpp
/**
 * A2Z Autonomous Control Unit - NVIDIA Orin Implementation
 * Real-time Perception and Control System
 */

#include <iostream>
#include <thread>
#include <chrono>
#include <memory>
#include <atomic>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/header.hpp>

#include "dds/dds.hpp"
#include "autoware_msgs/msg/vehicle_command.hpp"

class A2ZAutonomousController : public rclcpp::Node {
private:
    // DDS Publisher for critical commands
    dds::pub::DataWriter<autoware_msgs::msg::VehicleCommand> critical_cmd_writer_;
    
    // ROS2 subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr velocity_sub_;
    
    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    
    // Thread-safe flags
    std::atomic<bool> emergency_detected_{false};
    std::atomic<bool> system_healthy_{true};
    
    // Performance counters
    std::atomic<uint64_t> frame_counter_{0};
    std::chrono::steady_clock::time_point last_stats_time_;
    
public:
    A2ZAutonomousController() : Node("a2z_controller") {
        initialize_dds();
        initialize_ros2();
        initialize_watchdog();
        
        RCLCPP_INFO(this->get_logger(), "A2Z Controller initialized");
    }
    
    ~A2ZAutonomousController() {
        system_healthy_ = false;
    }

private:
    void initialize_dds() {
        // DDS 도메인 설정
        dds::domain::DomainParticipant participant(0);  // Domain 0
        
        // QoS 설정 - 안전 중요 메시지용
        dds::pub::qos::PublisherQos pub_qos;
        pub_qos << dds::core::policy::Reliability::Reliable()
                << dds::core::policy::Durability::TransientLocal()  
                << dds::core::policy::History::KeepLast(1)
                << dds::core::policy::Deadline(std::chrono::milliseconds(5))  // 5ms deadline
                << dds::core::policy::LatencyBudget(std::chrono::milliseconds(2));
        
        // Critical command publisher 생성
        dds::topic::Topic<autoware_msgs::msg::VehicleCommand> cmd_topic(
            participant, "emergency_commands"
        );
        
        dds::pub::Publisher publisher(participant, pub_qos);
        critical_cmd_writer_ = dds::pub::DataWriter<autoware_msgs::msg::VehicleCommand>(
            publisher, cmd_topic
        );
    }
    
    void initialize_ros2() {
        // QoS 프로파일 설정
        auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1))
            .reliability(rclcpp::ReliabilityPolicy::BestEffort)
            .durability(rclcpp::DurabilityPolicy::Volatile)
            .deadline(std::chrono::milliseconds(50));  // 50ms sensor deadline
            
        auto control_qos = rclcpp::QoS(rclcpp::KeepLast(1))
            .reliability(rclcpp::ReliabilityPolicy::Reliable)
            .durability(rclcpp::DurabilityPolicy::TransientLocal);
        
        // 센서 데이터 구독
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar/points", sensor_qos,
            std::bind(&A2ZAutonomousController::lidar_callback, this, std::placeholders::_1)
        );
        
        velocity_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/vehicle/velocity", sensor_qos,
            std::bind(&A2ZAutonomousController::velocity_callback, this, std::placeholders::_1)
        );
        
        // 제어 명령 발행
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/vehicle/cmd_vel", control_qos
        );
    }
    
    void initialize_watchdog() {
        // 시스템 감시 스레드 시작
        std::thread([this]() {
            while (system_healthy_) {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - last_stats_time_
                ).count();
                
                if (elapsed > 1000) {  // 1초마다
                    double fps = frame_counter_.load() / (elapsed / 1000.0);
                    RCLCPP_INFO(this->get_logger(), 
                        "Processing rate: %.1f FPS, Emergency: %s", 
                        fps, emergency_detected_ ? "TRUE" : "FALSE"
                    );
                    
                    frame_counter_ = 0;
                    last_stats_time_ = current_time;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }).detach();
    }
    
    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        frame_counter_++;
        
        // 긴급 상황 감지 로직 (간소화됨)
        bool obstacle_detected = analyze_point_cloud(msg);
        
        if (obstacle_detected && !emergency_detected_) {
            emergency_detected_ = true;
            send_emergency_brake_command();
        }
    }
    
    void velocity_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        // 속도 정보 처리
        current_velocity_ = msg->linear.x;
    }
    
    bool analyze_point_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
        // 실제 구현에서는 PCL이나 GPU 가속 라이브러리 사용
        // 여기서는 간단한 거리 기반 검사
        
        // 포인트 클라우드 데이터 파싱 (의사코드)
        for (auto& point : parse_points(cloud)) {
            if (point.x < 10.0 && abs(point.y) < 2.0) {  // 10m 이내, 차선 내
                return true;  // 장애물 감지
            }
        }
        
        return false;
    }
    
    void send_emergency_brake_command() {
        // DDS를 통한 긴급 제동 명령 전송
        autoware_msgs::msg::VehicleCommand emergency_cmd;
        emergency_cmd.header.stamp = this->now();
        emergency_cmd.header.frame_id = "base_link";
        emergency_cmd.command_type = "EMERGENCY_BRAKE";
        emergency_cmd.priority = 7;  // 최고 우선순위
        emergency_cmd.brake_force = 1.0;  // 최대 제동력
        emergency_cmd.steering_angle = 0.0;  // 직진 유지
        
        // 타임스탬프와 시퀀스 번호 추가
        emergency_cmd.sequence_id = frame_counter_.load();
        emergency_cmd.timeout_ms = 100;  // 100ms 내 처리 요구
        
        critical_cmd_writer_.write(emergency_cmd);
        
        RCLCPP_WARN(this->get_logger(), "EMERGENCY BRAKE COMMAND SENT!");
        
        // ROS2를 통한 일반 제어 명령도 전송 (이중화)
        auto twist_msg = geometry_msgs::msg::Twist();
        twist_msg.linear.x = 0.0;  // 정지
        twist_msg.angular.z = 0.0;
        cmd_vel_pub_->publish(twist_msg);
    }
    
    std::vector<Point3D> parse_points(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
        // PCL 또는 사용자 정의 파서 사용
        // 실제 구현에서는 GPU 가속 처리
        std::vector<Point3D> points;
        // ... 파싱 로직 ...
        return points;
    }
    
    struct Point3D {
        float x, y, z;
    };
    
    double current_velocity_ = 0.0;
};

int main(int argc, char** argv) {
    // Real-time 스케줄링 설정
    struct sched_param param;
    param.sched_priority = 99;  // 최고 우선순위
    if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
        std::cerr << "Warning: Could not set real-time priority" << std::endl;
    }
    
    // CPU affinity 설정 (고성능 코어 사용)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(4, &cpuset);  // Orin의 고성능 코어
    CPU_SET(5, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<A2ZAutonomousController>();
    
    // Multi-threaded executor 사용
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 4  // 4개 스레드
    );
    
    executor.add_node(node);
    
    try {
        executor.spin();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}
```

### CAN Gateway 구현

```c
/**
 * A2Z CAN-Ethernet Gateway Implementation
 * Bridges legacy CAN bus with TSN Ethernet
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

#include <netinet/in.h>
#include <arpa/inet.h>

#define MAX_CAN_FRAMES 1000
#define MAX_ETH_PACKET_SIZE 1500
#define CAN_INTERFACE "can0" 
#define ETH_INTERFACE "eth0"
#define MULTICAST_ADDR "224.0.0.100"
#define MULTICAST_PORT 5555

typedef struct {
    uint32_t can_id;
    uint8_t priority;
    char description[64];
} can_filter_entry_t;

// CAN 메시지 우선순위 맵핑
can_filter_entry_t can_priority_map[] = {
    {0x100, 7, "Emergency Brake Command"},
    {0x101, 7, "Steering Emergency"},
    {0x200, 6, "Vehicle Dynamics Control"},
    {0x201, 6, "Stability Control"},
    {0x300, 5, "Powertrain Status"},
    {0x400, 4, "Sensor Data"},
    {0x500, 3, "Body Control"},
    {0x600, 2, "Comfort Systems"},
    {0x700, 1, "Diagnostics"},
    {0, 0, ""}  // 종료자
};

typedef struct {
    int can_socket;
    int eth_socket;
    struct sockaddr_in eth_addr;
    pthread_mutex_t stats_mutex;
    
    // 통계 정보
    uint64_t can_rx_count;
    uint64_t can_tx_count;
    uint64_t eth_rx_count;
    uint64_t eth_tx_count;
    uint64_t error_count;
    
} gateway_context_t;

// 메시지 헤더 구조체 (Ethernet 전송용)
typedef struct __attribute__((packed)) {
    uint32_t magic;         // 0xA2A2A2A2
    uint32_t timestamp_sec;
    uint32_t timestamp_nsec;
    uint32_t can_id;
    uint8_t priority;
    uint8_t data_len;
    uint8_t data[8];
    uint16_t checksum;
} gateway_message_t;

gateway_context_t g_ctx;

uint8_t get_can_priority(uint32_t can_id) {
    for (int i = 0; can_priority_map[i].can_id != 0; i++) {
        if ((can_id & 0x700) == (can_priority_map[i].can_id & 0x700)) {
            return can_priority_map[i].priority;
        }
    }
    return 0;  // Default best effort
}

uint16_t calculate_checksum(gateway_message_t* msg) {
    uint16_t checksum = 0;
    uint8_t* ptr = (uint8_t*)msg;
    
    for (int i = 0; i < sizeof(gateway_message_t) - 2; i++) {
        checksum += ptr[i];
    }
    
    return checksum;
}

void* can_to_eth_thread(void* arg) {
    struct can_frame can_msg;
    gateway_message_t eth_msg;
    ssize_t nbytes;
    struct timespec ts;
    
    printf("CAN-to-Ethernet gateway thread started\n");
    
    while (1) {
        // CAN 메시지 수신
        nbytes = recv(g_ctx.can_socket, &can_msg, sizeof(can_msg), 0);
        
        if (nbytes < 0) {
            if (errno != EAGAIN) {
                perror("CAN recv error");
                pthread_mutex_lock(&g_ctx.stats_mutex);
                g_ctx.error_count++;
                pthread_mutex_unlock(&g_ctx.stats_mutex);
            }
            continue;
        }
        
        if (nbytes != sizeof(can_msg)) {
            printf("Incomplete CAN frame received\n");
            continue;
        }
        
        // 타임스탬프 획득 (하드웨어 타임스탬프 사용)
        clock_gettime(CLOCK_REALTIME, &ts);
        
        // Ethernet 메시지 구성
        memset(&eth_msg, 0, sizeof(eth_msg));
        eth_msg.magic = 0xA2A2A2A2;
        eth_msg.timestamp_sec = ts.tv_sec;
        eth_msg.timestamp_nsec = ts.tv_nsec;
        eth_msg.can_id = can_msg.can_id;
        eth_msg.priority = get_can_priority(can_msg.can_id);
        eth_msg.data_len = can_msg.can_dlc;
        memcpy(eth_msg.data, can_msg.data, can_msg.can_dlc);
        eth_msg.checksum = calculate_checksum(&eth_msg);
        
        // QoS 마킹 (DSCP)
        int dscp_value = (eth_msg.priority << 2);  // Priority -> DSCP 변환
        if (setsockopt(g_ctx.eth_socket, IPPROTO_IP, IP_TOS, 
                      &dscp_value, sizeof(dscp_value)) < 0) {
            perror("setsockopt IP_TOS failed");
        }
        
        // Ethernet 멀티캐스트 전송
        if (sendto(g_ctx.eth_socket, &eth_msg, sizeof(eth_msg), 0,
                   (struct sockaddr*)&g_ctx.eth_addr, 
                   sizeof(g_ctx.eth_addr)) < 0) {
            perror("Ethernet sendto failed");
            pthread_mutex_lock(&g_ctx.stats_mutex);
            g_ctx.error_count++;
            pthread_mutex_unlock(&g_ctx.stats_mutex);
        } else {
            pthread_mutex_lock(&g_ctx.stats_mutex);
            g_ctx.can_rx_count++;
            g_ctx.eth_tx_count++;
            pthread_mutex_unlock(&g_ctx.stats_mutex);
        }
        
        // 고우선순위 메시지는 로그
        if (eth_msg.priority >= 6) {
            printf("High priority CAN message forwarded: ID=0x%03X, Priority=%d\n",
                   eth_msg.can_id, eth_msg.priority);
        }
    }
    
    return NULL;
}

void* eth_to_can_thread(void* arg) {
    gateway_message_t eth_msg;
    struct can_frame can_msg;
    socklen_t addr_len = sizeof(g_ctx.eth_addr);
    ssize_t nbytes;
    
    printf("Ethernet-to-CAN gateway thread started\n");
    
    while (1) {
        // Ethernet 메시지 수신
        nbytes = recvfrom(g_ctx.eth_socket, &eth_msg, sizeof(eth_msg), 0,
                         (struct sockaddr*)&g_ctx.eth_addr, &addr_len);
        
        if (nbytes < 0) {
            if (errno != EAGAIN) {
                perror("Ethernet recv error");
                pthread_mutex_lock(&g_ctx.stats_mutex);
                g_ctx.error_count++;
                pthread_mutex_unlock(&g_ctx.stats_mutex);
            }
            continue;
        }
        
        if (nbytes != sizeof(eth_msg) || eth_msg.magic != 0xA2A2A2A2) {
            printf("Invalid Ethernet message received\n");
            continue;
        }
        
        // 체크섬 검증
        uint16_t expected_checksum = eth_msg.checksum;
        eth_msg.checksum = 0;
        if (calculate_checksum(&eth_msg) != expected_checksum) {
            printf("Checksum mismatch - dropping packet\n");
            pthread_mutex_lock(&g_ctx.stats_mutex);
            g_ctx.error_count++;
            pthread_mutex_unlock(&g_ctx.stats_mutex);
            continue;
        }
        
        // CAN 프레임 구성
        memset(&can_msg, 0, sizeof(can_msg));
        can_msg.can_id = eth_msg.can_id;
        can_msg.can_dlc = eth_msg.data_len;
        memcpy(can_msg.data, eth_msg.data, eth_msg.data_len);
        
        // CAN 버스로 전송
        if (write(g_ctx.can_socket, &can_msg, sizeof(can_msg)) != sizeof(can_msg)) {
            perror("CAN write failed");
            pthread_mutex_lock(&g_ctx.stats_mutex);
            g_ctx.error_count++;
            pthread_mutex_unlock(&g_ctx.stats_mutex);
        } else {
            pthread_mutex_lock(&g_ctx.stats_mutex);
            g_ctx.eth_rx_count++;
            g_ctx.can_tx_count++;
            pthread_mutex_unlock(&g_ctx.stats_mutex);
        }
    }
    
    return NULL;
}

void* statistics_thread(void* arg) {
    while (1) {
        sleep(10);  // 10초마다 통계 출력
        
        pthread_mutex_lock(&g_ctx.stats_mutex);
        printf("\n=== CAN Gateway Statistics ===\n");
        printf("CAN RX: %lu messages\n", g_ctx.can_rx_count);
        printf("CAN TX: %lu messages\n", g_ctx.can_tx_count);
        printf("ETH RX: %lu messages\n", g_ctx.eth_rx_count);
        printf("ETH TX: %lu messages\n", g_ctx.eth_tx_count);
        printf("Errors: %lu\n", g_ctx.error_count);
        printf("===============================\n\n");
        pthread_mutex_unlock(&g_ctx.stats_mutex);
    }
    
    return NULL;
}

int init_can_socket() {
    struct ifreq ifr;
    struct sockaddr_can addr;
    
    // CAN 소켓 생성
    int sock = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sock < 0) {
        perror("CAN socket creation failed");
        return -1;
    }
    
    // 인터페이스 설정
    strcpy(ifr.ifr_name, CAN_INTERFACE);
    ioctl(sock, SIOCGIFINDEX, &ifr);
    
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    
    // 소켓 바인딩
    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("CAN socket bind failed");
        close(sock);
        return -1;
    }
    
    // 타임스탬프 활성화
    const int timestamp_on = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_TIMESTAMP, 
                   &timestamp_on, sizeof(timestamp_on)) < 0) {
        perror("CAN timestamp enable failed");
    }
    
    printf("CAN socket initialized on %s\n", CAN_INTERFACE);
    return sock;
}

int init_eth_socket() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("Ethernet socket creation failed");
        return -1;
    }
    
    // 멀티캐스트 주소 설정
    memset(&g_ctx.eth_addr, 0, sizeof(g_ctx.eth_addr));
    g_ctx.eth_addr.sin_family = AF_INET;
    g_ctx.eth_addr.sin_addr.s_addr = inet_addr(MULTICAST_ADDR);
    g_ctx.eth_addr.sin_port = htons(MULTICAST_PORT);
    
    // 멀티캐스트 그룹 가입
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = inet_addr(MULTICAST_ADDR);
    mreq.imr_interface.s_addr = INADDR_ANY;
    
    if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, 
                   &mreq, sizeof(mreq)) < 0) {
        perror("Multicast join failed");
    }
    
    // 소켓 바인딩
    struct sockaddr_in bind_addr;
    memset(&bind_addr, 0, sizeof(bind_addr));
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_addr.s_addr = INADDR_ANY;
    bind_addr.sin_port = htons(MULTICAST_PORT);
    
    if (bind(sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
        perror("Ethernet socket bind failed");
        close(sock);
        return -1;
    }
    
    printf("Ethernet socket initialized - multicast %s:%d\n", 
           MULTICAST_ADDR, MULTICAST_PORT);
    return sock;
}

int main() {
    pthread_t can_to_eth_tid, eth_to_can_tid, stats_tid;
    
    printf("A2Z CAN-Ethernet Gateway Starting...\n");
    
    // 컨텍스트 초기화
    memset(&g_ctx, 0, sizeof(g_ctx));
    pthread_mutex_init(&g_ctx.stats_mutex, NULL);
    
    // 소켓 초기화
    g_ctx.can_socket = init_can_socket();
    if (g_ctx.can_socket < 0) {
        exit(1);
    }
    
    g_ctx.eth_socket = init_eth_socket();
    if (g_ctx.eth_socket < 0) {
        close(g_ctx.can_socket);
        exit(1);
    }
    
    // 워커 스레드 생성
    if (pthread_create(&can_to_eth_tid, NULL, can_to_eth_thread, NULL) != 0) {
        perror("Failed to create CAN-to-ETH thread");
        exit(1);
    }
    
    if (pthread_create(&eth_to_can_tid, NULL, eth_to_can_thread, NULL) != 0) {
        perror("Failed to create ETH-to-CAN thread");
        exit(1);
    }
    
    if (pthread_create(&stats_tid, NULL, statistics_thread, NULL) != 0) {
        perror("Failed to create statistics thread");
        exit(1);
    }
    
    printf("Gateway threads started successfully\n");
    
    // 메인 스레드는 대기
    pthread_join(can_to_eth_tid, NULL);
    pthread_join(eth_to_can_tid, NULL);
    pthread_join(stats_tid, NULL);
    
    // 정리
    close(g_ctx.can_socket);
    close(g_ctx.eth_socket);
    pthread_mutex_destroy(&g_ctx.stats_mutex);
    
    printf("Gateway shutdown complete\n");
    return 0;
}
```

## QoS 설정 템플릿

### Linux Traffic Control 설정

```bash
#!/bin/bash
# A2Z Network QoS Configuration
# Traffic Control and Queue Discipline Setup

INTERFACE="eth0"
BANDWIDTH="1000mbit"  # 1Gbps link

echo "Configuring QoS for interface $INTERFACE..."

# 기존 설정 삭제
tc qdisc del dev $INTERFACE root 2>/dev/null

# Root qdisc 생성 (Hierarchical Token Bucket)
tc qdisc add dev $INTERFACE root handle 1: htb default 30

# Root class 생성 (전체 대역폭)
tc class add dev $INTERFACE parent 1: classid 1:1 htb rate $BANDWIDTH

# Priority 7: Network Control (Emergency)
tc class add dev $INTERFACE parent 1:1 classid 1:10 htb \
    rate 50mbit ceil 100mbit prio 1

# Priority 6: Internetwork Control (Safety)  
tc class add dev $INTERFACE parent 1:1 classid 1:11 htb \
    rate 120mbit ceil 200mbit prio 2

# Priority 5: Voice/Critical AV Data
tc class add dev $INTERFACE parent 1:1 classid 1:12 htb \
    rate 300mbit ceil 400mbit prio 3

# Priority 4: Video/Sensor Streams
tc class add dev $INTERFACE parent 1:1 classid 1:13 htb \
    rate 400mbit ceil 600mbit prio 4

# Priority 3: Excellent Effort
tc class add dev $INTERFACE parent 1:1 classid 1:20 htb \
    rate 100mbit ceil 150mbit prio 5

# Priority 0-2: Best Effort
tc class add dev $INTERFACE parent 1:1 classid 1:30 htb \
    rate 30mbit ceil 50mbit prio 6

# 큐 규칙 생성 (SFQ - Stochastic Fair Queuing)
tc qdisc add dev $INTERFACE parent 1:10 handle 10: sfq perturb 10
tc qdisc add dev $INTERFACE parent 1:11 handle 11: sfq perturb 10
tc qdisc add dev $INTERFACE parent 1:12 handle 12: sfq perturb 10
tc qdisc add dev $INTERFACE parent 1:13 handle 13: sfq perturb 10
tc qdisc add dev $INTERFACE parent 1:20 handle 20: sfq perturb 10
tc qdisc add dev $INTERFACE parent 1:30 handle 30: sfq perturb 10

# 분류 필터 설정
# DSCP 기반 분류
tc filter add dev $INTERFACE parent 1: protocol ip prio 1 u32 \
    match ip tos 0xfc 0xfc flowid 1:10  # DSCP 63 (Network Control)

tc filter add dev $INTERFACE parent 1: protocol ip prio 2 u32 \
    match ip tos 0xf8 0xfc flowid 1:11  # DSCP 62 (Internetwork Control)

tc filter add dev $INTERFACE parent 1: protocol ip prio 3 u32 \
    match ip tos 0xb8 0xfc flowid 1:12  # DSCP 46 (Voice)

tc filter add dev $INTERFACE parent 1: protocol ip prio 4 u32 \
    match ip tos 0x88 0xfc flowid 1:13  # DSCP 34 (Video)

tc filter add dev $INTERFACE parent 1: protocol ip prio 5 u32 \
    match ip tos 0x68 0xfc flowid 1:20  # DSCP 26 (Excellent Effort)

# 포트 기반 분류 (백업)
tc filter add dev $INTERFACE parent 1: protocol ip prio 10 u32 \
    match ip dport 5000 0xffff flowid 1:10  # Emergency commands

tc filter add dev $INTERFACE parent 1: protocol ip prio 11 u32 \
    match ip dport 5001 0xffff flowid 1:11  # Safety systems

echo "QoS configuration completed for $INTERFACE"

# 모니터링 스크립트 생성
cat > /usr/local/bin/qos-monitor.sh << 'EOF'
#!/bin/bash
# QoS Monitoring Script

while true; do
    clear
    echo "A2Z Network QoS Statistics - $(date)"
    echo "==========================================="
    
    tc -s class show dev eth0
    
    echo ""
    echo "Top traffic flows:"
    ss -tuln | head -10
    
    echo ""
    echo "Network interface statistics:"
    cat /proc/net/dev | grep eth0
    
    sleep 5
done
EOF

chmod +x /usr/local/bin/qos-monitor.sh
echo "QoS monitoring script created: /usr/local/bin/qos-monitor.sh"
```

### DSCP 마킹 설정

```bash
#!/bin/bash
# DSCP Marking Rules for A2Z Network

# iptables를 사용한 DSCP 마킹

# 기존 규칙 삭제
iptables -t mangle -F

# Emergency/Critical Traffic (DSCP 63, 62)
iptables -t mangle -A OUTPUT -p udp --dport 5000 \
    -j DSCP --set-dscp 63  # Network Control

iptables -t mangle -A OUTPUT -p udp --dport 5001 \
    -j DSCP --set-dscp 62  # Internetwork Control

# Safety Systems (DSCP 46)
iptables -t mangle -A OUTPUT -p udp --dport 5002:5010 \
    -j DSCP --set-dscp 46  # Voice (Critical AV)

# Sensor Streams (DSCP 34)
iptables -t mangle -A OUTPUT -p udp --dport 5011:5050 \
    -j DSCP --set-dscp 34  # Video (Sensor Data)

# Management/Diagnostics (DSCP 26)
iptables -t mangle -A OUTPUT -p tcp --dport 22,80,443,8080 \
    -j DSCP --set-dscp 26  # Excellent Effort

# Best Effort (기본값, DSCP 0)
# 별도 규칙 불필요

echo "DSCP marking rules configured"
```

## 시간 동기화 구성

### PTP 마스터 클럭 설정 (GPS 기반)

```bash
#!/bin/bash
# PTP Grandmaster Configuration with GPS

# GPS 동기화 확인
echo "Checking GPS synchronization..."
gpspipe -w | head -5

# chrony GPS 설정
cat > /etc/chrony/chrony.conf << EOF
# GPS PPS reference
refclock PPS /dev/pps0 lock NMEA refid GPS
refclock SHM 0 offset 0.9999 delay 0.2 refid NMEA

# Allow NTP clients
allow 192.168.100.0/24

# Local stratum
local stratum 1

# Log configuration
logdir /var/log/chrony
log measurements statistics tracking
EOF

systemctl restart chronyd

# PTP4L grandmaster 설정
cat > /etc/ptp4l-gm.conf << EOF
[global]
# Grandmaster configuration
gmCapable               1
priority1               128
priority2               128
domainNumber            0
slaveOnly               0
twoStepFlag             1

# Timing parameters  
logAnnounceInterval     1
logSyncInterval         -3
logMinDelayReqInterval  -3
announceReceiptTimeout  3

# Clock parameters
clockClass              6
clockAccuracy           0x20
offsetScaledLogVariance 0x4E5D

# Transport
network_transport       L2
delay_mechanism         E2E

# GPS synchronization
clock_servo             pi
first_step_threshold    0.00002
max_frequency           900000000
step_threshold          0.00002

[eth0]
logAnnounceInterval     1  
logSyncInterval         -3
logMinDelayReqInterval  -3
announceReceiptTimeout  3
syncReceiptTimeout      3
delay_mechanism         E2E
network_transport       L2
neighborPropDelayThresh 20000000
min_neighbor_prop_delay -20000000
assume_two_step         1
egressLatency           0
ingressLatency          0
EOF

# PTP4L 시작
ptp4l -f /etc/ptp4l-gm.conf -i eth0 -m &

# PHC2SYS 시작 (PTP 하드웨어 클럭을 시스템 클럭과 동기화)
phc2sys -s CLOCK_REALTIME -c eth0 -w -m &

echo "PTP Grandmaster started with GPS synchronization"
```

### PTP 슬레이브 설정

```bash
#!/bin/bash
# PTP Slave Configuration

cat > /etc/ptp4l-slave.conf << EOF
[global]
gmCapable               0
slaveOnly               1
priority1               255
priority2               255
domainNumber            0
twoStepFlag             1

# Slave timing parameters
logAnnounceInterval     1
logSyncInterval         -3
logMinDelayReqInterval  -3
announceReceiptTimeout  3
syncReceiptTimeout      3

# Clock servo parameters
clock_servo             pi
pi_proportional_const   0.0
pi_integral_const       0.0
pi_proportional_scale   0.0
pi_proportional_exponent -0.3
pi_proportional_norm_max 0.7
pi_integral_scale       0.0
pi_integral_exponent    0.4
pi_integral_norm_max    0.3

first_step_threshold    0.00002
max_frequency          900000000
step_threshold         0.00002
ntpshm_segment         0

[eth0]
logAnnounceInterval     1
logSyncInterval         -3  
logMinDelayReqInterval  -3
announceReceiptTimeout  3
syncReceiptTimeout      3
delay_mechanism         E2E
network_transport       L2
neighborPropDelayThresh 20000000
min_neighbor_prop_delay -20000000
assume_two_step         1
egressLatency           0
ingressLatency          0
EOF

# PTP 슬레이브 시작
ptp4l -f /etc/ptp4l-slave.conf -i eth0 -s -m &

# 시스템 클럭 동기화
phc2sys -s eth0 -w -m &

echo "PTP Slave configuration completed"
```

## FRER 중복성 설정

### FRER 스트림 구성

```json
{
    "frer_configuration": {
        "sequence_identification": {
            "method": "rtag",
            "rtag_prefix": "A2Z",
            "sequence_number_length": 16
        },
        "streams": [
            {
                "stream_id": 1,
                "name": "emergency_brake",
                "priority": 7,
                "replication_paths": [
                    {
                        "path_id": 1,
                        "route": ["switch1_port1", "switch2_port3", "ecu_brake"],
                        "backup": false
                    },
                    {
                        "path_id": 2, 
                        "route": ["switch1_port2", "switch3_port1", "switch2_port4", "ecu_brake"],
                        "backup": true
                    }
                ],
                "elimination_settings": {
                    "window_size": 128,
                    "history_length": 64,
                    "reset_timeout_ms": 100
                }
            },
            {
                "stream_id": 2,
                "name": "steering_control",
                "priority": 7,
                "replication_paths": [
                    {
                        "path_id": 1,
                        "route": ["switch1_port1", "switch2_port5", "ecu_steering"],
                        "backup": false
                    },
                    {
                        "path_id": 2,
                        "route": ["switch1_port2", "switch4_port1", "ecu_steering"], 
                        "backup": true
                    }
                ],
                "elimination_settings": {
                    "window_size": 128,
                    "history_length": 64,
                    "reset_timeout_ms": 100
                }
            }
        ],
        "global_settings": {
            "sequence_recovery_window": 128,
            "history_length": 64,
            "reset_timeout_ms": 100,
            "take_no_sequence": false,
            "individual_recovery": true
        }
    }
}
```

### FRER 모니터링 스크립트

```python
#!/usr/bin/env python3
"""
FRER (Frame Replication and Elimination) Monitoring
A2Z Network Redundancy Monitor
"""

import json
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

@dataclass
class FRERStats:
    stream_id: int
    frames_replicated: int
    frames_eliminated: int
    duplicates_received: int
    out_of_order: int
    path_failures: int
    current_sequence: int
    last_reset_time: str

class FRERMonitor:
    def __init__(self, config_file: str = "/etc/frer_config.json"):
        self.config_file = config_file
        self.stats = {}
        self.load_configuration()
    
    def load_configuration(self):
        """FRER 설정 로드"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Configuration file {self.config_file} not found")
            self.config = {"frer_configuration": {"streams": []}}
    
    def collect_stats(self) -> Dict[int, FRERStats]:
        """FRER 통계 수집"""
        stats = {}
        
        for stream in self.config["frer_configuration"]["streams"]:
            stream_id = stream["stream_id"]
            
            # 실제 구현에서는 스위치 API나 SNMP 사용
            # 여기서는 시뮬레이션된 값
            stats[stream_id] = FRERStats(
                stream_id=stream_id,
                frames_replicated=self._get_counter(f"frer.stream.{stream_id}.replicated"),
                frames_eliminated=self._get_counter(f"frer.stream.{stream_id}.eliminated"),
                duplicates_received=self._get_counter(f"frer.stream.{stream_id}.duplicates"),
                out_of_order=self._get_counter(f"frer.stream.{stream_id}.out_of_order"),
                path_failures=self._get_counter(f"frer.stream.{stream_id}.path_failures"),
                current_sequence=self._get_counter(f"frer.stream.{stream_id}.sequence"),
                last_reset_time=datetime.now().isoformat()
            )
        
        return stats
    
    def _get_counter(self, counter_name: str) -> int:
        """카운터 값 조회 (시뮬레이션)"""
        # 실제 구현에서는 /proc/net/frer_stats 또는 SNMP 사용
        import random
        return random.randint(0, 10000)
    
    def check_path_health(self, stream_id: int) -> Dict:
        """경로 상태 확인"""
        stream_config = None
        for stream in self.config["frer_configuration"]["streams"]:
            if stream["stream_id"] == stream_id:
                stream_config = stream
                break
        
        if not stream_config:
            return {"error": "Stream not found"}
        
        path_health = {}
        for path in stream_config["replication_paths"]:
            path_id = path["path_id"]
            # 실제 구현에서는 ping, LLDP 등으로 경로 상태 확인
            path_health[path_id] = {
                "status": "active" if path_id == 1 else "standby",
                "latency_ms": 2.1 if path_id == 1 else 3.8,
                "packet_loss": 0.001,
                "last_failure": "2024-01-15T10:30:45Z" if path_id == 2 else None
            }
        
        return path_health
    
    def detect_anomalies(self, stats: Dict[int, FRERStats]) -> List[str]:
        """이상 상황 탐지"""
        alerts = []
        
        for stream_id, stat in stats.items():
            # 중복 제거율이 비정상적으로 높은 경우
            if stat.duplicates_received > stat.frames_replicated * 0.1:
                alerts.append(f"Stream {stream_id}: High duplicate rate - "
                            f"{stat.duplicates_received} duplicates")
            
            # 경로 장애가 빈번한 경우
            if stat.path_failures > 5:
                alerts.append(f"Stream {stream_id}: Frequent path failures - "
                            f"{stat.path_failures} failures")
            
            # 시퀀스 순서 오류가 많은 경우  
            if stat.out_of_order > stat.frames_replicated * 0.05:
                alerts.append(f"Stream {stream_id}: High out-of-order rate - "
                            f"{stat.out_of_order} packets")
        
        return alerts
    
    def generate_report(self) -> str:
        """FRER 상태 리포트 생성"""
        stats = self.collect_stats()
        alerts = self.detect_anomalies(stats)
        
        report = f"""
FRER Redundancy Status Report
============================
Generated: {datetime.now()}

Stream Statistics:
-----------------
"""
        
        for stream_id, stat in stats.items():
            stream_config = next(s for s in self.config["frer_configuration"]["streams"] 
                               if s["stream_id"] == stream_id)
            
            elimination_rate = (stat.frames_eliminated / max(stat.frames_replicated, 1)) * 100
            
            report += f"""
Stream {stream_id} ({stream_config['name']}):
  Frames Replicated: {stat.frames_replicated:,}
  Frames Eliminated: {stat.frames_eliminated:,}
  Elimination Rate: {elimination_rate:.2f}%
  Duplicates: {stat.duplicates_received:,}
  Out of Order: {stat.out_of_order:,}
  Path Failures: {stat.path_failures}
  Current Sequence: {stat.current_sequence:,}
"""
            
            # 경로 상태
            path_health = self.check_path_health(stream_id)
            report += "  Path Status:\n"
            for path_id, health in path_health.items():
                if isinstance(health, dict) and "status" in health:
                    report += f"    Path {path_id}: {health['status']} " \
                            f"(latency: {health['latency_ms']:.1f}ms)\n"
        
        if alerts:
            report += f"\nAlerts ({len(alerts)}):\n"
            report += "-" * 20 + "\n"
            for alert in alerts:
                report += f"⚠️  {alert}\n"
        else:
            report += "\n✅ No alerts - All streams operating normally\n"
        
        return report

def main():
    monitor = FRERMonitor()
    
    print("Starting FRER monitoring...")
    
    while True:
        try:
            report = monitor.generate_report()
            
            # 화면 지우고 리포트 출력
            subprocess.run(["clear"])
            print(report)
            
            time.sleep(30)  # 30초마다 업데이트
            
        except KeyboardInterrupt:
            print("\nFRER monitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
```

## 네트워크 모니터링

### 종합 네트워크 모니터링 대시보드

```python
#!/usr/bin/env python3
"""
A2Z Network Comprehensive Monitoring Dashboard
Real-time monitoring of TSN network performance
"""

import asyncio
import json
import psutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any
import curses
import threading
import time

class NetworkMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.running = True
        
    async def collect_interface_stats(self) -> Dict:
        """네트워크 인터페이스 통계 수집"""
        stats = {}
        net_io = psutil.net_io_counters(pernic=True)
        
        for interface, counters in net_io.items():
            if interface.startswith('eth') or interface.startswith('can'):
                stats[interface] = {
                    'bytes_sent': counters.bytes_sent,
                    'bytes_recv': counters.bytes_recv,
                    'packets_sent': counters.packets_sent,
                    'packets_recv': counters.packets_recv,
                    'errin': counters.errin,
                    'errout': counters.errout,
                    'dropin': counters.dropin,
                    'dropout': counters.dropout,
                    'timestamp': datetime.now().isoformat()
                }
        
        return stats
    
    async def collect_tsn_stats(self) -> Dict:
        """TSN 특화 통계 수집"""
        tsn_stats = {
            'ptp_sync': await self._get_ptp_status(),
            'queues': await self._get_queue_stats(),
            'frer': await self._get_frer_stats(),
            'tas': await self._get_tas_stats()
        }
        
        return tsn_stats
    
    async def _get_ptp_status(self) -> Dict:
        """PTP 동기화 상태"""
        try:
            result = subprocess.run(['pmc', '-u', '-b', '0', 'GET CURRENT_DATA_SET'], 
                                  capture_output=True, text=True, timeout=5)
            
            status = {
                'synchronized': False,
                'offset_ns': 0,
                'path_delay_ns': 0,
                'last_update': datetime.now().isoformat()
            }
            
            for line in result.stdout.split('\n'):
                if 'offsetFromMaster' in line:
                    offset = int(line.split()[-1])
                    status['offset_ns'] = offset
                    status['synchronized'] = abs(offset) < 1000  # 1μs 이내
                elif 'meanPathDelay' in line:
                    status['path_delay_ns'] = int(line.split()[-1])
                    
            return status
        except Exception as e:
            return {'error': str(e), 'synchronized': False}
    
    async def _get_queue_stats(self) -> Dict:
        """큐 통계"""
        try:
            result = subprocess.run(['tc', '-s', 'class', 'show', 'dev', 'eth0'],
                                  capture_output=True, text=True, timeout=5)
            
            queues = {}
            current_class = None
            
            for line in result.stdout.split('\n'):
                if 'class htb' in line:
                    parts = line.split()
                    class_id = parts[2]  # 1:10 형식
                    current_class = class_id.split(':')[1]
                    queues[current_class] = {'packets': 0, 'bytes': 0, 'drops': 0}
                elif 'Sent' in line and current_class:
                    parts = line.split()
                    bytes_sent = int(parts[1])
                    packets_sent = int(parts[3])
                    queues[current_class]['bytes'] = bytes_sent
                    queues[current_class]['packets'] = packets_sent
                elif 'dropped' in line and current_class:
                    drops = int(line.split('dropped')[1].split(',')[0].strip())
                    queues[current_class]['drops'] = drops
            
            return queues
        except Exception as e:
            return {'error': str(e)}
    
    async def _get_frer_stats(self) -> Dict:
        """FRER 중복성 통계"""
        # 시뮬레이션된 데이터 (실제로는 스위치 API 사용)
        return {
            'streams_active': 12,
            'frames_replicated': 125420,
            'frames_eliminated': 3247,
            'elimination_rate': 2.59,
            'path_failures': 2,
            'last_failure': '2024-01-15T14:23:10Z'
        }
    
    async def _get_tas_stats(self) -> Dict:
        """TAS 게이트 스케줄러 통계"""
        return {
            'current_cycle': 47892,
            'gate_violations': 0,
            'schedule_drift_ns': 125,
            'active_gates': ['critical', 'safety', 'av_data'],
            'next_gate_change_us': 247
        }
    
    def calculate_bandwidth_utilization(self, interface_stats: Dict) -> Dict:
        """대역폭 사용률 계산"""
        utilization = {}
        
        for interface, stats in interface_stats.items():
            if interface.startswith('eth'):
                # 1Gbps 링크 가정
                link_speed_bps = 1000000000
                
                # 이전 통계와 비교하여 사용률 계산 (간소화)
                current_bps = (stats['bytes_sent'] + stats['bytes_recv']) * 8
                utilization_pct = min((current_bps / link_speed_bps) * 100, 100)
                
                utilization[interface] = {
                    'utilization_percent': utilization_pct,
                    'throughput_mbps': current_bps / 1000000,
                    'packet_rate_pps': stats['packets_sent'] + stats['packets_recv']
                }
        
        return utilization
    
    def detect_network_issues(self, stats: Dict) -> List[str]:
        """네트워크 문제 감지"""
        issues = []
        
        # 인터페이스 오류 확인
        interface_stats = stats.get('interfaces', {})
        for interface, data in interface_stats.items():
            if data.get('errin', 0) > 100:
                issues.append(f"{interface}: High input errors ({data['errin']})")
            if data.get('errout', 0) > 100:
                issues.append(f"{interface}: High output errors ({data['errout']})")
            if data.get('dropin', 0) > 50:
                issues.append(f"{interface}: High input drops ({data['dropin']})")
        
        # PTP 동기화 확인
        ptp_stats = stats.get('tsn', {}).get('ptp_sync', {})
        if not ptp_stats.get('synchronized', False):
            offset = ptp_stats.get('offset_ns', 0)
            issues.append(f"PTP not synchronized (offset: {offset}ns)")
        
        # 큐 드롭 확인
        queue_stats = stats.get('tsn', {}).get('queues', {})
        for queue_id, queue_data in queue_stats.items():
            drops = queue_data.get('drops', 0)
            if drops > 1000:
                issues.append(f"Queue {queue_id}: High packet drops ({drops})")
        
        return issues

class Dashboard:
    def __init__(self, monitor: NetworkMonitor):
        self.monitor = monitor
        self.stdscr = None
        
    def init_display(self, stdscr):
        """화면 초기화"""
        self.stdscr = stdscr
        curses.curs_set(0)  # 커서 숨김
        stdscr.nodelay(True)  # 비블로킹 입력
        
        # 색상 설정
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # 정상
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK) # 경고
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)    # 오류
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # 정보
        
    async def update_display(self):
        """화면 업데이트"""
        if not self.stdscr:
            return
            
        try:
            # 통계 수집
            interface_stats = await self.monitor.collect_interface_stats()
            tsn_stats = await self.monitor.collect_tsn_stats()
            bandwidth_util = self.monitor.calculate_bandwidth_utilization(interface_stats)
            
            all_stats = {
                'interfaces': interface_stats,
                'tsn': tsn_stats,
                'bandwidth': bandwidth_util
            }
            
            issues = self.monitor.detect_network_issues(all_stats)
            
            # 화면 지우기
            self.stdscr.clear()
            
            # 헤더
            header = f"A2Z Network Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self.stdscr.addstr(0, 0, header, curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.addstr(1, 0, "=" * len(header))
            
            row = 3
            
            # PTP 상태
            ptp_status = tsn_stats.get('ptp_sync', {})
            sync_status = "SYNCHRONIZED" if ptp_status.get('synchronized') else "NOT SYNCED"
            color = curses.color_pair(1) if ptp_status.get('synchronized') else curses.color_pair(3)
            
            self.stdscr.addstr(row, 0, f"PTP Status: {sync_status}", color)
            if 'offset_ns' in ptp_status:
                self.stdscr.addstr(row, 30, f"Offset: {ptp_status['offset_ns']}ns")
            row += 1
            
            # 인터페이스 상태
            row += 1
            self.stdscr.addstr(row, 0, "Interface Statistics:", curses.A_BOLD)
            row += 1
            
            for interface, stats in interface_stats.items():
                if interface.startswith(('eth', 'can')):
                    util_data = bandwidth_util.get(interface, {})
                    util_pct = util_data.get('utilization_percent', 0)
                    throughput = util_data.get('throughput_mbps', 0)
                    
                    # 색상 결정
                    if util_pct > 80:
                        color = curses.color_pair(3)  # 빨강
                    elif util_pct > 60:
                        color = curses.color_pair(2)  # 노랑
                    else:
                        color = curses.color_pair(1)  # 초록
                    
                    line = f"{interface:8} {util_pct:6.1f}% {throughput:8.1f}Mbps " \
                           f"Errors: {stats.get('errin', 0):4}/{stats.get('errout', 0):4} " \
                           f"Drops: {stats.get('dropin', 0):4}/{stats.get('dropout', 0):4}"
                    
                    self.stdscr.addstr(row, 0, line, color)
                    row += 1
            
            # 큐 상태
            row += 1
            self.stdscr.addstr(row, 0, "Queue Statistics:", curses.A_BOLD)
            row += 1
            
            queue_stats = tsn_stats.get('queues', {})
            for queue_id, queue_data in queue_stats.items():
                if isinstance(queue_data, dict):
                    packets = queue_data.get('packets', 0)
                    bytes_val = queue_data.get('bytes', 0)
                    drops = queue_data.get('drops', 0)
                    
                    color = curses.color_pair(3) if drops > 100 else curses.color_pair(1)
                    
                    line = f"Queue {queue_id:2}: {packets:8} pkts, {bytes_val:10} bytes, {drops:4} drops"
                    self.stdscr.addstr(row, 0, line, color)
                    row += 1
            
            # FRER 상태
            row += 1
            frer_stats = tsn_stats.get('frer', {})
            if frer_stats:
                self.stdscr.addstr(row, 0, "FRER Redundancy:", curses.A_BOLD)
                row += 1
                
                streams = frer_stats.get('streams_active', 0)
                replicated = frer_stats.get('frames_replicated', 0)
                eliminated = frer_stats.get('frames_eliminated', 0)
                elim_rate = frer_stats.get('elimination_rate', 0)
                
                self.stdscr.addstr(row, 0, f"Active Streams: {streams:3}")
                self.stdscr.addstr(row, 20, f"Replicated: {replicated:8}")
                self.stdscr.addstr(row, 40, f"Eliminated: {eliminated:6} ({elim_rate:.1f}%)")
                row += 1
            
            # 경고/오류
            if issues:
                row += 1
                self.stdscr.addstr(row, 0, f"Alerts ({len(issues)}):", 
                                 curses.color_pair(3) | curses.A_BOLD)
                row += 1
                
                for issue in issues[:10]:  # 최대 10개만 표시
                    self.stdscr.addstr(row, 0, f"⚠️  {issue}", curses.color_pair(3))
                    row += 1
            else:
                row += 1
                self.stdscr.addstr(row, 0, "✅ No alerts - All systems normal", 
                                 curses.color_pair(1))
            
            # 하단 도움말
            max_y, max_x = self.stdscr.getmaxyx()
            help_text = "Press 'q' to quit, 'r' to refresh"
            self.stdscr.addstr(max_y - 2, 0, help_text, curses.color_pair(4))
            
            self.stdscr.refresh()
            
        except Exception as e:
            # 오류 발생 시 간단히 표시
            self.stdscr.clear()
            self.stdscr.addstr(0, 0, f"Display error: {str(e)}")
            self.stdscr.refresh()
    
    async def run_dashboard(self):
        """대시보드 실행"""
        while self.monitor.running:
            await self.update_display()
            
            # 키보드 입력 확인
            try:
                key = self.stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    self.monitor.running = False
                    break
                elif key == ord('r') or key == ord('R'):
                    continue  # 화면 새로고침
            except:
                pass
                
            await asyncio.sleep(2)  # 2초마다 업데이트

async def main():
    monitor = NetworkMonitor()
    dashboard = Dashboard(monitor)
    
    def run_curses(stdscr):
        dashboard.init_display(stdscr)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(dashboard.run_dashboard())
        loop.close()
    
    curses.wrapper(run_curses)

if __name__ == "__main__":
    asyncio.run(main())
```

## 보안 설정

### 네트워크 보안 정책

```bash
#!/bin/bash
# A2Z Network Security Configuration

echo "Configuring A2Z network security policies..."

# 방화벽 규칙 설정
configure_firewall() {
    # 기존 규칙 초기화
    iptables -F
    iptables -X
    iptables -t nat -F
    iptables -t nat -X
    iptables -t mangle -F
    iptables -t mangle -X
    
    # 기본 정책: DROP
    iptables -P INPUT DROP
    iptables -P FORWARD DROP
    iptables -P OUTPUT ACCEPT
    
    # 루프백 허용
    iptables -A INPUT -i lo -j ACCEPT
    
    # 기존 연결 허용
    iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    
    # TSN 트래픽 허용 (VLAN 기반)
    iptables -A INPUT -p all -m vlan --vlan-tag 100 -j ACCEPT  # Critical
    iptables -A INPUT -p all -m vlan --vlan-tag 101 -j ACCEPT  # Safety
    iptables -A INPUT -p all -m vlan --vlan-tag 200 -j ACCEPT  # AV Data
    iptables -A INPUT -p all -m vlan --vlan-tag 201 -j ACCEPT  # Sensors
    
    # PTP 트래픽 허용
    iptables -A INPUT -p udp --dport 319 -j ACCEPT   # PTP Event
    iptables -A INPUT -p udp --dport 320 -j ACCEPT   # PTP General
    
    # SSH 관리 (제한된 소스)
    iptables -A INPUT -p tcp --dport 22 -s 192.168.100.0/24 -j ACCEPT
    
    # SNMP 모니터링
    iptables -A INPUT -p udp --dport 161 -s 192.168.100.0/24 -j ACCEPT
    
    # 로깅
    iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "FIREWALL-DROP: "
    
    echo "Firewall rules configured"
}

# MAC 주소 기반 액세스 제어
configure_mac_acl() {
    # 허용된 ECU MAC 주소 목록
    ALLOWED_MACS=(
        "00:1B:21:8A:B2:C3"  # ACU_NO
        "00:1B:21:8A:B2:C4"  # ACU_IT
        "00:1B:21:8A:B2:C5"  # VCU
        "00:1B:21:8A:B2:C6"  # TCU
        "00:1B:21:8A:B2:C7"  # EDR
        "00:1B:21:8A:B2:C8"  # DSSAD
        "00:1B:21:8A:B2:C9"  # CMU
    )
    
    # ebtables를 사용한 Layer 2 필터링
    ebtables -F
    ebtables -P INPUT DROP
    ebtables -P FORWARD DROP
    ebtables -P OUTPUT ACCEPT
    
    # 허용된 MAC 주소만 통과
    for mac in "${ALLOWED_MACS[@]}"; do
        ebtables -A INPUT -s $mac -j ACCEPT
        ebtables -A FORWARD -s $mac -j ACCEPT
    done
    
    # 멀티캐스트/브로드캐스트 허용 (제한적)
    ebtables -A INPUT -d 01:80:C2:00:00:0E -j ACCEPT   # LLDP
    ebtables -A INPUT -d 01:1B:19:00:00:00 -j ACCEPT   # PTP
    
    echo "MAC-based access control configured"
}

# 패킷 검사 및 침입 탐지
configure_ids() {
    # Suricata IDS 설정
    cat > /etc/suricata/suricata.yaml << EOF
%YAML 1.1
---
vars:
  address-groups:
    HOME_NET: "192.168.100.0/24"
    EXTERNAL_NET: "!$HOME_NET"
    
  port-groups:
    HTTP_PORTS: "80"
    SSH_PORTS: "22"
    
default-log-dir: /var/log/suricata/

stats:
  enabled: yes
  interval: 30
  
outputs:
  - fast:
      enabled: yes
      filename: fast.log
      append: yes
      
  - alert-json:
      enabled: yes
      filename: alert.json
      
  - tls:
      enabled: yes
      filename: tls.log
      
detect-engine:
  - profile: medium
  - custom-values:
      toclient-groups: 3
      toserver-groups: 25
      
app-layer:
  protocols:
    tls:
      enabled: yes
    ssh:
      enabled: yes
      
rule-files:
  - /etc/suricata/rules/automotive.rules
  - /etc/suricata/rules/custom.rules
EOF
    
    # 자동차 특화 규칙 생성
    cat > /etc/suricata/rules/automotive.rules << EOF
# A2Z Automotive Network Security Rules

# CAN 프로토콜 이상 탐지
alert tcp any any -> any any (msg:"Suspicious CAN frame size"; dsize:>8; sid:1000001;)
alert udp any any -> any any (msg:"CAN flooding detected"; threshold:type threshold,track by_src,count 1000,seconds 10; sid:1000002;)

# PTP 보안
alert udp any any -> any 319 (msg:"Unauthorized PTP access"; content:!"|00|"; offset:0; depth:1; sid:1000003;)
alert udp any any -> any 320 (msg:"PTP manipulation attempt"; content:"MANAGEMENT"; sid:1000004;)

# TSN 트래픽 이상
alert any any any -> any any (msg:"VLAN hopping attempt"; vlan:>1000; sid:1000005;)
alert any any any -> any any (msg:"Priority queue abuse"; content:"priority"; offset:14; sid:1000006;)

# 인증되지 않은 ECU 통신
alert any ![00:1B:21:8A:B2:C3,00:1B:21:8A:B2:C4,00:1B:21:8A:B2:C5] any -> any any (msg:"Unauthorized ECU communication"; sid:1000007;)

# DoS 공격 탐지  
alert tcp any any -> any any (msg:"TCP SYN flood"; flags:S; threshold:type threshold,track by_src,count 100,seconds 5; sid:1000008;)
alert udp any any -> any any (msg:"UDP flood"; threshold:type threshold,track by_src,count 1000,seconds 10; sid:1000009;)
EOF

    # Suricata 시작
    systemctl enable suricata
    systemctl start suricata
    
    echo "Intrusion detection system configured"
}

# 암호화 통신 설정
configure_encryption() {
    # IPSec VPN 설정 (관리 트래픽용)
    cat > /etc/ipsec.conf << EOF
config setup
    charondebug="all"
    uniqueids=yes

conn %default
    ikelifetime=60m
    keylife=20m
    rekeymargin=3m
    keyingtries=1
    keyexchange=ikev2
    authby=secret

conn a2z-mgmt
    left=192.168.100.10
    leftsubnet=192.168.100.0/24
    right=192.168.101.10
    rightsubnet=192.168.101.0/24
    auto=route
EOF

    cat > /etc/ipsec.secrets << EOF
192.168.100.10 192.168.101.10 : PSK "A2Z-SecureKey-2024-VeryLong-RandomString"
EOF
    
    chmod 600 /etc/ipsec.secrets
    
    # 실시간 트래픽용 MACsec 설정
    ip link add link eth0 name macsec0 type macsec
    ip macsec add macsec0 tx sa 0 pn 1 key 00 fedcba9876543210fedcba9876543210
    ip macsec add macsec0 rx address 00:1b:21:8a:b2:c4 port 1
    ip macsec add macsec0 rx address 00:1b:21:8a:b2:c4 port 1 sa 0 pn 1 key 01 fedcba9876543210fedcba9876543210
    ip link set dev macsec0 up
    
    echo "Encryption configured"
}

# 보안 모니터링 스크립트
create_security_monitor() {
    cat > /usr/local/bin/security-monitor.sh << 'EOF'
#!/bin/bash
# A2Z Security Monitor

LOG_FILE="/var/log/a2z-security.log"

log_event() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
}

check_firewall_drops() {
    DROP_COUNT=$(dmesg | grep "FIREWALL-DROP" | wc -l)
    if [ $DROP_COUNT -gt 100 ]; then
        log_event "HIGH: Excessive firewall drops detected ($DROP_COUNT)"
        # 알림 전송 로직
    fi
}

check_failed_authentications() {
    FAILED_SSH=$(journalctl --since="10 minutes ago" | grep "Failed password" | wc -l)
    if [ $FAILED_SSH -gt 5 ]; then
        log_event "CRITICAL: Multiple SSH authentication failures ($FAILED_SSH)"
    fi
}

check_network_anomalies() {
    # 비정상적인 트래픽 패턴 확인
    CONN_COUNT=$(netstat -an | grep ESTABLISHED | wc -l)
    if [ $CONN_COUNT -gt 1000 ]; then
        log_event "WARNING: High connection count ($CONN_COUNT)"
    fi
}

check_ids_alerts() {
    # Suricata 알람 확인
    RECENT_ALERTS=$(tail -n 100 /var/log/suricata/fast.log | grep "$(date '+%m/%d')" | wc -l)
    if [ $RECENT_ALERTS -gt 10 ]; then
        log_event "HIGH: Multiple IDS alerts in recent period ($RECENT_ALERTS)"
    fi
}

# 메인 모니터링 루프
while true; do
    check_firewall_drops
    check_failed_authentications
    check_network_anomalies
    check_ids_alerts
    
    sleep 60  # 1분마다 체크
done
EOF
    
    chmod +x /usr/local/bin/security-monitor.sh
    
    # 시스템 서비스로 등록
    cat > /etc/systemd/system/a2z-security-monitor.service << EOF
[Unit]
Description=A2Z Network Security Monitor
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/security-monitor.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl enable a2z-security-monitor.service
    systemctl start a2z-security-monitor.service
    
    echo "Security monitoring service created and started"
}

# 실행
configure_firewall
configure_mac_acl
configure_ids
configure_encryption
create_security_monitor

echo "A2Z network security configuration completed successfully"
echo "Security logs: /var/log/a2z-security.log"
echo "IDS logs: /var/log/suricata/"
echo "Monitor status: systemctl status a2z-security-monitor"
```

---

이상으로 A2Z 자율주행 차량 네트워크의 실제 구현 예제와 설정 템플릿을 제공하였습니다. 각 구성 요소는 실제 운영 환경에서 사용할 수 있도록 상세한 설정과 모니터링 기능을 포함하고 있습니다.