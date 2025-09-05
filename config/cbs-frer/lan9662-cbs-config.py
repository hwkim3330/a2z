#!/usr/bin/env python3
"""
LAN9662 CBS (Credit-Based Shaper) Configuration Script
For A2Z Autonomous Vehicle LiDAR/Camera Bandwidth Guarantee

CBS는 IEEE 802.1Qav 표준으로 시간 민감 트래픽에 대역폭을 보장합니다.
각 LAN9662는 센서 직근에서 CBS를 통해 대역폭을 예약합니다.
"""

import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrafficClass(Enum):
    """IEEE 802.1Q Traffic Classes"""
    CLASS_A = 6  # SR Class A - Highest priority for LiDAR
    CLASS_B = 5  # SR Class B - Camera and Radar  
    CONTROL = 7  # Network Control
    BEST_EFFORT = 0  # Default traffic

@dataclass
class CBSConfig:
    """CBS Configuration Parameters"""
    port_id: int
    traffic_class: TrafficClass
    idle_slope: int  # bits per second (positive)
    send_slope: int  # bits per second (negative)
    credit_hi: int   # bytes (max accumulated credit)
    credit_lo: int   # bytes (max negative credit)
    bandwidth_mbps: int  # Allocated bandwidth in Mbps
    description: str

class LAN9662_CBS_Configurator:
    """CBS Configuration Manager for LAN9662 Switches"""
    
    # LAN9662 Hardware Limits
    MAX_PORTS = 8
    PORT_SPEED_GBPS = 1
    TOTAL_BANDWIDTH_MBPS = 1000
    
    def __init__(self, switch_id: str, ip_address: str):
        self.switch_id = switch_id
        self.ip_address = ip_address
        self.cbs_configs: List[CBSConfig] = []
        self.allocated_bandwidth = 0
        
        logger.info(f"Initializing CBS for {switch_id} at {ip_address}")
    
    def calculate_cbs_parameters(self, bandwidth_mbps: int, 
                                max_frame_size: int = 1522) -> tuple:
        """
        Calculate CBS idle_slope and send_slope based on bandwidth
        
        Formula:
        idle_slope = allocated_bandwidth
        send_slope = idle_slope - port_speed
        credit_hi = max_frame_size * 2
        credit_lo = max_frame_size
        """
        idle_slope = bandwidth_mbps * 1_000_000  # Convert to bps
        send_slope = idle_slope - (self.PORT_SPEED_GBPS * 1_000_000_000)
        credit_hi = max_frame_size * 2
        credit_lo = -max_frame_size
        
        return idle_slope, send_slope, credit_hi, credit_lo
    
    def add_lidar_cbs(self, port_id: int) -> CBSConfig:
        """Configure CBS for LiDAR sensor (400 Mbps)"""
        bandwidth_mbps = 400
        
        if self.allocated_bandwidth + bandwidth_mbps > self.TOTAL_BANDWIDTH_MBPS:
            raise ValueError(f"Insufficient bandwidth for LiDAR on port {port_id}")
        
        idle_slope, send_slope, credit_hi, credit_lo = \
            self.calculate_cbs_parameters(bandwidth_mbps, 9000)  # Jumbo frames for LiDAR
        
        config = CBSConfig(
            port_id=port_id,
            traffic_class=TrafficClass.CLASS_A,
            idle_slope=idle_slope,
            send_slope=send_slope,
            credit_hi=credit_hi,
            credit_lo=credit_lo,
            bandwidth_mbps=bandwidth_mbps,
            description=f"LiDAR Sensor Port {port_id}"
        )
        
        self.cbs_configs.append(config)
        self.allocated_bandwidth += bandwidth_mbps
        
        logger.info(f"Added LiDAR CBS: Port {port_id}, {bandwidth_mbps} Mbps")
        return config
    
    def add_camera_cbs(self, port_id: int) -> CBSConfig:
        """Configure CBS for Camera (200 Mbps)"""
        bandwidth_mbps = 200
        
        if self.allocated_bandwidth + bandwidth_mbps > self.TOTAL_BANDWIDTH_MBPS:
            raise ValueError(f"Insufficient bandwidth for Camera on port {port_id}")
        
        idle_slope, send_slope, credit_hi, credit_lo = \
            self.calculate_cbs_parameters(bandwidth_mbps, 1522)
        
        config = CBSConfig(
            port_id=port_id,
            traffic_class=TrafficClass.CLASS_B,
            idle_slope=idle_slope,
            send_slope=send_slope,
            credit_hi=credit_hi,
            credit_lo=credit_lo,
            bandwidth_mbps=bandwidth_mbps,
            description=f"Camera Port {port_id}"
        )
        
        self.cbs_configs.append(config)
        self.allocated_bandwidth += bandwidth_mbps
        
        logger.info(f"Added Camera CBS: Port {port_id}, {bandwidth_mbps} Mbps")
        return config
    
    def add_radar_cbs(self, port_id: int) -> CBSConfig:
        """Configure CBS for Radar (50 Mbps)"""
        bandwidth_mbps = 50
        
        if self.allocated_bandwidth + bandwidth_mbps > self.TOTAL_BANDWIDTH_MBPS:
            raise ValueError(f"Insufficient bandwidth for Radar on port {port_id}")
        
        idle_slope, send_slope, credit_hi, credit_lo = \
            self.calculate_cbs_parameters(bandwidth_mbps, 1522)
        
        config = CBSConfig(
            port_id=port_id,
            traffic_class=TrafficClass.CLASS_B,
            idle_slope=idle_slope,
            send_slope=send_slope,
            credit_hi=credit_hi,
            credit_lo=credit_lo,
            bandwidth_mbps=bandwidth_mbps,
            description=f"Radar Port {port_id}"
        )
        
        self.cbs_configs.append(config)
        self.allocated_bandwidth += bandwidth_mbps
        
        logger.info(f"Added Radar CBS: Port {port_id}, {bandwidth_mbps} Mbps")
        return config
    
    def generate_cli_commands(self) -> List[str]:
        """Generate Microchip CLI commands for CBS configuration"""
        commands = []
        commands.append(f"# CBS Configuration for {self.switch_id}")
        commands.append("configure terminal")
        
        for config in self.cbs_configs:
            commands.extend([
                f"interface gigabitethernet 0/{config.port_id}",
                f"qos map dscp-cos {config.traffic_class.value * 8} cos {config.traffic_class.value}",
                f"qos trust cos",
                f"qos cbs port-enable",
                f"exit"
            ])
            
            # Configure CBS parameters per class
            commands.extend([
                f"qos cbs class {config.traffic_class.value}",
                f"idle-slope {config.idle_slope}",
                f"send-slope {config.send_slope}", 
                f"credit-hi {config.credit_hi}",
                f"credit-lo {config.credit_lo}",
                f"exit"
            ])
        
        commands.append("end")
        commands.append("write memory")
        
        return commands
    
    def generate_json_config(self) -> str:
        """Generate JSON configuration for automation"""
        config_dict = {
            "switch_id": self.switch_id,
            "ip_address": self.ip_address,
            "total_bandwidth_mbps": self.TOTAL_BANDWIDTH_MBPS,
            "allocated_bandwidth_mbps": self.allocated_bandwidth,
            "available_bandwidth_mbps": self.TOTAL_BANDWIDTH_MBPS - self.allocated_bandwidth,
            "cbs_configurations": []
        }
        
        for config in self.cbs_configs:
            config_dict["cbs_configurations"].append({
                "port_id": config.port_id,
                "traffic_class": config.traffic_class.name,
                "bandwidth_mbps": config.bandwidth_mbps,
                "idle_slope_bps": config.idle_slope,
                "send_slope_bps": config.send_slope,
                "credit_hi_bytes": config.credit_hi,
                "credit_lo_bytes": config.credit_lo,
                "description": config.description
            })
        
        return json.dumps(config_dict, indent=2)
    
    def validate_configuration(self) -> bool:
        """Validate CBS configuration"""
        # Check total bandwidth allocation
        if self.allocated_bandwidth > self.TOTAL_BANDWIDTH_MBPS:
            logger.error(f"Over-allocated bandwidth: {self.allocated_bandwidth} > {self.TOTAL_BANDWIDTH_MBPS}")
            return False
        
        # Check for port conflicts
        used_ports = set()
        for config in self.cbs_configs:
            if config.port_id in used_ports:
                logger.error(f"Port {config.port_id} configured multiple times")
                return False
            used_ports.add(config.port_id)
        
        # Check CBS parameters validity
        for config in self.cbs_configs:
            if config.idle_slope <= 0:
                logger.error(f"Invalid idle_slope for port {config.port_id}")
                return False
            if config.send_slope >= 0:
                logger.error(f"Invalid send_slope for port {config.port_id}")
                return False
        
        logger.info("CBS configuration validated successfully")
        return True

def configure_all_lan9662_switches():
    """Configure CBS for all 6 LAN9662 switches in A2Z vehicle"""
    
    switches_config = {
        "LAN9662-1": {
            "ip": "192.168.1.11",
            "sensors": [
                ("lidar", 1),
                ("radar", 2),
                ("camera", 3)
            ]
        },
        "LAN9662-2": {
            "ip": "192.168.1.12",
            "sensors": [
                ("lidar", 1),
                ("radar", 2),
                ("camera", 3),
                ("camera", 4)
            ]
        },
        "LAN9662-3": {
            "ip": "192.168.1.13",
            "sensors": [
                ("lidar", 1),
                ("radar", 2),
                ("camera", 3)
            ]
        },
        "LAN9662-4": {
            "ip": "192.168.1.14",
            "sensors": [
                ("lidar", 1),
                ("radar", 2),
                ("camera", 3)
            ]
        },
        "LAN9662-5": {
            "ip": "192.168.1.15",
            "sensors": [
                ("lidar", 1),
                ("camera", 2),
                ("camera", 3)
            ]
        },
        "LAN9662-6": {
            "ip": "192.168.1.16",
            "sensors": [
                ("lidar", 1),
                ("radar", 2),
                ("camera", 3)
            ]
        }
    }
    
    all_configs = {}
    
    for switch_id, switch_info in switches_config.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Configuring {switch_id}")
        logger.info(f"{'='*50}")
        
        configurator = LAN9662_CBS_Configurator(switch_id, switch_info["ip"])
        
        for sensor_type, port_id in switch_info["sensors"]:
            if sensor_type == "lidar":
                configurator.add_lidar_cbs(port_id)
            elif sensor_type == "camera":
                configurator.add_camera_cbs(port_id)
            elif sensor_type == "radar":
                configurator.add_radar_cbs(port_id)
        
        if configurator.validate_configuration():
            all_configs[switch_id] = {
                "json": configurator.generate_json_config(),
                "cli": configurator.generate_cli_commands()
            }
            
            # Save CLI commands
            with open(f"cbs_{switch_id.lower()}_commands.txt", "w") as f:
                f.write("\n".join(configurator.generate_cli_commands()))
            
            # Save JSON config
            with open(f"cbs_{switch_id.lower()}_config.json", "w") as f:
                f.write(configurator.generate_json_config())
            
            logger.info(f"Configuration saved for {switch_id}")
            logger.info(f"Total allocated: {configurator.allocated_bandwidth} Mbps")
            logger.info(f"Available: {configurator.TOTAL_BANDWIDTH_MBPS - configurator.allocated_bandwidth} Mbps")
    
    return all_configs

def main():
    """Main execution"""
    logger.info("Starting LAN9662 CBS Configuration for A2Z Vehicle Network")
    logger.info("="*60)
    
    try:
        configs = configure_all_lan9662_switches()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("CBS CONFIGURATION SUMMARY")
        logger.info("="*60)
        
        total_lidar_bandwidth = 0
        total_camera_bandwidth = 0
        total_radar_bandwidth = 0
        
        for switch_id in configs:
            with open(f"cbs_{switch_id.lower()}_config.json", "r") as f:
                config = json.load(f)
                for cbs in config["cbs_configurations"]:
                    if "LiDAR" in cbs["description"]:
                        total_lidar_bandwidth += cbs["bandwidth_mbps"]
                    elif "Camera" in cbs["description"]:
                        total_camera_bandwidth += cbs["bandwidth_mbps"]
                    elif "Radar" in cbs["description"]:
                        total_radar_bandwidth += cbs["bandwidth_mbps"]
        
        logger.info(f"Total LiDAR Bandwidth: {total_lidar_bandwidth} Mbps")
        logger.info(f"Total Camera Bandwidth: {total_camera_bandwidth} Mbps")
        logger.info(f"Total Radar Bandwidth: {total_radar_bandwidth} Mbps")
        logger.info(f"Total Allocated: {total_lidar_bandwidth + total_camera_bandwidth + total_radar_bandwidth} Mbps")
        
        logger.info("\n✅ CBS Configuration Complete!")
        logger.info("Files generated:")
        logger.info("  - cbs_lan9662-*_commands.txt (CLI commands)")
        logger.info("  - cbs_lan9662-*_config.json (JSON config)")
        
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()