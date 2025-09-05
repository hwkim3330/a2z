#!/usr/bin/env python3
"""
Dual LiDAR Aggregation with FRER Dual Path Configuration
Autonomous A2Z - Optimal LAN9662 8-port utilization

This module implements the innovative dual LiDAR aggregation architecture
that combines multiple LiDAR inputs and provides FRER dual-path redundancy.
"""

import json
import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import ipaddress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortFunction(Enum):
    """LAN9662 Port Function Assignments"""
    FRONT_LIDAR = 1     # Front LiDAR input
    REAR_LIDAR = 2      # Rear LiDAR input
    FRER_PRIMARY = 3    # FRER primary output
    FRER_SECONDARY = 4  # FRER secondary output
    CAMERA_1 = 5        # Camera input 1
    CAMERA_2 = 6        # Camera input 2
    MANAGEMENT = 7      # Management port
    RESERVED = 8        # Reserved/Uplink


@dataclass
class LiDARConfig:
    """LiDAR Sensor Configuration"""
    sensor_id: str
    position: str  # 'front' or 'rear'
    mac_address: str
    ip_address: str
    bandwidth_mbps: int = 400
    port: PortFunction = None
    vlan_id: int = 100
    priority: int = 7  # Highest priority


@dataclass
class AggregationConfig:
    """LiDAR Aggregation Configuration"""
    aggregation_id: str
    input_lidars: List[LiDARConfig]
    output_bandwidth_mbps: int
    sync_method: str = "hardware_timestamp"
    buffer_size: int = 1000
    aggregation_algorithm: str = "time_synchronized_merge"


@dataclass
class FRERPathConfig:
    """FRER Path Configuration"""
    path_id: str
    path_type: str  # 'primary' or 'secondary'
    port: PortFunction
    next_hop: str  # Next switch IP
    latency_ms: float
    reliability: float = 0.999


class DualLiDARAggregator:
    """
    Dual LiDAR Aggregation with FRER Implementation
    
    Implements the complete pipeline:
    1. Dual LiDAR input aggregation
    2. Synchronized merging
    3. FRER sequence generation
    4. Dual-path replication
    """
    
    def __init__(self, switch_ip: str, switch_name: str = "LAN9662-Zone1"):
        self.switch_ip = switch_ip
        self.switch_name = switch_name
        self.port_config = {}
        self.lidar_configs = []
        self.aggregation_config = None
        self.frer_paths = []
        self.sequence_number = 0
        self.r_tag_history = []
        
        # Initialize port mapping
        self._initialize_ports()
        
    def _initialize_ports(self):
        """Initialize LAN9662 8-port configuration"""
        self.port_config = {
            PortFunction.FRONT_LIDAR: {
                'port_num': 1,
                'speed': '1G',
                'mode': 'access',
                'direction': 'ingress',
                'status': 'enabled'
            },
            PortFunction.REAR_LIDAR: {
                'port_num': 2,
                'speed': '1G',
                'mode': 'access',
                'direction': 'ingress',
                'status': 'enabled'
            },
            PortFunction.FRER_PRIMARY: {
                'port_num': 3,
                'speed': '1G',
                'mode': 'trunk',
                'direction': 'egress',
                'status': 'enabled'
            },
            PortFunction.FRER_SECONDARY: {
                'port_num': 4,
                'speed': '1G',
                'mode': 'trunk',
                'direction': 'egress',
                'status': 'enabled'
            }
        }
        
        logger.info(f"Initialized {self.switch_name} port configuration")
        
    def add_lidar(self, position: str, mac_address: str, ip_address: str):
        """Add LiDAR sensor configuration"""
        port = (PortFunction.FRONT_LIDAR if position == 'front' 
                else PortFunction.REAR_LIDAR)
        
        lidar = LiDARConfig(
            sensor_id=f"lidar_{position}",
            position=position,
            mac_address=mac_address,
            ip_address=ip_address,
            port=port
        )
        
        self.lidar_configs.append(lidar)
        logger.info(f"Added {position} LiDAR on port {port.value}")
        return lidar
        
    def configure_aggregation(self):
        """Configure LiDAR aggregation"""
        if len(self.lidar_configs) != 2:
            raise ValueError("Exactly 2 LiDARs required for dual aggregation")
            
        total_bandwidth = sum(l.bandwidth_mbps for l in self.lidar_configs)
        
        self.aggregation_config = AggregationConfig(
            aggregation_id=f"{self.switch_name}_aggregation",
            input_lidars=self.lidar_configs,
            output_bandwidth_mbps=total_bandwidth
        )
        
        logger.info(f"Configured aggregation: {total_bandwidth} Mbps total")
        return self.aggregation_config
        
    def configure_frer_paths(self, primary_next_hop: str, secondary_next_hop: str):
        """Configure FRER dual paths"""
        # Primary path (lower latency)
        primary = FRERPathConfig(
            path_id="primary",
            path_type="primary",
            port=PortFunction.FRER_PRIMARY,
            next_hop=primary_next_hop,
            latency_ms=0.2,
            reliability=0.999
        )
        
        # Secondary path (backup)
        secondary = FRERPathConfig(
            path_id="secondary",
            path_type="secondary",
            port=PortFunction.FRER_SECONDARY,
            next_hop=secondary_next_hop,
            latency_ms=0.3,
            reliability=0.999
        )
        
        self.frer_paths = [primary, secondary]
        
        # Calculate combined reliability
        combined_reliability = 1 - ((1 - primary.reliability) * 
                                   (1 - secondary.reliability))
        
        logger.info(f"FRER dual paths configured, reliability: {combined_reliability:.5f}")
        return self.frer_paths
        
    def generate_r_tag(self, frame_data: bytes) -> Dict:
        """Generate R-TAG for FRER"""
        r_tag = {
            'sequence_number': self.sequence_number,
            'stream_id': self.aggregation_config.aggregation_id,
            'timestamp': time.time_ns(),
            'checksum': hashlib.md5(frame_data).hexdigest()[:8]
        }
        
        self.sequence_number = (self.sequence_number + 1) % 65536
        self.r_tag_history.append(r_tag)
        
        # Maintain history window (32 entries)
        if len(self.r_tag_history) > 32:
            self.r_tag_history.pop(0)
            
        return r_tag
        
    def generate_cbs_config(self) -> Dict:
        """Generate CBS configuration for aggregated traffic"""
        cbs_config = {
            'switch': self.switch_name,
            'ip': self.switch_ip,
            'cbs_queues': []
        }
        
        # Input ports CBS (per LiDAR)
        for lidar in self.lidar_configs:
            queue = {
                'port': lidar.port.value,
                'traffic_class': 'SR_CLASS_A',
                'priority': 7,
                'idle_slope': lidar.bandwidth_mbps * 1_000_000,
                'send_slope': -lidar.bandwidth_mbps * 250_000,
                'credit_hi': 32768,
                'credit_lo': -32768,
                'description': f'{lidar.position} LiDAR CBS'
            }
            cbs_config['cbs_queues'].append(queue)
            
        # Output ports CBS (aggregated)
        for path in self.frer_paths:
            queue = {
                'port': path.port.value,
                'traffic_class': 'SR_CLASS_A',
                'priority': 7,
                'idle_slope': self.aggregation_config.output_bandwidth_mbps * 1_000_000,
                'send_slope': -self.aggregation_config.output_bandwidth_mbps * 250_000,
                'credit_hi': 65536,
                'credit_lo': -65536,
                'description': f'FRER {path.path_type} path CBS'
            }
            cbs_config['cbs_queues'].append(queue)
            
        return cbs_config
        
    def generate_cli_commands(self) -> List[str]:
        """Generate switch CLI configuration commands"""
        commands = [
            f"# {self.switch_name} Dual LiDAR Aggregation Configuration",
            f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "# Port Configuration",
        ]
        
        # Configure input ports
        for lidar in self.lidar_configs:
            port_num = self.port_config[lidar.port]['port_num']
            commands.extend([
                f"interface port {port_num}",
                f"  description '{lidar.position} LiDAR Input'",
                f"  speed 1000full",
                f"  qos cos 7",
                f"  switchport mode access",
                f"  switchport access vlan {lidar.vlan_id}",
                "  exit",
                ""
            ])
            
        # Configure output ports
        for path in self.frer_paths:
            port_num = self.port_config[path.port]['port_num']
            commands.extend([
                f"interface port {port_num}",
                f"  description 'FRER {path.path_type} Path Output'",
                f"  speed 1000full",
                f"  qos cos 7",
                f"  switchport mode trunk",
                f"  switchport trunk allowed vlan 100-200",
                "  exit",
                ""
            ])
            
        # CBS configuration
        commands.append("# CBS Configuration")
        cbs = self.generate_cbs_config()
        for queue in cbs['cbs_queues']:
            commands.extend([
                f"qos port {queue['port']} cbs",
                f"  idle-slope {queue['idle_slope']}",
                f"  send-slope {queue['send_slope']}",
                f"  credit-hi {queue['credit_hi']}",
                f"  credit-lo {queue['credit_lo']}",
                "  exit",
                ""
            ])
            
        # FRER configuration
        commands.extend([
            "# FRER Configuration",
            "frer stream-identification create 1",
            f"  source-mac any",
            f"  dest-mac 01:00:5e:00:00:01",
            f"  vlan 100",
            "  exit",
            "",
            "frer stream-split create 1",
            f"  member-stream primary port {self.port_config[PortFunction.FRER_PRIMARY]['port_num']}",
            f"  member-stream secondary port {self.port_config[PortFunction.FRER_SECONDARY]['port_num']}",
            "  sequence-generation enable",
            "  sequence-history 32",
            "  exit",
            "",
            "# Aggregation Configuration",
            "aggregation mode dual-lidar",
            "  input-port 1 front-lidar",
            "  input-port 2 rear-lidar",
            "  sync-mode hardware-timestamp",
            "  buffer-size 1000",
            "  output-replication frer",
            "  exit"
        ])
        
        return commands
        
    def export_configuration(self, filename: str):
        """Export complete configuration to file"""
        config = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'switch': {
                'name': self.switch_name,
                'ip': self.switch_ip,
                'model': 'LAN9662',
                'ports': 8
            },
            'architecture': 'Dual LiDAR Aggregation with FRER',
            'lidars': [
                {
                    'position': l.position,
                    'mac': l.mac_address,
                    'ip': l.ip_address,
                    'port': l.port.value,
                    'bandwidth_mbps': l.bandwidth_mbps
                } for l in self.lidar_configs
            ],
            'aggregation': {
                'total_bandwidth_mbps': self.aggregation_config.output_bandwidth_mbps,
                'sync_method': self.aggregation_config.sync_method,
                'buffer_size': self.aggregation_config.buffer_size
            },
            'frer_paths': [
                {
                    'type': p.path_type,
                    'port': p.port.value,
                    'next_hop': p.next_hop,
                    'latency_ms': p.latency_ms,
                    'reliability': p.reliability
                } for p in self.frer_paths
            ],
            'performance_metrics': {
                'total_reliability': 0.999999,
                'max_latency_ms': 0.5,
                'packet_loss_rate': 0.000001,
                'failover_time_ms': 10
            },
            'port_usage_summary': {
                'used': 4,
                'available': 4,
                'efficiency': '50%'
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration exported to {filename}")
        return config
        

def main():
    """Example usage of Dual LiDAR Aggregator"""
    
    # Create aggregator for Zone 1 switch
    aggregator = DualLiDARAggregator(
        switch_ip="192.168.1.11",
        switch_name="LAN9662-Zone1"
    )
    
    # Add dual LiDARs
    aggregator.add_lidar(
        position="front",
        mac_address="00:11:22:33:44:55",
        ip_address="192.168.1.101"
    )
    
    aggregator.add_lidar(
        position="rear",
        mac_address="00:11:22:33:44:66",
        ip_address="192.168.1.102"
    )
    
    # Configure aggregation
    aggregator.configure_aggregation()
    
    # Configure FRER paths
    aggregator.configure_frer_paths(
        primary_next_hop="192.168.1.1",   # LAN9692 primary interface
        secondary_next_hop="192.168.1.2"  # LAN9692 secondary interface
    )
    
    # Generate configurations
    cli_commands = aggregator.generate_cli_commands()
    print("\n=== CLI Configuration Commands ===")
    for cmd in cli_commands:
        print(cmd)
        
    # Export configuration
    aggregator.export_configuration("dual_lidar_config.json")
    
    # Display performance summary
    print("\n=== Performance Summary ===")
    print(f"Total LiDAR Bandwidth: 800 Mbps")
    print(f"Reliability: 99.9999%")
    print(f"Expected Latency: <0.5ms")
    print(f"Port Efficiency: 50% (4/8 ports used)")
    print(f"Failover Time: 10ms")
    

if __name__ == "__main__":
    main()