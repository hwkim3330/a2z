#!/usr/bin/env python3
"""
LAN9662 FRER (Frame Replication and Elimination for Reliability) Configuration
For A2Z Autonomous Vehicle Safety-Critical Sensor Data

FRER (IEEE 802.1CB) provides seamless redundancy by:
1. Replicating frames at ingress (LAN9662 near sensors)  
2. Sending copies via multiple paths
3. Eliminating duplicates at egress (LAN9692 central)

Each LAN9662 acts as a FRER replication point for LiDAR data.
"""

import json
import logging
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import ipaddress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FRERAlgorithm(Enum):
    """FRER Recovery Algorithms"""
    VECTOR = "vector_recovery"       # Vector recovery algorithm
    INDIVIDUAL = "individual_recovery"  # Individual recovery algorithm
    MATCH = "match_recovery"         # Match recovery algorithm

class StreamHandleType(Enum):
    """Stream Handle Allocation Types"""
    LIDAR = 0x1000      # LiDAR streams (0x1000-0x1FFF)
    CAMERA = 0x2000     # Camera streams (0x2000-0x2FFF)
    RADAR = 0x3000      # Radar streams (0x3000-0x3FFF)
    CONTROL = 0x4000    # Control streams (0x4000-0x4FFF)

@dataclass
class FRERStream:
    """FRER Stream Configuration"""
    stream_handle: int
    stream_id: str
    source_mac: str
    dest_mac: str
    vlan_id: int
    paths: List[str]  # List of path IDs
    algorithm: FRERAlgorithm
    history_length: int  # Sequence history length
    reset_timeout_ms: int
    latent_error_detection: bool
    description: str

@dataclass 
class FRERPath:
    """FRER Path Definition"""
    path_id: str
    switches: List[str]  # Ordered list of switches
    egress_port: int
    priority: int  # Path priority (1=primary, 2=secondary, 3=tertiary)
    active: bool = True

class LAN9662_FRER_Configurator:
    """FRER Configuration Manager for LAN9662 Edge Switches"""
    
    # R-TAG parameters
    RTAG_ETHERTYPE = 0xF1C1  # FRER R-TAG EtherType
    MAX_SEQ_NUMBER = 65535    # 16-bit sequence number
    
    def __init__(self, switch_id: str, ip_address: str):
        self.switch_id = switch_id
        self.ip_address = ip_address
        self.frer_streams: List[FRERStream] = []
        self.frer_paths: List[FRERPath] = []
        self.stream_counter = 0
        
        logger.info(f"Initializing FRER for {switch_id} at {ip_address}")
    
    def generate_stream_handle(self, stream_type: StreamHandleType) -> int:
        """Generate unique stream handle"""
        base_handle = stream_type.value
        handle = base_handle + self.stream_counter
        self.stream_counter += 1
        return handle
    
    def add_lidar_frer_stream(self, lidar_id: str, vlan_id: int = 100) -> FRERStream:
        """Configure FRER for LiDAR sensor with 3-way replication"""
        
        # Generate MAC addresses
        source_mac = self._generate_mac_from_id(lidar_id)
        dest_mac = "01:80:C2:00:00:0E"  # Multicast for safety-critical
        
        # Define 3 redundant paths
        paths = [
            f"path_primary_{lidar_id}",
            f"path_secondary_{lidar_id}", 
            f"path_tertiary_{lidar_id}"
        ]
        
        stream = FRERStream(
            stream_handle=self.generate_stream_handle(StreamHandleType.LIDAR),
            stream_id=f"lidar_{lidar_id}_{self.switch_id}",
            source_mac=source_mac,
            dest_mac=dest_mac,
            vlan_id=vlan_id,
            paths=paths,
            algorithm=FRERAlgorithm.VECTOR,
            history_length=256,  # Large history for LiDAR
            reset_timeout_ms=100,
            latent_error_detection=True,
            description=f"LiDAR {lidar_id} Safety-Critical Stream"
        )
        
        self.frer_streams.append(stream)
        
        # Create path definitions
        self._create_lidar_paths(lidar_id, paths)
        
        logger.info(f"Added LiDAR FRER stream: {stream.stream_id}")
        return stream
    
    def add_camera_frer_stream(self, camera_id: str, vlan_id: int = 101) -> FRERStream:
        """Configure FRER for Camera with 2-way replication"""
        
        source_mac = self._generate_mac_from_id(camera_id)
        dest_mac = "01:80:C2:00:00:0F"
        
        # Camera uses 2 paths (less critical than LiDAR)
        paths = [
            f"path_primary_{camera_id}",
            f"path_secondary_{camera_id}"
        ]
        
        stream = FRERStream(
            stream_handle=self.generate_stream_handle(StreamHandleType.CAMERA),
            stream_id=f"camera_{camera_id}_{self.switch_id}",
            source_mac=source_mac,
            dest_mac=dest_mac,
            vlan_id=vlan_id,
            paths=paths,
            algorithm=FRERAlgorithm.INDIVIDUAL,
            history_length=128,
            reset_timeout_ms=200,
            latent_error_detection=False,
            description=f"Camera {camera_id} Stream"
        )
        
        self.frer_streams.append(stream)
        self._create_camera_paths(camera_id, paths)
        
        logger.info(f"Added Camera FRER stream: {stream.stream_id}")
        return stream
    
    def _generate_mac_from_id(self, device_id: str) -> str:
        """Generate deterministic MAC address from device ID"""
        hash_obj = hashlib.md5(device_id.encode())
        hash_bytes = hash_obj.digest()[:5]
        # Set locally administered bit
        mac_bytes = bytearray([0x02]) + hash_bytes
        return ':'.join(f'{b:02x}' for b in mac_bytes).upper()
    
    def _create_lidar_paths(self, lidar_id: str, path_ids: List[str]):
        """Create 3 redundant paths for LiDAR traffic"""
        
        # Primary path (shortest)
        primary_path = FRERPath(
            path_id=path_ids[0],
            switches=[self.switch_id, "LAN9692"],
            egress_port=1,
            priority=1
        )
        
        # Secondary path (through adjacent switch)
        adjacent_switch = self._get_adjacent_switch()
        secondary_path = FRERPath(
            path_id=path_ids[1],
            switches=[self.switch_id, adjacent_switch, "LAN9692"],
            egress_port=2,
            priority=2
        )
        
        # Tertiary path (longest, most redundant)
        backup_switch = self._get_backup_switch()
        tertiary_path = FRERPath(
            path_id=path_ids[2],
            switches=[self.switch_id, backup_switch, adjacent_switch, "LAN9692"],
            egress_port=3,
            priority=3
        )
        
        self.frer_paths.extend([primary_path, secondary_path, tertiary_path])
    
    def _create_camera_paths(self, camera_id: str, path_ids: List[str]):
        """Create 2 redundant paths for Camera traffic"""
        
        primary_path = FRERPath(
            path_id=path_ids[0],
            switches=[self.switch_id, "LAN9692"],
            egress_port=4,
            priority=1
        )
        
        adjacent_switch = self._get_adjacent_switch()
        secondary_path = FRERPath(
            path_id=path_ids[1],
            switches=[self.switch_id, adjacent_switch, "LAN9692"],
            egress_port=5,
            priority=2
        )
        
        self.frer_paths.extend([primary_path, secondary_path])
    
    def _get_adjacent_switch(self) -> str:
        """Get adjacent switch based on current switch ID"""
        switch_map = {
            "LAN9662-1": "LAN9662-2",
            "LAN9662-2": "LAN9662-3",
            "LAN9662-3": "LAN9662-1",
            "LAN9662-4": "LAN9662-5",
            "LAN9662-5": "LAN9662-6",
            "LAN9662-6": "LAN9662-4"
        }
        return switch_map.get(self.switch_id, "LAN9662-2")
    
    def _get_backup_switch(self) -> str:
        """Get backup switch for tertiary path"""
        switch_map = {
            "LAN9662-1": "LAN9662-4",
            "LAN9662-2": "LAN9662-5",
            "LAN9662-3": "LAN9662-6",
            "LAN9662-4": "LAN9662-1",
            "LAN9662-5": "LAN9662-2",
            "LAN9662-6": "LAN9662-3"
        }
        return switch_map.get(self.switch_id, "LAN9662-5")
    
    def generate_cli_commands(self) -> List[str]:
        """Generate Microchip CLI commands for FRER configuration"""
        commands = []
        commands.append(f"# FRER Configuration for {self.switch_id}")
        commands.append("configure terminal")
        
        # Enable FRER globally
        commands.extend([
            "frer enable",
            f"frer r-tag ethertype {hex(self.RTAG_ETHERTYPE)}",
            ""
        ])
        
        for stream in self.frer_streams:
            commands.append(f"# Stream: {stream.description}")
            
            # Configure stream identification
            commands.extend([
                f"frer stream {stream.stream_handle}",
                f"identification source-mac {stream.source_mac}",
                f"identification dest-mac {stream.dest_mac}",
                f"identification vlan {stream.vlan_id}",
                ""
            ])
            
            # Configure sequence generation (at ingress)
            commands.extend([
                f"sequence-generation enable",
                f"sequence-generation algorithm {stream.algorithm.value}",
                ""
            ])
            
            # Configure replication (multiple egress ports)
            for i, path in enumerate(stream.paths):
                path_obj = next((p for p in self.frer_paths if p.path_id == path), None)
                if path_obj:
                    commands.extend([
                        f"replication port {path_obj.egress_port}",
                        f"replication port {path_obj.egress_port} enable",
                        ""
                    ])
            
            # Configure sequence recovery parameters
            commands.extend([
                f"sequence-recovery history-length {stream.history_length}",
                f"sequence-recovery reset-timeout {stream.reset_timeout_ms}",
                ""
            ])
            
            if stream.latent_error_detection:
                commands.append("sequence-recovery latent-error-detection enable")
            
            commands.append("exit\n")
        
        # Configure per-port FRER settings
        for path in self.frer_paths:
            commands.extend([
                f"interface gigabitethernet 0/{path.egress_port}",
                "frer enable",
                "exit"
            ])
        
        commands.extend(["end", "write memory"])
        return commands
    
    def generate_json_config(self) -> str:
        """Generate JSON configuration for automation"""
        config_dict = {
            "switch_id": self.switch_id,
            "ip_address": self.ip_address,
            "frer_enabled": True,
            "r_tag_ethertype": hex(self.RTAG_ETHERTYPE),
            "streams": [],
            "paths": []
        }
        
        for stream in self.frer_streams:
            config_dict["streams"].append({
                "handle": stream.stream_handle,
                "id": stream.stream_id,
                "source_mac": stream.source_mac,
                "dest_mac": stream.dest_mac,
                "vlan_id": stream.vlan_id,
                "paths": stream.paths,
                "algorithm": stream.algorithm.value,
                "history_length": stream.history_length,
                "reset_timeout_ms": stream.reset_timeout_ms,
                "latent_error_detection": stream.latent_error_detection,
                "description": stream.description
            })
        
        for path in self.frer_paths:
            config_dict["paths"].append({
                "path_id": path.path_id,
                "switches": path.switches,
                "egress_port": path.egress_port,
                "priority": path.priority,
                "active": path.active
            })
        
        return json.dumps(config_dict, indent=2)
    
    def validate_configuration(self) -> bool:
        """Validate FRER configuration"""
        
        # Check for stream handle conflicts
        handles = [s.stream_handle for s in self.frer_streams]
        if len(handles) != len(set(handles)):
            logger.error("Duplicate stream handles detected")
            return False
        
        # Validate MAC addresses
        for stream in self.frer_streams:
            try:
                # Check MAC format
                bytes.fromhex(stream.source_mac.replace(':', ''))
                bytes.fromhex(stream.dest_mac.replace(':', ''))
            except ValueError:
                logger.error(f"Invalid MAC address in stream {stream.stream_id}")
                return False
        
        # Validate paths exist for streams
        for stream in self.frer_streams:
            for path_id in stream.paths:
                if not any(p.path_id == path_id for p in self.frer_paths):
                    logger.error(f"Path {path_id} not defined for stream {stream.stream_id}")
                    return False
        
        # Check port conflicts
        used_ports = set()
        for path in self.frer_paths:
            if path.egress_port in used_ports:
                logger.warning(f"Port {path.egress_port} used by multiple paths")
            used_ports.add(path.egress_port)
        
        logger.info("FRER configuration validated successfully")
        return True

def configure_all_lan9662_frer():
    """Configure FRER for all 6 LAN9662 switches"""
    
    switches = {
        "LAN9662-1": {"ip": "192.168.1.11", "zone": "front_left"},
        "LAN9662-2": {"ip": "192.168.1.12", "zone": "front_center"},
        "LAN9662-3": {"ip": "192.168.1.13", "zone": "front_right"},
        "LAN9662-4": {"ip": "192.168.1.14", "zone": "rear_left"},
        "LAN9662-5": {"ip": "192.168.1.15", "zone": "rear_center"},
        "LAN9662-6": {"ip": "192.168.1.16", "zone": "rear_right"}
    }
    
    all_configs = {}
    
    for switch_id, info in switches.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Configuring FRER for {switch_id}")
        logger.info(f"{'='*50}")
        
        configurator = LAN9662_FRER_Configurator(switch_id, info["ip"])
        
        # Add LiDAR stream (all switches have LiDAR)
        configurator.add_lidar_frer_stream(f"{info['zone']}_lidar")
        
        # Front switches also have cameras
        if "front" in info["zone"]:
            configurator.add_camera_frer_stream(f"{info['zone']}_camera")
        
        if configurator.validate_configuration():
            all_configs[switch_id] = {
                "json": configurator.generate_json_config(),
                "cli": configurator.generate_cli_commands()
            }
            
            # Save configurations
            with open(f"frer_{switch_id.lower()}_commands.txt", "w") as f:
                f.write("\n".join(configurator.generate_cli_commands()))
            
            with open(f"frer_{switch_id.lower()}_config.json", "w") as f:
                f.write(configurator.generate_json_config())
            
            logger.info(f"FRER configuration saved for {switch_id}")
            logger.info(f"Streams configured: {len(configurator.frer_streams)}")
            logger.info(f"Paths configured: {len(configurator.frer_paths)}")
    
    return all_configs

def main():
    """Main execution"""
    logger.info("Starting LAN9662 FRER Configuration for A2Z Vehicle")
    logger.info("="*60)
    
    try:
        configs = configure_all_lan9662_frer()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("FRER CONFIGURATION SUMMARY")
        logger.info("="*60)
        
        total_streams = 0
        total_paths = 0
        
        for switch_id in configs:
            with open(f"frer_{switch_id.lower()}_config.json", "r") as f:
                config = json.load(f)
                total_streams += len(config["streams"])
                total_paths += len(config["paths"])
        
        logger.info(f"Total FRER Streams: {total_streams}")
        logger.info(f"Total Redundant Paths: {total_paths}")
        logger.info(f"Average Paths per Stream: {total_paths/total_streams:.1f}")
        
        # Calculate redundancy level
        lidar_streams = sum(1 for s in configs for c in json.loads(configs[s]["json"])["streams"] 
                          if "LiDAR" in c["description"])
        camera_streams = total_streams - lidar_streams
        
        logger.info(f"\nStream Distribution:")
        logger.info(f"  LiDAR Streams (3-way): {lidar_streams}")
        logger.info(f"  Camera Streams (2-way): {camera_streams}")
        
        logger.info("\n✅ FRER Configuration Complete!")
        logger.info("Files generated:")
        logger.info("  - frer_lan9662-*_commands.txt (CLI commands)")
        logger.info("  - frer_lan9662-*_config.json (JSON config)")
        
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()