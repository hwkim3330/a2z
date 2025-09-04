#!/usr/bin/env python3
"""
A2Z Automated Configuration Generator
Intelligent configuration generation for all Microchip TSN switch models
"""

import json
import yaml
import re
import ipaddress
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from datetime import datetime
import argparse
import os

class SwitchModel(Enum):
    LAN9692 = "LAN9692"  # 66Gbps, 30 ports
    LAN9668 = "LAN9668"  # 8 ports, 8Gbps
    LAN9662 = "LAN9662"  # 4 ports, industrial
    LAN9698 = "LAN9698"  # 98 ports, datacenter
    KSZ9897 = "KSZ9897"  # 7 ports, managed
    KSZ9567 = "KSZ9567"  # 7 ports, automotive
    KSZ9477 = "KSZ9477"  # 7 ports, industrial

class ConfigTemplate(Enum):
    AUTOMOTIVE = "automotive"
    INDUSTRIAL = "industrial" 
    DATACENTER = "datacenter"
    CUSTOM = "custom"

@dataclass
class NetworkRequirements:
    """Network requirements specification"""
    total_bandwidth_mbps: int
    num_critical_streams: int
    num_sensors: int
    num_controllers: int
    redundancy_level: int  # 1-3
    max_latency_ms: float
    enable_frer: bool = True
    enable_tas: bool = True
    enable_cbs: bool = True
    enable_gptp: bool = True

@dataclass
class VLANConfig:
    """VLAN configuration"""
    vlan_id: int
    name: str
    priority: int
    ports: List[str]

@dataclass
class FRERStreamConfig:
    """FRER stream configuration"""
    stream_id: int
    name: str
    bandwidth_mbps: int
    redundancy_paths: int
    max_latency_ms: float
    recovery_window: int
    history_length: int
    source_port: str
    dest_ports: List[str]

@dataclass
class QoSConfig:
    """QoS configuration"""
    class_id: int
    name: str
    priority: int
    bandwidth_percent: float
    cbs_enabled: bool
    idle_slope: int
    send_slope: int

class ConfigurationGenerator:
    """Main configuration generator"""
    
    def __init__(self):
        self.switch_capabilities = self._load_switch_capabilities()
        self.config_templates = self._load_templates()
        
    def _load_switch_capabilities(self) -> Dict[str, Dict]:
        """Load switch model capabilities"""
        return {
            SwitchModel.LAN9692: {
                'ports': 30,
                'switching_capacity_gbps': 66,
                'tsn_features': ['802.1CB', '802.1Qbv', '802.1Qav', '802.1AS', '802.1Qci'],
                'cpu': 'ARM Cortex-A53',
                'memory_mb': 512,
                'temperature_range': (-40, 85)
            },
            SwitchModel.LAN9668: {
                'ports': 8,
                'switching_capacity_gbps': 8,
                'tsn_features': ['802.1CB', '802.1Qbv', '802.1Qav', '802.1AS'],
                'cpu': 'ARM Cortex-A7',
                'memory_mb': 256,
                'temperature_range': (-40, 85)
            },
            SwitchModel.LAN9662: {
                'ports': 4,
                'switching_capacity_gbps': 4,
                'tsn_features': ['802.1Qbv', '802.1Qav', '802.1AS'],
                'cpu': 'ARM9',
                'memory_mb': 128,
                'temperature_range': (-40, 105)
            },
            SwitchModel.KSZ9897: {
                'ports': 7,
                'switching_capacity_gbps': 7,
                'tsn_features': ['802.1Qav', '802.1AS'],
                'cpu': 'MIPS',
                'memory_mb': 64,
                'temperature_range': (0, 70)
            }
        }
    
    def _load_templates(self) -> Dict[ConfigTemplate, Dict]:
        """Load configuration templates"""
        return {
            ConfigTemplate.AUTOMOTIVE: {
                'vlans': [
                    {'id': 10, 'name': 'SAFETY_CRITICAL', 'priority': 7},
                    {'id': 20, 'name': 'SENSOR_DATA', 'priority': 6},
                    {'id': 30, 'name': 'CONTROL', 'priority': 5},
                    {'id': 40, 'name': 'DIAGNOSTICS', 'priority': 3}
                ],
                'qos_classes': [
                    {'class': 7, 'name': 'Network Control', 'bandwidth': 5},
                    {'class': 6, 'name': 'Emergency', 'bandwidth': 15},
                    {'class': 5, 'name': 'Critical', 'bandwidth': 40},
                    {'class': 4, 'name': 'Vehicle Control', 'bandwidth': 25},
                    {'class': 3, 'name': 'Best Effort', 'bandwidth': 15}
                ],
                'security': 'high',
                'redundancy': 'dual-path'
            },
            ConfigTemplate.INDUSTRIAL: {
                'vlans': [
                    {'id': 100, 'name': 'PROCESS_CONTROL', 'priority': 7},
                    {'id': 200, 'name': 'SCADA', 'priority': 6},
                    {'id': 300, 'name': 'HMI', 'priority': 4},
                    {'id': 400, 'name': 'MAINTENANCE', 'priority': 2}
                ],
                'qos_classes': [
                    {'class': 7, 'name': 'Real-time Control', 'bandwidth': 50},
                    {'class': 5, 'name': 'Monitoring', 'bandwidth': 30},
                    {'class': 3, 'name': 'Logging', 'bandwidth': 20}
                ],
                'security': 'medium',
                'redundancy': 'ring'
            }
        }
    
    def analyze_requirements(self, requirements: NetworkRequirements) -> Dict[str, Any]:
        """Analyze requirements and recommend configuration"""
        analysis = {
            'recommended_switches': [],
            'topology': '',
            'concerns': [],
            'optimizations': []
        }
        
        # Calculate total bandwidth needed
        total_bandwidth_gbps = requirements.total_bandwidth_mbps / 1000
        
        # Recommend switch models
        if total_bandwidth_gbps > 10:
            analysis['recommended_switches'].append({
                'model': SwitchModel.LAN9692.value,
                'role': 'central',
                'quantity': 1
            })
            analysis['recommended_switches'].append({
                'model': SwitchModel.LAN9668.value,
                'role': 'zone',
                'quantity': max(2, requirements.num_sensors // 4)
            })
            analysis['topology'] = 'hierarchical-star'
        elif total_bandwidth_gbps > 4:
            analysis['recommended_switches'].append({
                'model': SwitchModel.LAN9668.value,
                'role': 'central',
                'quantity': 1
            })
            analysis['recommended_switches'].append({
                'model': SwitchModel.LAN9662.value,
                'role': 'edge',
                'quantity': requirements.num_sensors // 3
            })
            analysis['topology'] = 'distributed-star'
        else:
            analysis['recommended_switches'].append({
                'model': SwitchModel.KSZ9897.value,
                'role': 'standalone',
                'quantity': 1
            })
            analysis['topology'] = 'flat'
        
        # Check for concerns
        if requirements.max_latency_ms < 1.0 and not requirements.enable_tas:
            analysis['concerns'].append("Sub-millisecond latency requires TAS enabled")
        
        if requirements.redundancy_level > 2 and not requirements.enable_frer:
            analysis['concerns'].append("High redundancy requires FRER enabled")
        
        # Suggest optimizations
        if requirements.num_critical_streams > 5:
            analysis['optimizations'].append("Consider traffic shaping with CBS for critical streams")
        
        if requirements.num_sensors > 10:
            analysis['optimizations'].append("Implement sensor data aggregation to reduce bandwidth")
        
        return analysis
    
    def generate_base_config(self, switch_model: SwitchModel, 
                            hostname: str, 
                            management_ip: str) -> Dict[str, Any]:
        """Generate base switch configuration"""
        capabilities = self.switch_capabilities.get(switch_model, {})
        
        config = {
            'version': '2.0',
            'generated': datetime.now().isoformat(),
            'model': switch_model.value,
            'hostname': hostname,
            'system': {
                'hostname': hostname,
                'location': 'A2Z Vehicle',
                'contact': 'engineering@autoa2z.com',
                'description': f'{switch_model.value} TSN Switch'
            },
            'management': {
                'ip_address': management_ip,
                'netmask': '255.255.255.0',
                'gateway': str(ipaddress.ip_address(management_ip.split('/')[0]) + 1),
                'ssh': {'enabled': True, 'port': 22},
                'https': {'enabled': True, 'port': 443},
                'snmp': {
                    'enabled': True,
                    'version': 'v3',
                    'community': self._generate_secure_string('snmp')
                }
            },
            'interfaces': self._generate_interface_config(switch_model, capabilities.get('ports', 8)),
            'features': {
                'spanning_tree': {'mode': 'rstp', 'priority': 32768},
                'lldp': {'enabled': True, 'tx_interval': 30},
                'igmp_snooping': {'enabled': True, 'version': 3}
            }
        }
        
        return config
    
    def _generate_interface_config(self, model: SwitchModel, num_ports: int) -> List[Dict]:
        """Generate interface configurations"""
        interfaces = []
        
        for i in range(1, num_ports + 1):
            interface = {
                'name': f'GigabitEthernet1/{i}',
                'enabled': True,
                'speed': 'auto',
                'duplex': 'auto',
                'mtu': 9000,
                'description': '',
                'flow_control': False
            }
            
            # Assign roles based on port number
            if i <= 2:
                interface['description'] = 'Uplink/Trunk'
                interface['mode'] = 'trunk'
                interface['allowed_vlans'] = 'all'
                interface['speed'] = '1000'
            elif i <= num_ports - 2:
                interface['description'] = f'Access Port {i}'
                interface['mode'] = 'access'
                interface['vlan'] = 1
            else:
                interface['description'] = 'Management'
                interface['mode'] = 'access'
                interface['vlan'] = 99
            
            interfaces.append(interface)
        
        return interfaces
    
    def generate_vlan_config(self, template: ConfigTemplate, 
                           custom_vlans: Optional[List[VLANConfig]] = None) -> Dict[str, Any]:
        """Generate VLAN configuration"""
        if custom_vlans:
            vlans = custom_vlans
        else:
            template_config = self.config_templates.get(template, {})
            vlans = [
                VLANConfig(
                    vlan_id=v['id'],
                    name=v['name'],
                    priority=v['priority'],
                    ports=[]
                )
                for v in template_config.get('vlans', [])
            ]
        
        config = {
            'vlans': []
        }
        
        for vlan in vlans:
            vlan_config = {
                'id': vlan.vlan_id,
                'name': vlan.name,
                'state': 'active',
                'priority': vlan.priority,
                'ports': vlan.ports if vlan.ports else [],
                'igmp_snooping': True,
                'statistics': True
            }
            config['vlans'].append(vlan_config)
        
        return config
    
    def generate_frer_config(self, requirements: NetworkRequirements,
                           streams: Optional[List[FRERStreamConfig]] = None) -> Dict[str, Any]:
        """Generate FRER configuration"""
        if not requirements.enable_frer:
            return {'frer': {'enabled': False}}
        
        config = {
            'frer': {
                'enabled': True,
                'global': {
                    'mode': 'elimination-and-recovery',
                    'tag_format': 'standard',
                    'sequence_space': 65536,
                    'recovery_timeout': 100,  # milliseconds
                    'latent_error_detection': True
                },
                'streams': []
            }
        }
        
        # Use provided streams or generate default
        if not streams:
            streams = self._generate_default_streams(requirements)
        
        for stream in streams:
            stream_config = {
                'stream_id': stream.stream_id,
                'name': stream.name,
                'enabled': True,
                'bandwidth_kbps': stream.bandwidth_mbps * 1000,
                'redundancy': {
                    'paths': stream.redundancy_paths,
                    'elimination': True,
                    'recovery': True
                },
                'sequence': {
                    'recovery_window': stream.recovery_window,
                    'history_length': stream.history_length,
                    'reset_timeout': 1000
                },
                'paths': [],
                'statistics': True
            }
            
            # Generate paths
            for path_num in range(stream.redundancy_paths):
                path = {
                    'path_id': path_num + 1,
                    'priority': 0 if path_num == 0 else 1,
                    'ingress': stream.source_port,
                    'egress': stream.dest_ports[path_num % len(stream.dest_ports)]
                }
                stream_config['paths'].append(path)
            
            config['frer']['streams'].append(stream_config)
        
        return config
    
    def _generate_default_streams(self, requirements: NetworkRequirements) -> List[FRERStreamConfig]:
        """Generate default FRER streams based on requirements"""
        streams = []
        
        # Critical sensor streams
        for i in range(min(requirements.num_critical_streams, 4)):
            stream = FRERStreamConfig(
                stream_id=1001 + i,
                name=f'Critical_Stream_{i+1}',
                bandwidth_mbps=100,
                redundancy_paths=min(requirements.redundancy_level, 3),
                max_latency_ms=requirements.max_latency_ms,
                recovery_window=128,
                history_length=64,
                source_port='GigabitEthernet1/3',
                dest_ports=['GigabitEthernet1/5', 'GigabitEthernet1/6']
            )
            streams.append(stream)
        
        return streams
    
    def generate_qos_config(self, template: ConfigTemplate,
                          custom_qos: Optional[List[QoSConfig]] = None) -> Dict[str, Any]:
        """Generate QoS configuration"""
        config = {
            'qos': {
                'enabled': True,
                'mode': 'advanced',
                'trust': 'cos',
                'classes': []
            }
        }
        
        if custom_qos:
            qos_classes = custom_qos
        else:
            template_config = self.config_templates.get(template, {})
            qos_classes = []
            for qc in template_config.get('qos_classes', []):
                qos = QoSConfig(
                    class_id=qc['class'],
                    name=qc['name'],
                    priority=qc['class'],
                    bandwidth_percent=qc['bandwidth'],
                    cbs_enabled=qc['class'] >= 5,
                    idle_slope=qc['bandwidth'] * 10000,
                    send_slope=qc['bandwidth'] * -10000
                )
                qos_classes.append(qos)
        
        for qos in qos_classes:
            class_config = {
                'class': qos.class_id,
                'name': qos.name,
                'priority': qos.priority,
                'bandwidth': {
                    'percent': qos.bandwidth_percent,
                    'minimum_kbps': int(qos.bandwidth_percent * 10000),
                    'maximum_kbps': int(qos.bandwidth_percent * 10000 * 1.2)
                },
                'shaping': {
                    'cbs': {
                        'enabled': qos.cbs_enabled,
                        'idle_slope': qos.idle_slope,
                        'send_slope': qos.send_slope,
                        'high_credit': qos.idle_slope * 10,
                        'low_credit': qos.send_slope * 10
                    }
                },
                'scheduling': 'strict' if qos.priority >= 6 else 'wrr',
                'weight': 10 - qos.priority if qos.priority < 6 else 0
            }
            config['qos']['classes'].append(class_config)
        
        return config
    
    def generate_tas_config(self, requirements: NetworkRequirements) -> Dict[str, Any]:
        """Generate Time-Aware Shaper (802.1Qbv) configuration"""
        if not requirements.enable_tas:
            return {'tas': {'enabled': False}}
        
        config = {
            'tas': {
                'enabled': True,
                'admin_base_time': 0,
                'admin_cycle_time': 1000000,  # 1ms in nanoseconds
                'admin_control_list': []
            }
        }
        
        # Generate gate control list
        # Critical traffic gets first 60% of cycle
        critical_duration = 600000  # 600 microseconds
        control_duration = 250000   # 250 microseconds
        besteffort_duration = 150000  # 150 microseconds
        
        config['tas']['admin_control_list'] = [
            {
                'index': 0,
                'operation': 'set-gates',
                'gate_states': 0b11000000,  # Gates 7,6 open (critical)
                'time_interval': critical_duration
            },
            {
                'index': 1,
                'operation': 'set-gates',
                'gate_states': 0b00110000,  # Gates 5,4 open (control)
                'time_interval': control_duration
            },
            {
                'index': 2,
                'operation': 'set-gates',
                'gate_states': 0b00001111,  # Gates 3,2,1,0 open (best effort)
                'time_interval': besteffort_duration
            }
        ]
        
        return config
    
    def generate_gptp_config(self, requirements: NetworkRequirements,
                            role: str = 'boundary-clock') -> Dict[str, Any]:
        """Generate gPTP (802.1AS) configuration"""
        if not requirements.enable_gptp:
            return {'gptp': {'enabled': False}}
        
        config = {
            'gptp': {
                'enabled': True,
                'domain': 0,
                'mode': role,  # 'grandmaster', 'boundary-clock', 'ordinary-clock'
                'priority1': 128,
                'priority2': 128,
                'clock_class': 248,
                'clock_accuracy': 0xFE,  # Unknown
                'offset_scaled_log_variance': 0xFFFF,
                'current_utc_offset': 37,  # As of 2024
                'time_source': 0xA0,  # Internal oscillator
                'sync_interval': -3,  # 125ms
                'announce_interval': 0,  # 1 second
                'pdelay_interval': 0,  # 1 second
                'announce_receipt_timeout': 3
            }
        }
        
        if role == 'grandmaster':
            config['gptp']['priority1'] = 1
            config['gptp']['clock_class'] = 6  # GPS synchronized
            config['gptp']['time_source'] = 0x20  # GPS
        
        return config
    
    def generate_security_config(self, level: str = 'high') -> Dict[str, Any]:
        """Generate security configuration"""
        config = {
            'security': {
                'level': level,
                'authentication': {
                    'radius': {
                        'enabled': level in ['high', 'medium'],
                        'servers': [
                            {'ip': '192.168.100.10', 'port': 1812, 'secret': self._generate_secure_string('radius')}
                        ]
                    },
                    'local_users': [
                        {
                            'username': 'admin',
                            'password_hash': self._generate_password_hash('A2Z_Admin_2024!'),
                            'privilege': 15
                        },
                        {
                            'username': 'operator',
                            'password_hash': self._generate_password_hash('A2Z_Oper_2024!'),
                            'privilege': 7
                        }
                    ]
                },
                'access_control': {
                    'ssh': {
                        'version': 2,
                        'timeout': 120,
                        'max_sessions': 5,
                        'key_exchange': ['ecdh-sha2-nistp256', 'ecdh-sha2-nistp384'],
                        'ciphers': ['aes256-gcm', 'aes128-gcm'],
                        'macs': ['hmac-sha2-256', 'hmac-sha2-512']
                    },
                    'https': {
                        'tls_version': '1.3',
                        'ciphers': 'HIGH:!aNULL:!MD5:!3DES'
                    }
                },
                'port_security': {
                    'enabled': level == 'high',
                    'max_mac_addresses': 5,
                    'violation_action': 'shutdown',
                    'aging_time': 300
                },
                'dhcp_snooping': {
                    'enabled': level in ['high', 'medium'],
                    'trusted_ports': ['GigabitEthernet1/1', 'GigabitEthernet1/2']
                },
                'ip_source_guard': {
                    'enabled': level == 'high'
                },
                'storm_control': {
                    'broadcast': {'level': 10.0, 'action': 'shutdown'},
                    'multicast': {'level': 10.0, 'action': 'shutdown'},
                    'unicast': {'level': 50.0, 'action': 'trap'}
                }
            }
        }
        
        return config
    
    def _generate_secure_string(self, seed: str) -> str:
        """Generate secure string for passwords/secrets"""
        timestamp = datetime.now().isoformat()
        combined = f"{seed}_{timestamp}_A2Z"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _generate_password_hash(self, password: str) -> str:
        """Generate password hash"""
        salt = os.urandom(16).hex()
        combined = f"{password}{salt}"
        return hashlib.pbkdf2_hmac('sha256', combined.encode(), salt.encode(), 100000).hex()
    
    def generate_complete_config(self, 
                                switch_model: SwitchModel,
                                hostname: str,
                                management_ip: str,
                                requirements: NetworkRequirements,
                                template: ConfigTemplate = ConfigTemplate.AUTOMOTIVE) -> Dict[str, Any]:
        """Generate complete switch configuration"""
        
        # Analyze requirements
        analysis = self.analyze_requirements(requirements)
        
        # Generate all configuration sections
        config = self.generate_base_config(switch_model, hostname, management_ip)
        config['analysis'] = analysis
        config.update(self.generate_vlan_config(template))
        config.update(self.generate_frer_config(requirements))
        config.update(self.generate_qos_config(template))
        config.update(self.generate_tas_config(requirements))
        config.update(self.generate_gptp_config(requirements))
        config.update(self.generate_security_config('high'))
        
        # Add monitoring configuration
        config['monitoring'] = {
            'snmp': {
                'enabled': True,
                'trap_hosts': ['192.168.100.100'],
                'community': 'A2Z_monitoring'
            },
            'syslog': {
                'enabled': True,
                'servers': [
                    {'ip': '192.168.100.101', 'port': 514, 'protocol': 'tcp'}
                ],
                'level': 'informational'
            },
            'netflow': {
                'enabled': True,
                'version': 9,
                'collectors': [
                    {'ip': '192.168.100.102', 'port': 2055}
                ]
            }
        }
        
        # Add backup configuration
        config['backup'] = {
            'schedule': 'daily',
            'time': '02:00',
            'destination': 'tftp://192.168.100.103/configs/',
            'retention_days': 30
        }
        
        return config
    
    def export_config(self, config: Dict[str, Any], format: str = 'cli') -> str:
        """Export configuration to specified format"""
        if format == 'json':
            return json.dumps(config, indent=2)
        elif format == 'yaml':
            return yaml.dump(config, default_flow_style=False)
        elif format == 'cli':
            return self._generate_cli_commands(config)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_cli_commands(self, config: Dict[str, Any]) -> str:
        """Generate CLI commands from configuration"""
        commands = []
        commands.append("! A2Z TSN Switch Configuration")
        commands.append(f"! Generated: {config.get('generated', datetime.now().isoformat())}")
        commands.append(f"! Model: {config.get('model', 'Unknown')}")
        commands.append("!")
        commands.append("configure terminal")
        
        # System configuration
        system = config.get('system', {})
        commands.append(f"hostname {system.get('hostname', 'switch')}")
        
        # Management configuration
        mgmt = config.get('management', {})
        commands.append("interface vlan 1")
        commands.append(f"  ip address {mgmt.get('ip_address', '192.168.1.1')} {mgmt.get('netmask', '255.255.255.0')}")
        commands.append("  no shutdown")
        commands.append("exit")
        
        # VLAN configuration
        for vlan in config.get('vlans', {}).get('vlans', []):
            commands.append(f"vlan {vlan['id']}")
            commands.append(f"  name {vlan['name']}")
            commands.append("exit")
        
        # Interface configuration
        for interface in config.get('interfaces', []):
            commands.append(f"interface {interface['name']}")
            commands.append(f"  description {interface.get('description', '')}")
            if interface.get('mode') == 'trunk':
                commands.append("  switchport mode trunk")
                commands.append(f"  switchport trunk allowed vlan {interface.get('allowed_vlans', 'all')}")
            else:
                commands.append("  switchport mode access")
                commands.append(f"  switchport access vlan {interface.get('vlan', 1)}")
            commands.append(f"  speed {interface.get('speed', 'auto')}")
            commands.append(f"  duplex {interface.get('duplex', 'auto')}")
            commands.append("  no shutdown")
            commands.append("exit")
        
        # FRER configuration
        frer = config.get('frer', {})
        if frer.get('enabled'):
            commands.append("frer global enable")
            for stream in frer.get('streams', []):
                commands.append(f"frer stream {stream['stream_id']}")
                commands.append(f"  description {stream['name']}")
                commands.append(f"  bandwidth {stream['bandwidth_kbps']}")
                commands.append(f"  redundancy paths {stream['redundancy']['paths']}")
                commands.append(f"  sequence-recovery-window {stream['sequence']['recovery_window']}")
                commands.append(f"  sequence-history-length {stream['sequence']['history_length']}")
                commands.append("exit")
        
        # QoS configuration
        qos = config.get('qos', {})
        if qos.get('enabled'):
            commands.append("qos mode advanced")
            commands.append(f"qos trust {qos.get('trust', 'cos')}")
            for qos_class in qos.get('classes', []):
                commands.append(f"class-map class{qos_class['class']}")
                commands.append(f"  match cos {qos_class['class']}")
                commands.append("exit")
                commands.append(f"policy-map qos-policy")
                commands.append(f"  class class{qos_class['class']}")
                commands.append(f"    bandwidth percent {qos_class['bandwidth']['percent']}")
                if qos_class['shaping']['cbs']['enabled']:
                    commands.append(f"    cbs idle-slope {qos_class['shaping']['cbs']['idle_slope']}")
                    commands.append(f"    cbs send-slope {qos_class['shaping']['cbs']['send_slope']}")
                commands.append("exit")
        
        # gPTP configuration
        gptp = config.get('gptp', {})
        if gptp.get('enabled'):
            commands.append(f"ptp mode {gptp.get('mode', 'boundary-clock')}")
            commands.append(f"ptp domain {gptp.get('domain', 0)}")
            commands.append(f"ptp priority1 {gptp.get('priority1', 128)}")
            commands.append(f"ptp priority2 {gptp.get('priority2', 128)}")
        
        # Security configuration
        security = config.get('security', {})
        if security.get('port_security', {}).get('enabled'):
            commands.append("port-security enable")
        
        commands.append("end")
        commands.append("write memory")
        
        return "\n".join(commands)
    
    def validate_config(self, config: Dict[str, Any], switch_model: SwitchModel) -> Dict[str, Any]:
        """Validate configuration against switch capabilities"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        capabilities = self.switch_capabilities.get(switch_model, {})
        
        # Check port count
        interfaces = config.get('interfaces', [])
        if len(interfaces) > capabilities.get('ports', 0):
            validation['valid'] = False
            validation['errors'].append(f"Configuration has {len(interfaces)} ports but {switch_model.value} only supports {capabilities.get('ports')}")
        
        # Check TSN features
        if config.get('frer', {}).get('enabled') and '802.1CB' not in capabilities.get('tsn_features', []):
            validation['valid'] = False
            validation['errors'].append(f"{switch_model.value} does not support FRER (802.1CB)")
        
        if config.get('tas', {}).get('enabled') and '802.1Qbv' not in capabilities.get('tsn_features', []):
            validation['valid'] = False
            validation['errors'].append(f"{switch_model.value} does not support TAS (802.1Qbv)")
        
        # Check bandwidth allocation
        total_bandwidth = sum(
            s.get('bandwidth_kbps', 0) 
            for s in config.get('frer', {}).get('streams', [])
        ) / 1000000  # Convert to Gbps
        
        if total_bandwidth > capabilities.get('switching_capacity_gbps', 0):
            validation['warnings'].append(f"Total configured bandwidth ({total_bandwidth:.1f}Gbps) exceeds switching capacity")
        
        return validation


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='A2Z TSN Switch Configuration Generator')
    parser.add_argument('--model', type=str, required=True, choices=[m.value for m in SwitchModel],
                      help='Switch model')
    parser.add_argument('--hostname', type=str, required=True, help='Switch hostname')
    parser.add_argument('--ip', type=str, required=True, help='Management IP address')
    parser.add_argument('--template', type=str, default='automotive',
                      choices=[t.value for t in ConfigTemplate], help='Configuration template')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--format', type=str, default='cli',
                      choices=['cli', 'json', 'yaml'], help='Output format')
    
    # Requirements
    parser.add_argument('--bandwidth', type=int, default=1000, help='Total bandwidth (Mbps)')
    parser.add_argument('--streams', type=int, default=4, help='Number of critical streams')
    parser.add_argument('--sensors', type=int, default=8, help='Number of sensors')
    parser.add_argument('--controllers', type=int, default=4, help='Number of controllers')
    parser.add_argument('--redundancy', type=int, default=2, help='Redundancy level (1-3)')
    parser.add_argument('--latency', type=float, default=1.0, help='Max latency (ms)')
    
    args = parser.parse_args()
    
    # Create requirements
    requirements = NetworkRequirements(
        total_bandwidth_mbps=args.bandwidth,
        num_critical_streams=args.streams,
        num_sensors=args.sensors,
        num_controllers=args.controllers,
        redundancy_level=args.redundancy,
        max_latency_ms=args.latency
    )
    
    # Generate configuration
    generator = ConfigurationGenerator()
    
    print(f"Generating configuration for {args.model}...")
    config = generator.generate_complete_config(
        switch_model=SwitchModel(args.model),
        hostname=args.hostname,
        management_ip=args.ip,
        requirements=requirements,
        template=ConfigTemplate(args.template)
    )
    
    # Validate configuration
    validation = generator.validate_config(config, SwitchModel(args.model))
    
    if not validation['valid']:
        print("Configuration validation failed:")
        for error in validation['errors']:
            print(f"  ERROR: {error}")
        return 1
    
    if validation['warnings']:
        print("Configuration warnings:")
        for warning in validation['warnings']:
            print(f"  WARNING: {warning}")
    
    # Export configuration
    output = generator.export_config(config, args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Configuration saved to {args.output}")
    else:
        print("\n" + "=" * 80)
        print("GENERATED CONFIGURATION")
        print("=" * 80)
        print(output)
    
    # Print analysis
    print("\n" + "=" * 80)
    print("CONFIGURATION ANALYSIS")
    print("=" * 80)
    analysis = config.get('analysis', {})
    print(f"Recommended Topology: {analysis.get('topology')}")
    print("\nRecommended Switches:")
    for switch in analysis.get('recommended_switches', []):
        print(f"  - {switch['quantity']}x {switch['model']} ({switch['role']})")
    
    if analysis.get('concerns'):
        print("\nConcerns:")
        for concern in analysis['concerns']:
            print(f"  - {concern}")
    
    if analysis.get('optimizations'):
        print("\nOptimizations:")
        for opt in analysis['optimizations']:
            print(f"  - {opt}")


if __name__ == "__main__":
    main()