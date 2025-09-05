#!/usr/bin/env python3
"""
A2Z FRER Virtual Simulation Environment
Complete network simulation with virtual TSN switches
"""

import asyncio
import random
import time
import threading
import multiprocessing
from multiprocessing import Queue, Process, Manager
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import hashlib
import struct
import socket
import logging
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import simpy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FRER-Simulator')


class PacketType(Enum):
    DATA = "data"
    CONTROL = "control"
    SYNC = "sync"
    FRER = "frer"
    HEARTBEAT = "heartbeat"


class PortState(Enum):
    UP = "up"
    DOWN = "down"
    BLOCKED = "blocked"
    FORWARDING = "forwarding"


class PathState(Enum):
    ACTIVE = "active"
    STANDBY = "standby"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class Packet:
    """Network packet with FRER support"""
    id: str
    source: str
    destination: str
    type: PacketType
    size: int  # bytes
    priority: int  # 0-7
    stream_id: Optional[int] = None
    sequence_number: Optional[int] = None
    path_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    ttl: int = 64
    data: Optional[bytes] = None
    
    def calculate_rtag(self) -> bytes:
        """Calculate R-TAG for FRER"""
        if self.stream_id and self.sequence_number is not None:
            # R-TAG format: [Path ID: 4 bits][Sequence: 16 bits][CRC: 8 bits][Type: 4 bits]
            rtag_data = struct.pack(
                '>BHB',
                (self.path_id or 0) << 4,
                self.sequence_number & 0xFFFF,
                self.calculate_crc8()
            )
            return rtag_data
        return b''
    
    def calculate_crc8(self) -> int:
        """Calculate CRC-8 for packet integrity"""
        data = f"{self.id}{self.source}{self.destination}{self.sequence_number}".encode()
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ 0x07
                else:
                    crc <<= 1
            crc &= 0xFF
        return crc
    
    def duplicate(self, path_id: int) -> 'Packet':
        """Create duplicate packet for different path"""
        return Packet(
            id=f"{self.id}_p{path_id}",
            source=self.source,
            destination=self.destination,
            type=self.type,
            size=self.size,
            priority=self.priority,
            stream_id=self.stream_id,
            sequence_number=self.sequence_number,
            path_id=path_id,
            timestamp=self.timestamp,
            ttl=self.ttl,
            data=self.data
        )


@dataclass
class FRERStream:
    """FRER stream configuration"""
    stream_id: int
    name: str
    source_port: str
    dest_ports: List[str]
    bandwidth_mbps: int
    max_latency_ms: float
    redundancy_paths: int
    recovery_window_size: int = 128
    history_length: int = 64
    sequence_counter: int = field(default=0)
    received_sequences: Dict[int, float] = field(default_factory=dict)
    duplicate_count: int = field(default=0)
    lost_count: int = field(default=0)
    recovered_count: int = field(default=0)
    
    def get_next_sequence(self) -> int:
        """Get next sequence number"""
        self.sequence_counter = (self.sequence_counter + 1) & 0xFFFF
        return self.sequence_counter
    
    def check_duplicate(self, seq: int) -> bool:
        """Check if sequence is duplicate"""
        current_time = time.time()
        
        # Clean old sequences
        self.received_sequences = {
            s: t for s, t in self.received_sequences.items()
            if current_time - t < 1.0  # 1 second window
        }
        
        if seq in self.received_sequences:
            self.duplicate_count += 1
            return True
        
        self.received_sequences[seq] = current_time
        return False
    
    def check_sequence_recovery(self, seq: int) -> bool:
        """Check if packet is within recovery window"""
        if not self.received_sequences:
            return True
        
        last_seq = max(self.received_sequences.keys())
        diff = (seq - last_seq) & 0xFFFF
        
        if diff > 0x8000:  # Wrapped around
            diff = 0xFFFF - diff
        
        return diff <= self.recovery_window_size


class VirtualPort:
    """Virtual switch port"""
    
    def __init__(self, port_id: int, name: str, speed_mbps: int = 1000):
        self.id = port_id
        self.name = name
        self.speed_mbps = speed_mbps
        self.state = PortState.UP
        self.connected_to: Optional[str] = None
        self.vlan_id: Optional[int] = None
        self.priority = 0
        
        # Statistics
        self.tx_packets = 0
        self.rx_packets = 0
        self.tx_bytes = 0
        self.rx_bytes = 0
        self.dropped_packets = 0
        self.error_count = 0
        
        # QoS queues (8 priorities)
        self.queues = {i: [] for i in range(8)}
        self.queue_sizes = {i: 1000 for i in range(8)}  # packets
        
        # CBS parameters
        self.cbs_enabled = False
        self.idle_slope = 0
        self.send_slope = 0
        self.credit = 0
    
    def transmit(self, packet: Packet) -> bool:
        """Transmit packet through port"""
        if self.state != PortState.UP:
            self.dropped_packets += 1
            return False
        
        # Check queue space
        queue = self.queues[packet.priority]
        if len(queue) >= self.queue_sizes[packet.priority]:
            self.dropped_packets += 1
            return False
        
        # Add to queue
        queue.append(packet)
        
        # Update statistics
        self.tx_packets += 1
        self.tx_bytes += packet.size
        
        return True
    
    def receive(self, packet: Packet):
        """Receive packet on port"""
        if self.state != PortState.UP:
            self.dropped_packets += 1
            return
        
        self.rx_packets += 1
        self.rx_bytes += packet.size
    
    def process_queues(self, time_slice_us: int = 1000) -> List[Packet]:
        """Process QoS queues and return packets to send"""
        packets_to_send = []
        
        if self.cbs_enabled:
            # Credit Based Shaper processing
            self.credit += self.idle_slope * time_slice_us / 1000000
            self.credit = min(self.credit, self.idle_slope * 10)  # Max credit
            
            # Process high priority queues with CBS
            for priority in range(7, 4, -1):
                queue = self.queues[priority]
                while queue and self.credit > 0:
                    packet = queue.pop(0)
                    packets_to_send.append(packet)
                    self.credit += self.send_slope * packet.size / self.speed_mbps
        else:
            # Simple priority queuing
            for priority in range(7, -1, -1):
                queue = self.queues[priority]
                while queue:
                    packets_to_send.append(queue.pop(0))
                    if len(packets_to_send) >= 10:  # Batch size
                        return packets_to_send
        
        return packets_to_send


class VirtualSwitch:
    """Virtual TSN switch with FRER support"""
    
    def __init__(self, name: str, model: str, num_ports: int = 8):
        self.name = name
        self.model = model
        self.ports = {}
        self.mac_table = {}  # MAC -> Port mapping
        self.frer_streams = {}
        self.env: Optional[simpy.Environment] = None
        
        # Initialize ports
        for i in range(num_ports):
            port = VirtualPort(i, f"{name}_p{i}")
            self.ports[i] = port
        
        # Switch configuration
        self.switching_capacity_gbps = num_ports
        self.buffer_size_mb = 256
        self.latency_us = 5  # microseconds
        
        # FRER elimination table
        self.elimination_table = {}  # (stream_id, seq) -> timestamp
        
        # Statistics
        self.total_packets = 0
        self.frer_eliminated = 0
        self.broadcasts = 0
        
        # PTP/gPTP state
        self.clock_offset_ns = random.randint(-1000, 1000)
        self.clock_drift_ppb = random.randint(-10, 10)
        
        logger.info(f"Virtual switch {name} ({model}) initialized with {num_ports} ports")
    
    def connect_port(self, port_id: int, remote_switch: str, remote_port: int):
        """Connect port to another switch"""
        if port_id in self.ports:
            self.ports[port_id].connected_to = f"{remote_switch}:{remote_port}"
            logger.info(f"{self.name} port {port_id} connected to {remote_switch}:{remote_port}")
    
    def add_frer_stream(self, stream: FRERStream):
        """Add FRER stream configuration"""
        self.frer_streams[stream.stream_id] = stream
        logger.info(f"FRER stream {stream.stream_id} added to {self.name}")
    
    def process_packet(self, packet: Packet, ingress_port: int) -> List[Tuple[int, Packet]]:
        """Process incoming packet and return list of (egress_port, packet) tuples"""
        self.total_packets += 1
        forward_list = []
        
        # Check for FRER packet
        if packet.stream_id and packet.stream_id in self.frer_streams:
            stream = self.frer_streams[packet.stream_id]
            
            # Check for duplicate elimination
            if self.check_frer_duplicate(packet):
                self.frer_eliminated += 1
                logger.debug(f"FRER duplicate eliminated: stream {packet.stream_id}, seq {packet.sequence_number}")
                return []  # Drop duplicate
            
            # Check sequence recovery
            if not stream.check_sequence_recovery(packet.sequence_number):
                stream.lost_count += 1
                logger.warning(f"Packet out of recovery window: stream {packet.stream_id}, seq {packet.sequence_number}")
        
        # Learn MAC address
        self.mac_table[packet.source] = ingress_port
        
        # Forwarding decision
        if packet.destination in self.mac_table:
            # Unicast
            egress_port = self.mac_table[packet.destination]
            if egress_port != ingress_port:
                forward_list.append((egress_port, packet))
        else:
            # Broadcast/Unknown unicast
            self.broadcasts += 1
            for port_id, port in self.ports.items():
                if port_id != ingress_port and port.state == PortState.UP:
                    forward_list.append((port_id, packet))
        
        # Apply FRER replication if needed
        if packet.stream_id and packet.stream_id in self.frer_streams:
            stream = self.frer_streams[packet.stream_id]
            if len(forward_list) == 1 and stream.redundancy_paths > 1:
                # Replicate packet for redundant paths
                forward_list = self.replicate_frer_packet(packet, stream, forward_list[0][0])
        
        return forward_list
    
    def check_frer_duplicate(self, packet: Packet) -> bool:
        """Check if packet is FRER duplicate"""
        key = (packet.stream_id, packet.sequence_number)
        current_time = time.time()
        
        # Clean old entries
        self.elimination_table = {
            k: v for k, v in self.elimination_table.items()
            if current_time - v < 1.0
        }
        
        if key in self.elimination_table:
            return True
        
        self.elimination_table[key] = current_time
        return False
    
    def replicate_frer_packet(self, packet: Packet, stream: FRERStream, 
                            primary_port: int) -> List[Tuple[int, Packet]]:
        """Replicate packet for FRER redundancy"""
        replicated = []
        
        # Primary path
        replicated.append((primary_port, packet))
        
        # Find alternative paths
        alt_ports = [p for p in self.ports.keys() 
                    if p != primary_port and self.ports[p].state == PortState.UP]
        
        for i, alt_port in enumerate(alt_ports[:stream.redundancy_paths - 1]):
            dup_packet = packet.duplicate(path_id=i + 2)
            replicated.append((alt_port, dup_packet))
        
        return replicated
    
    def simulate_processing_delay(self) -> float:
        """Simulate switch processing delay"""
        base_delay = self.latency_us / 1000000  # Convert to seconds
        jitter = random.uniform(-0.1, 0.1) * base_delay
        return base_delay + jitter
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get switch statistics"""
        port_stats = {}
        for port_id, port in self.ports.items():
            port_stats[port_id] = {
                'state': port.state.value,
                'tx_packets': port.tx_packets,
                'rx_packets': port.rx_packets,
                'dropped': port.dropped_packets,
                'errors': port.error_count
            }
        
        return {
            'name': self.name,
            'model': self.model,
            'total_packets': self.total_packets,
            'frer_eliminated': self.frer_eliminated,
            'broadcasts': self.broadcasts,
            'mac_table_size': len(self.mac_table),
            'ports': port_stats
        }


class NetworkTopology:
    """Network topology manager"""
    
    def __init__(self):
        self.switches = {}
        self.links = []
        self.graph = nx.Graph()
        
    def add_switch(self, switch: VirtualSwitch):
        """Add switch to topology"""
        self.switches[switch.name] = switch
        self.graph.add_node(switch.name)
    
    def add_link(self, switch1: str, port1: int, switch2: str, port2: int,
                bandwidth_mbps: int = 1000, latency_ms: float = 0.1):
        """Add link between switches"""
        if switch1 in self.switches and switch2 in self.switches:
            self.switches[switch1].connect_port(port1, switch2, port2)
            self.switches[switch2].connect_port(port2, switch1, port1)
            
            self.links.append({
                'source': switch1,
                'source_port': port1,
                'dest': switch2,
                'dest_port': port2,
                'bandwidth': bandwidth_mbps,
                'latency': latency_ms
            })
            
            self.graph.add_edge(switch1, switch2, 
                              bandwidth=bandwidth_mbps,
                              latency=latency_ms)
            
            logger.info(f"Link added: {switch1}:{port1} <-> {switch2}:{port2}")
    
    def find_paths(self, source: str, dest: str) -> List[List[str]]:
        """Find all paths between source and destination"""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, dest))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def find_shortest_path(self, source: str, dest: str) -> Optional[List[str]]:
        """Find shortest path between nodes"""
        try:
            return nx.shortest_path(self.graph, source, dest)
        except nx.NetworkXNoPath:
            return None
    
    def visualize(self, show: bool = True):
        """Visualize network topology"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue',
                             node_size=3000, alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray',
                             width=2, alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10,
                              font_weight='bold')
        
        # Draw edge labels (bandwidth/latency)
        edge_labels = {}
        for edge in self.graph.edges():
            data = self.graph.get_edge_data(*edge)
            edge_labels[edge] = f"{data['bandwidth']}Mbps\n{data['latency']}ms"
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels,
                                    font_size=8)
        
        plt.title("A2Z TSN/FRER Network Topology")
        plt.axis('off')
        
        if show:
            plt.show()
        
        return plt.gcf()


class SimulationEngine:
    """Main simulation engine using SimPy"""
    
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.env = simpy.Environment()
        self.packet_store = simpy.Store(self.env)
        self.statistics = {
            'packets_sent': 0,
            'packets_received': 0,
            'packets_dropped': 0,
            'total_latency': 0,
            'max_latency': 0,
            'frer_recoveries': 0
        }
        
        # Attach environment to switches
        for switch in self.topology.switches.values():
            switch.env = self.env
    
    def packet_generator(self, stream: FRERStream, rate_pps: int):
        """Generate packets for a stream"""
        while True:
            # Generate packet
            seq = stream.get_next_sequence()
            packet = Packet(
                id=f"pkt_{stream.stream_id}_{seq}",
                source=stream.source_port,
                destination=stream.dest_ports[0],
                type=PacketType.FRER,
                size=1500,
                priority=7,  # High priority for FRER
                stream_id=stream.stream_id,
                sequence_number=seq
            )
            
            # Send packet
            self.send_packet(packet)
            self.statistics['packets_sent'] += 1
            
            # Wait for next packet
            yield self.env.timeout(1.0 / rate_pps)
    
    def send_packet(self, packet: Packet):
        """Send packet through network"""
        # Find source switch
        source_switch = None
        for switch_name, switch in self.topology.switches.items():
            if packet.source.startswith(switch_name):
                source_switch = switch
                break
        
        if not source_switch:
            logger.error(f"Source switch not found for packet {packet.id}")
            return
        
        # Process packet through switch
        self.env.process(self.process_packet_flow(packet, source_switch, 0))
    
    def process_packet_flow(self, packet: Packet, switch: VirtualSwitch, 
                           ingress_port: int):
        """Process packet flow through network"""
        # Simulate processing delay
        yield self.env.timeout(switch.simulate_processing_delay())
        
        # Process packet in switch
        forward_list = switch.process_packet(packet, ingress_port)
        
        for egress_port, fwd_packet in forward_list:
            port = switch.ports[egress_port]
            
            if port.transmit(fwd_packet):
                # Find connected switch
                if port.connected_to:
                    parts = port.connected_to.split(':')
                    next_switch_name = parts[0]
                    next_port = int(parts[1])
                    
                    if next_switch_name in self.topology.switches:
                        next_switch = self.topology.switches[next_switch_name]
                        
                        # Simulate link delay
                        link = self.find_link(switch.name, egress_port)
                        if link:
                            yield self.env.timeout(link['latency'] / 1000)
                        
                        # Continue processing in next switch
                        self.env.process(
                            self.process_packet_flow(fwd_packet, next_switch, next_port)
                        )
                else:
                    # Packet reached destination
                    self.statistics['packets_received'] += 1
                    latency = time.time() - fwd_packet.timestamp
                    self.statistics['total_latency'] += latency
                    self.statistics['max_latency'] = max(
                        self.statistics['max_latency'], latency
                    )
            else:
                self.statistics['packets_dropped'] += 1
    
    def find_link(self, switch_name: str, port: int) -> Optional[Dict]:
        """Find link information"""
        for link in self.topology.links:
            if (link['source'] == switch_name and link['source_port'] == port) or \
               (link['dest'] == switch_name and link['dest_port'] == port):
                return link
        return None
    
    def fault_injection(self, switch_name: str, port: int, duration: float):
        """Inject fault in network"""
        if switch_name in self.topology.switches:
            switch = self.topology.switches[switch_name]
            if port in switch.ports:
                original_state = switch.ports[port].state
                switch.ports[port].state = PortState.DOWN
                logger.warning(f"Fault injected: {switch_name}:{port} DOWN")
                
                yield self.env.timeout(duration)
                
                switch.ports[port].state = original_state
                logger.info(f"Fault cleared: {switch_name}:{port} UP")
                self.statistics['frer_recoveries'] += 1
    
    def run_simulation(self, duration: float, streams: List[FRERStream]):
        """Run simulation"""
        logger.info(f"Starting simulation for {duration} seconds")
        
        # Start packet generators
        for stream in streams:
            # Calculate packet rate from bandwidth
            rate_pps = (stream.bandwidth_mbps * 1000000) // (8 * 1500)
            self.env.process(self.packet_generator(stream, rate_pps))
        
        # Schedule some fault injections
        if duration > 10:
            self.env.process(self.fault_injection("central", 1, 2.0))
            
        # Run simulation
        self.env.run(until=duration)
        
        # Calculate statistics
        if self.statistics['packets_received'] > 0:
            avg_latency = self.statistics['total_latency'] / self.statistics['packets_received']
            packet_loss = (self.statistics['packets_sent'] - self.statistics['packets_received']) / self.statistics['packets_sent']
            
            logger.info("=" * 60)
            logger.info("SIMULATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"Duration: {duration} seconds")
            logger.info(f"Packets sent: {self.statistics['packets_sent']}")
            logger.info(f"Packets received: {self.statistics['packets_received']}")
            logger.info(f"Packets dropped: {self.statistics['packets_dropped']}")
            logger.info(f"Packet loss rate: {packet_loss:.4%}")
            logger.info(f"Average latency: {avg_latency*1000:.2f} ms")
            logger.info(f"Maximum latency: {self.statistics['max_latency']*1000:.2f} ms")
            logger.info(f"FRER recoveries: {self.statistics['frer_recoveries']}")
            
            # Per-switch statistics
            logger.info("\nSwitch Statistics:")
            for name, switch in self.topology.switches.items():
                stats = switch.get_statistics()
                logger.info(f"\n{name}:")
                logger.info(f"  Total packets: {stats['total_packets']}")
                logger.info(f"  FRER eliminated: {stats['frer_eliminated']}")
                logger.info(f"  Broadcasts: {stats['broadcasts']}")


class RealTimeVisualizer:
    """Real-time visualization of simulation"""
    
    def __init__(self, simulation: SimulationEngine):
        self.simulation = simulation
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('A2Z FRER Simulation - Real-time Monitor')
        
        # Data storage for plots
        self.time_data = []
        self.throughput_data = []
        self.latency_data = []
        self.packet_loss_data = []
        self.frer_recovery_data = []
        
    def update(self, frame):
        """Update visualization"""
        current_time = self.simulation.env.now
        self.time_data.append(current_time)
        
        # Calculate current metrics
        if self.simulation.statistics['packets_sent'] > 0:
            throughput = self.simulation.statistics['packets_received'] / max(current_time, 0.001)
            self.throughput_data.append(throughput)
            
            if self.simulation.statistics['packets_received'] > 0:
                avg_latency = self.simulation.statistics['total_latency'] / self.simulation.statistics['packets_received']
                self.latency_data.append(avg_latency * 1000)
            else:
                self.latency_data.append(0)
            
            loss_rate = (self.simulation.statistics['packets_sent'] - 
                        self.simulation.statistics['packets_received']) / \
                       self.simulation.statistics['packets_sent']
            self.packet_loss_data.append(loss_rate * 100)
        else:
            self.throughput_data.append(0)
            self.latency_data.append(0)
            self.packet_loss_data.append(0)
        
        self.frer_recovery_data.append(self.simulation.statistics.get('frer_recoveries', 0))
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot throughput
        self.axes[0, 0].plot(self.time_data, self.throughput_data, 'g-')
        self.axes[0, 0].set_title('Throughput (packets/sec)')
        self.axes[0, 0].set_xlabel('Time (s)')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot latency
        self.axes[0, 1].plot(self.time_data, self.latency_data, 'b-')
        self.axes[0, 1].set_title('Average Latency (ms)')
        self.axes[0, 1].set_xlabel('Time (s)')
        self.axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Target')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot packet loss
        self.axes[1, 0].plot(self.time_data, self.packet_loss_data, 'r-')
        self.axes[1, 0].set_title('Packet Loss (%)')
        self.axes[1, 0].set_xlabel('Time (s)')
        self.axes[1, 0].set_ylim([0, 5])
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot FRER recoveries
        self.axes[1, 1].plot(self.time_data, self.frer_recovery_data, 'm-')
        self.axes[1, 1].set_title('FRER Recoveries')
        self.axes[1, 1].set_xlabel('Time (s)')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def animate(self):
        """Start animation"""
        ani = FuncAnimation(self.fig, self.update, interval=1000, cache_frame_data=False)
        plt.show()
        return ani


def create_a2z_topology() -> NetworkTopology:
    """Create A2Z vehicle network topology"""
    topology = NetworkTopology()
    
    # Create switches
    central = VirtualSwitch("central", "LAN9692", 12)
    front = VirtualSwitch("front", "LAN9662", 8)
    rear = VirtualSwitch("rear", "LAN9662", 8)
    
    topology.add_switch(central)
    topology.add_switch(front)
    topology.add_switch(rear)
    
    # Create links (primary and backup)
    topology.add_link("central", 0, "front", 0, 1000, 0.1)  # Primary
    topology.add_link("central", 1, "front", 1, 1000, 0.1)  # Backup
    topology.add_link("central", 2, "rear", 0, 1000, 0.1)   # Primary
    topology.add_link("central", 3, "rear", 1, 1000, 0.1)   # Backup
    
    # Cross-link for additional redundancy
    topology.add_link("front", 2, "rear", 2, 1000, 0.2)
    
    return topology


def create_frer_streams() -> List[FRERStream]:
    """Create FRER stream configurations"""
    streams = [
        FRERStream(
            stream_id=1001,
            name="LiDAR System",
            source_port="front_sensor_1",
            dest_ports=["central_ecu_1"],
            bandwidth_mbps=100,
            max_latency_ms=1.0,
            redundancy_paths=2
        ),
        FRERStream(
            stream_id=1002,
            name="Camera Array",
            source_port="front_sensor_2",
            dest_ports=["central_ecu_1"],
            bandwidth_mbps=400,
            max_latency_ms=2.0,
            redundancy_paths=2
        ),
        FRERStream(
            stream_id=1003,
            name="Emergency Brake",
            source_port="central_controller_1",
            dest_ports=["front_actuator_1", "rear_actuator_1"],
            bandwidth_mbps=1,
            max_latency_ms=0.5,
            redundancy_paths=3
        ),
        FRERStream(
            stream_id=1004,
            name="Steering Control",
            source_port="central_controller_2",
            dest_ports=["front_actuator_2"],
            bandwidth_mbps=10,
            max_latency_ms=1.0,
            redundancy_paths=2
        )
    ]
    
    return streams


def main():
    """Main simulation execution"""
    print("=" * 60)
    print("A2Z FRER VIRTUAL SIMULATION ENVIRONMENT")
    print("=" * 60)
    
    # Create topology
    print("\nCreating network topology...")
    topology = create_a2z_topology()
    
    # Visualize topology
    print("Visualizing topology...")
    topology.visualize(show=False)
    
    # Create FRER streams
    print("\nConfiguring FRER streams...")
    streams = create_frer_streams()
    
    # Add streams to switches
    for stream in streams:
        for switch in topology.switches.values():
            switch.add_frer_stream(stream)
    
    # Create simulation engine
    print("\nInitializing simulation engine...")
    simulation = SimulationEngine(topology)
    
    # Create visualizer (optional)
    # visualizer = RealTimeVisualizer(simulation)
    
    # Run simulation
    print("\nRunning simulation...")
    simulation.run_simulation(duration=30.0, streams=streams)
    
    # Show final statistics
    print("\n" + "=" * 60)
    print("FINAL NETWORK STATE")
    print("=" * 60)
    
    for name, switch in topology.switches.items():
        stats = switch.get_statistics()
        print(f"\nSwitch: {name}")
        print(f"  Model: {switch.model}")
        print(f"  Total packets processed: {stats['total_packets']}")
        print(f"  FRER duplicates eliminated: {stats['frer_eliminated']}")
        print(f"  MAC table entries: {stats['mac_table_size']}")
        
        # Port statistics
        active_ports = sum(1 for p in stats['ports'].values() if p['state'] == 'up')
        total_tx = sum(p['tx_packets'] for p in stats['ports'].values())
        total_dropped = sum(p['dropped'] for p in stats['ports'].values())
        
        print(f"  Active ports: {active_ports}/{len(stats['ports'])}")
        print(f"  Total TX packets: {total_tx}")
        print(f"  Total dropped: {total_dropped}")
    
    # FRER stream statistics
    print("\n" + "=" * 60)
    print("FRER STREAM STATISTICS")
    print("=" * 60)
    
    for stream in streams:
        print(f"\nStream {stream.stream_id}: {stream.name}")
        print(f"  Sequence counter: {stream.sequence_counter}")
        print(f"  Duplicates detected: {stream.duplicate_count}")
        print(f"  Packets lost: {stream.lost_count}")
        print(f"  Packets recovered: {stream.recovered_count}")
    
    print("\nSimulation completed successfully!")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()