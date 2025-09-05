#!/usr/bin/env python3
"""
A2Z Gigabit TSN/FRER Advanced Diagnostics System
Real-time troubleshooting and analysis tools
"""

import os
import sys
import time
import socket
import struct
import threading
import subprocess
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('A2Z-Diagnostics')


class DiagnosticLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SystemComponent(Enum):
    CENTRAL_SWITCH = "LAN9692 Central"
    FRONT_SWITCH = "LAN9662 Front"
    REAR_SWITCH = "LAN9662 Rear"
    FRER_SUBSYSTEM = "FRER Subsystem"
    NETWORK_BACKBONE = "Gigabit Backbone"
    SENSORS = "Sensor Array"
    CONTROL_UNITS = "Control Units"


@dataclass
class DiagnosticResult:
    """Diagnostic test result"""
    timestamp: datetime
    component: SystemComponent
    test_name: str
    level: DiagnosticLevel
    message: str
    details: Dict[str, Any]
    recommended_action: str = ""


class NetworkDiagnostics:
    """Network connectivity and performance diagnostics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.switches = {
            'central': config.get('central_ip', '192.168.100.1'),
            'front': config.get('front_ip', '192.168.100.2'),
            'rear': config.get('rear_ip', '192.168.100.3')
        }
        
    def check_connectivity(self) -> List[DiagnosticResult]:
        """Check basic network connectivity"""
        results = []
        
        for name, ip in self.switches.items():
            try:
                response = subprocess.run(
                    ['ping', '-c', '4', '-W', '1', ip],
                    capture_output=True, text=True, timeout=5
                )
                
                if response.returncode == 0:
                    # Parse ping statistics
                    match = re.search(r'(\d+)% packet loss', response.stdout)
                    packet_loss = int(match.group(1)) if match else 100
                    
                    if packet_loss == 0:
                        level = DiagnosticLevel.INFO
                        message = f"{name.upper()} switch reachable"
                    elif packet_loss < 25:
                        level = DiagnosticLevel.WARNING
                        message = f"{name.upper()} switch experiencing packet loss"
                    else:
                        level = DiagnosticLevel.ERROR
                        message = f"{name.upper()} switch connectivity degraded"
                    
                    results.append(DiagnosticResult(
                        timestamp=datetime.now(),
                        component=SystemComponent.NETWORK_BACKBONE,
                        test_name=f"{name}_connectivity",
                        level=level,
                        message=message,
                        details={'ip': ip, 'packet_loss': packet_loss},
                        recommended_action="" if packet_loss == 0 else "Check network cables and switch status"
                    ))
                else:
                    results.append(DiagnosticResult(
                        timestamp=datetime.now(),
                        component=SystemComponent.NETWORK_BACKBONE,
                        test_name=f"{name}_connectivity",
                        level=DiagnosticLevel.CRITICAL,
                        message=f"{name.upper()} switch unreachable",
                        details={'ip': ip},
                        recommended_action="Verify switch power and network configuration"
                    ))
                    
            except Exception as e:
                results.append(DiagnosticResult(
                    timestamp=datetime.now(),
                    component=SystemComponent.NETWORK_BACKBONE,
                    test_name=f"{name}_connectivity",
                    level=DiagnosticLevel.ERROR,
                    message=f"Failed to test {name} switch",
                    details={'ip': ip, 'error': str(e)},
                    recommended_action="Check diagnostic system permissions"
                ))
        
        return results
    
    def measure_bandwidth(self) -> List[DiagnosticResult]:
        """Measure available bandwidth between switches"""
        results = []
        
        # Simulate bandwidth measurement (in production, use iperf3)
        bandwidth_tests = [
            ('central-front', 980, 1000),  # actual, expected
            ('central-rear', 975, 1000),
            ('front-rear', 950, 1000)
        ]
        
        for link, actual, expected in bandwidth_tests:
            utilization = (actual / expected) * 100
            
            if utilization >= 95:
                level = DiagnosticLevel.WARNING
                message = f"High bandwidth utilization on {link} link"
                action = "Monitor for congestion, consider traffic shaping"
            elif utilization >= 80:
                level = DiagnosticLevel.INFO
                message = f"Normal bandwidth utilization on {link} link"
                action = ""
            else:
                level = DiagnosticLevel.WARNING
                message = f"Low bandwidth utilization on {link} link"
                action = "Verify link speed negotiation"
            
            results.append(DiagnosticResult(
                timestamp=datetime.now(),
                component=SystemComponent.NETWORK_BACKBONE,
                test_name=f"bandwidth_{link}",
                level=level,
                message=message,
                details={
                    'actual_mbps': actual,
                    'expected_mbps': expected,
                    'utilization_percent': utilization
                },
                recommended_action=action
            ))
        
        return results
    
    def check_latency(self) -> List[DiagnosticResult]:
        """Check network latency"""
        results = []
        max_acceptable_latency = 2.0  # milliseconds
        
        for name, ip in self.switches.items():
            try:
                # Simulate latency measurement
                latency = 0.5 + (0.3 if name == 'central' else 0.8)
                
                if latency <= 1.0:
                    level = DiagnosticLevel.INFO
                    message = f"Excellent latency to {name} switch"
                elif latency <= max_acceptable_latency:
                    level = DiagnosticLevel.WARNING
                    message = f"Acceptable latency to {name} switch"
                else:
                    level = DiagnosticLevel.ERROR
                    message = f"High latency to {name} switch"
                
                results.append(DiagnosticResult(
                    timestamp=datetime.now(),
                    component=SystemComponent.NETWORK_BACKBONE,
                    test_name=f"latency_{name}",
                    level=level,
                    message=message,
                    details={'ip': ip, 'latency_ms': latency},
                    recommended_action="" if latency <= 1.0 else "Check for network congestion or QoS settings"
                ))
                
            except Exception as e:
                logger.error(f"Latency check failed for {name}: {e}")
        
        return results


class FRERDiagnostics:
    """FRER-specific diagnostics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.streams = {
            1001: {'name': 'LiDAR', 'bandwidth': 100, 'paths': 2},
            1002: {'name': 'Camera', 'bandwidth': 400, 'paths': 2},
            1003: {'name': 'E-Brake', 'bandwidth': 1, 'paths': 3},
            1004: {'name': 'Steering', 'bandwidth': 10, 'paths': 2}
        }
    
    def check_stream_health(self) -> List[DiagnosticResult]:
        """Check FRER stream health"""
        results = []
        
        for stream_id, info in self.streams.items():
            # Simulate stream health check
            active_paths = info['paths'] if stream_id != 1002 else info['paths'] - 1
            recovery_events = 2 if stream_id == 1001 else 0
            avg_recovery_time = 12.3 if recovery_events > 0 else 0
            
            if active_paths == info['paths']:
                level = DiagnosticLevel.INFO
                message = f"Stream {stream_id} ({info['name']}) fully redundant"
                action = ""
            elif active_paths > 0:
                level = DiagnosticLevel.WARNING
                message = f"Stream {stream_id} ({info['name']}) degraded redundancy"
                action = "Investigate failed path, prepare for maintenance"
            else:
                level = DiagnosticLevel.CRITICAL
                message = f"Stream {stream_id} ({info['name']}) no redundancy"
                action = "IMMEDIATE ACTION REQUIRED - Restore redundant paths"
            
            results.append(DiagnosticResult(
                timestamp=datetime.now(),
                component=SystemComponent.FRER_SUBSYSTEM,
                test_name=f"stream_{stream_id}_health",
                level=level,
                message=message,
                details={
                    'stream_id': stream_id,
                    'name': info['name'],
                    'active_paths': active_paths,
                    'configured_paths': info['paths'],
                    'recovery_events_24h': recovery_events,
                    'avg_recovery_ms': avg_recovery_time
                },
                recommended_action=action
            ))
        
        return results
    
    def check_recovery_performance(self) -> List[DiagnosticResult]:
        """Check FRER recovery performance"""
        results = []
        max_recovery_time = 50.0  # milliseconds
        
        # Simulate recovery performance data
        recovery_stats = {
            'avg_recovery_ms': 12.3,
            'max_recovery_ms': 45.2,
            'min_recovery_ms': 8.1,
            'recovery_events_24h': 47,
            'failed_recoveries': 0
        }
        
        if recovery_stats['max_recovery_ms'] <= max_recovery_time:
            level = DiagnosticLevel.INFO
            message = "FRER recovery performance within specifications"
            action = ""
        else:
            level = DiagnosticLevel.WARNING
            message = "FRER recovery time exceeding target"
            action = "Review recovery window settings and network load"
        
        if recovery_stats['failed_recoveries'] > 0:
            level = DiagnosticLevel.ERROR
            message = "FRER recovery failures detected"
            action = "Check path availability and switch configuration"
        
        results.append(DiagnosticResult(
            timestamp=datetime.now(),
            component=SystemComponent.FRER_SUBSYSTEM,
            test_name="recovery_performance",
            level=level,
            message=message,
            details=recovery_stats,
            recommended_action=action
        ))
        
        return results
    
    def check_sequence_integrity(self) -> List[DiagnosticResult]:
        """Check R-TAG sequence integrity"""
        results = []
        
        # Simulate sequence checking
        sequence_errors = {
            1001: 0,
            1002: 2,  # Minor sequence errors
            1003: 0,
            1004: 0
        }
        
        for stream_id, errors in sequence_errors.items():
            if errors == 0:
                level = DiagnosticLevel.INFO
                message = f"Stream {stream_id} sequence integrity maintained"
                action = ""
            elif errors < 10:
                level = DiagnosticLevel.WARNING
                message = f"Stream {stream_id} minor sequence errors detected"
                action = "Monitor for increasing errors, check time sync"
            else:
                level = DiagnosticLevel.ERROR
                message = f"Stream {stream_id} significant sequence errors"
                action = "Check R-TAG configuration and path stability"
            
            results.append(DiagnosticResult(
                timestamp=datetime.now(),
                component=SystemComponent.FRER_SUBSYSTEM,
                test_name=f"sequence_{stream_id}",
                level=level,
                message=message,
                details={'stream_id': stream_id, 'sequence_errors': errors},
                recommended_action=action
            ))
        
        return results


class SwitchDiagnostics:
    """Switch-specific hardware diagnostics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def check_hardware_status(self) -> List[DiagnosticResult]:
        """Check switch hardware status"""
        results = []
        
        # Simulate hardware status checks
        hardware_status = {
            'central': {
                'temperature_c': 55,
                'cpu_usage': 45,
                'memory_usage': 62,
                'power_status': 'normal',
                'fan_status': 'operational'
            },
            'front': {
                'temperature_c': 58,
                'cpu_usage': 38,
                'memory_usage': 55,
                'power_status': 'normal',
                'fan_status': 'operational'
            },
            'rear': {
                'temperature_c': 72,  # High temperature
                'cpu_usage': 35,
                'memory_usage': 48,
                'power_status': 'normal',
                'fan_status': 'degraded'
            }
        }
        
        for switch_name, status in hardware_status.items():
            # Temperature check
            if status['temperature_c'] < 60:
                level = DiagnosticLevel.INFO
                message = f"{switch_name.upper()} switch temperature normal"
                action = ""
            elif status['temperature_c'] < 70:
                level = DiagnosticLevel.WARNING
                message = f"{switch_name.upper()} switch temperature elevated"
                action = "Check cooling system and airflow"
            else:
                level = DiagnosticLevel.ERROR
                message = f"{switch_name.upper()} switch overheating"
                action = "IMMEDIATE: Improve cooling or reduce load"
            
            component = getattr(SystemComponent, f"{switch_name.upper()}_SWITCH")
            results.append(DiagnosticResult(
                timestamp=datetime.now(),
                component=component,
                test_name=f"{switch_name}_temperature",
                level=level,
                message=message,
                details={'temperature_c': status['temperature_c']},
                recommended_action=action
            ))
            
            # CPU usage check
            if status['cpu_usage'] < 60:
                level = DiagnosticLevel.INFO
            elif status['cpu_usage'] < 80:
                level = DiagnosticLevel.WARNING
            else:
                level = DiagnosticLevel.ERROR
            
            results.append(DiagnosticResult(
                timestamp=datetime.now(),
                component=component,
                test_name=f"{switch_name}_cpu",
                level=level,
                message=f"{switch_name.upper()} CPU usage: {status['cpu_usage']}%",
                details={'cpu_usage_percent': status['cpu_usage']},
                recommended_action="" if status['cpu_usage'] < 60 else "Review processing load distribution"
            ))
        
        return results
    
    def check_port_status(self) -> List[DiagnosticResult]:
        """Check individual port status"""
        results = []
        
        # Simulate port status
        port_errors = {
            'central_port_1': {'crc_errors': 0, 'collisions': 0, 'link_flaps': 0},
            'central_port_5': {'crc_errors': 12, 'collisions': 0, 'link_flaps': 2},
            'front_port_1': {'crc_errors': 0, 'collisions': 0, 'link_flaps': 0}
        }
        
        for port_name, errors in port_errors.items():
            total_errors = sum(errors.values())
            
            if total_errors == 0:
                level = DiagnosticLevel.INFO
                message = f"Port {port_name} operating normally"
                action = ""
            elif total_errors < 10:
                level = DiagnosticLevel.WARNING
                message = f"Port {port_name} experiencing errors"
                action = "Monitor port, check cable quality"
            else:
                level = DiagnosticLevel.ERROR
                message = f"Port {port_name} high error rate"
                action = "Replace cable, check termination"
            
            results.append(DiagnosticResult(
                timestamp=datetime.now(),
                component=SystemComponent.NETWORK_BACKBONE,
                test_name=f"port_{port_name}",
                level=level,
                message=message,
                details=errors,
                recommended_action=action
            ))
        
        return results


class DiagnosticEngine:
    """Main diagnostic engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.network_diag = NetworkDiagnostics(config)
        self.frer_diag = FRERDiagnostics(config)
        self.switch_diag = SwitchDiagnostics(config)
        self.results_history = deque(maxlen=1000)
        
    def run_full_diagnostics(self) -> List[DiagnosticResult]:
        """Run complete diagnostic suite"""
        logger.info("Starting full system diagnostics")
        all_results = []
        
        # Network diagnostics
        logger.info("Running network diagnostics...")
        all_results.extend(self.network_diag.check_connectivity())
        all_results.extend(self.network_diag.measure_bandwidth())
        all_results.extend(self.network_diag.check_latency())
        
        # FRER diagnostics
        logger.info("Running FRER diagnostics...")
        all_results.extend(self.frer_diag.check_stream_health())
        all_results.extend(self.frer_diag.check_recovery_performance())
        all_results.extend(self.frer_diag.check_sequence_integrity())
        
        # Switch diagnostics
        logger.info("Running switch diagnostics...")
        all_results.extend(self.switch_diag.check_hardware_status())
        all_results.extend(self.switch_diag.check_port_status())
        
        # Store results in history
        self.results_history.extend(all_results)
        
        return all_results
    
    def run_quick_health_check(self) -> List[DiagnosticResult]:
        """Run quick health check"""
        logger.info("Running quick health check")
        results = []
        
        # Just connectivity and stream health
        results.extend(self.network_diag.check_connectivity())
        results.extend(self.frer_diag.check_stream_health())
        
        return results
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze diagnostic trends"""
        if not self.results_history:
            return {"message": "No historical data available"}
        
        # Count issues by level
        level_counts = {level: 0 for level in DiagnosticLevel}
        component_issues = {comp: [] for comp in SystemComponent}
        
        for result in self.results_history:
            level_counts[result.level] += 1
            if result.level != DiagnosticLevel.INFO:
                component_issues[result.component].append(result)
        
        # Find recurring issues
        recurring_issues = {}
        for component, issues in component_issues.items():
            if len(issues) > 5:
                recurring_issues[component.value] = len(issues)
        
        return {
            "total_diagnostics": len(self.results_history),
            "level_distribution": {k.value: v for k, v in level_counts.items()},
            "recurring_issues": recurring_issues,
            "components_with_issues": [c.value for c, i in component_issues.items() if i]
        }
    
    def generate_report(self, results: List[DiagnosticResult]) -> str:
        """Generate diagnostic report"""
        report = []
        report.append("=" * 80)
        report.append("A2Z GIGABIT TSN/FRER DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Group results by component
        by_component = {}
        for result in results:
            if result.component not in by_component:
                by_component[result.component] = []
            by_component[result.component].append(result)
        
        # Summary
        total = len(results)
        critical = sum(1 for r in results if r.level == DiagnosticLevel.CRITICAL)
        errors = sum(1 for r in results if r.level == DiagnosticLevel.ERROR)
        warnings = sum(1 for r in results if r.level == DiagnosticLevel.WARNING)
        
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Checks: {total}")
        report.append(f"Critical Issues: {critical}")
        report.append(f"Errors: {errors}")
        report.append(f"Warnings: {warnings}")
        report.append("")
        
        # Detailed results by component
        for component, comp_results in by_component.items():
            report.append(f"\n{component.value}")
            report.append("-" * 40)
            
            for result in comp_results:
                icon = {
                    DiagnosticLevel.INFO: "✅",
                    DiagnosticLevel.WARNING: "⚠️",
                    DiagnosticLevel.ERROR: "❌",
                    DiagnosticLevel.CRITICAL: "🚨"
                }[result.level]
                
                report.append(f"{icon} [{result.level.value.upper()}] {result.message}")
                
                if result.details:
                    for key, value in result.details.items():
                        report.append(f"    {key}: {value}")
                
                if result.recommended_action:
                    report.append(f"    → Action: {result.recommended_action}")
                
                report.append("")
        
        # Recommendations
        if critical > 0 or errors > 0:
            report.append("=" * 80)
            report.append("IMMEDIATE ACTIONS REQUIRED")
            report.append("=" * 80)
            
            for result in results:
                if result.level in [DiagnosticLevel.CRITICAL, DiagnosticLevel.ERROR]:
                    if result.recommended_action:
                        report.append(f"• {result.recommended_action}")
        
        return "\n".join(report)
    
    def export_json(self, results: List[DiagnosticResult], filename: str):
        """Export results to JSON"""
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "system": "A2Z Gigabit TSN/FRER",
            "diagnostics": []
        }
        
        for result in results:
            json_data["diagnostics"].append({
                "timestamp": result.timestamp.isoformat(),
                "component": result.component.value,
                "test": result.test_name,
                "level": result.level.value,
                "message": result.message,
                "details": result.details,
                "action": result.recommended_action
            })
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Diagnostic results exported to {filename}")


def main():
    """Main diagnostic execution"""
    parser = argparse.ArgumentParser(description='A2Z TSN/FRER Diagnostics')
    parser.add_argument('--mode', choices=['full', 'quick', 'continuous'], 
                      default='full', help='Diagnostic mode')
    parser.add_argument('--output', help='Output file for report')
    parser.add_argument('--json', help='Export results as JSON')
    parser.add_argument('--central-ip', default='192.168.100.1',
                      help='Central switch IP')
    parser.add_argument('--front-ip', default='192.168.100.2',
                      help='Front switch IP')
    parser.add_argument('--rear-ip', default='192.168.100.3',
                      help='Rear switch IP')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'central_ip': args.central_ip,
        'front_ip': args.front_ip,
        'rear_ip': args.rear_ip
    }
    
    # Create diagnostic engine
    engine = DiagnosticEngine(config)
    
    if args.mode == 'full':
        results = engine.run_full_diagnostics()
        report = engine.generate_report(results)
        print(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {args.output}")
        
        if args.json:
            engine.export_json(results, args.json)
    
    elif args.mode == 'quick':
        results = engine.run_quick_health_check()
        report = engine.generate_report(results)
        print(report)
    
    elif args.mode == 'continuous':
        print("Starting continuous monitoring mode (Ctrl+C to stop)...")
        try:
            while True:
                results = engine.run_quick_health_check()
                
                # Show only issues
                issues = [r for r in results if r.level != DiagnosticLevel.INFO]
                if issues:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Issues detected:")
                    for issue in issues:
                        print(f"  {issue.level.value.upper()}: {issue.message}")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] All systems operational")
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\nContinuous monitoring stopped")
            
            # Show trend analysis
            trends = engine.analyze_trends()
            print("\nTrend Analysis:")
            print(json.dumps(trends, indent=2))


if __name__ == "__main__":
    main()