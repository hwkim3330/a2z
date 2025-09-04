// A2Z Autonomous Vehicle Network Monitoring System
// Real-time FRER performance and vehicle telemetry dashboard

class A2ZNetworkMonitor {
    constructor() {
        this.wsConnection = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        this.frerData = {
            availability: 99.99,
            recoveryTime: 12.3,
            activeStreams: 4,
            networkUsage: "481/1000",
            pathFailures: 0,
            history: []
        };
        
        this.vehicleData = {
            roiiCount: 7,
            coiiCount: 4,
            totalBandwidth: "481 Mbps",
            safetyScore: 99.97,
            systems: {
                lidar: { status: 'healthy', throughput: '100 Mbps', latency: '8.2ms' },
                camera: { status: 'healthy', throughput: '400 Mbps', fps: '30' },
                emergency: { status: 'healthy', throughput: '1 Mbps', response: '38ms' },
                steering: { status: 'healthy', throughput: '10 Mbps', latency: '15ms' }
            }
        };
        
        this.performanceChart = null;
        this.alertHistory = [];
        
        this.initializeWebSocket();
        this.initializeCharts();
        this.setupEventListeners();
        this.startDataSimulation(); // For demo purposes
    }
    
    initializeWebSocket() {
        try {
            // In production, this would connect to actual A2Z monitoring server
            this.wsConnection = new WebSocket('ws://localhost:8080/a2z-monitor');
            
            this.wsConnection.onopen = (event) => {
                console.log('Connected to A2Z monitoring server');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');
            };
            
            this.wsConnection.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleIncomingData(data);
            };
            
            this.wsConnection.onclose = (event) => {
                console.log('Disconnected from A2Z monitoring server');
                this.updateConnectionStatus('disconnected');
                this.attemptReconnection();
            };
            
            this.wsConnection.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('error');
            };
            
        } catch (error) {
            console.log('WebSocket not available, using simulation mode');
            this.updateConnectionStatus('simulation');
        }
    }
    
    handleIncomingData(data) {
        switch(data.type) {
            case 'frer_metrics':
                this.updateFRERMetrics(data.payload);
                break;
            case 'vehicle_telemetry':
                this.updateVehicleStatus(data.payload);
                break;
            case 'system_alert':
                this.handleSystemAlert(data.payload);
                break;
            case 'network_topology':
                this.updateNetworkTopology(data.payload);
                break;
            default:
                console.warn('Unknown data type:', data.type);
        }
    }
    
    updateFRERMetrics(metrics) {
        this.frerData = { ...this.frerData, ...metrics };
        
        // Update FRER availability gauge
        const availabilityGauge = document.getElementById('frer-availability');
        const percentage = this.frerData.availability;
        
        if (availabilityGauge) {
            const rotation = (percentage / 100) * 360;
            availabilityGauge.style.background = 
                `conic-gradient(${this.getAvailabilityColor(percentage)} 0deg, ${this.getAvailabilityColor(percentage)} ${rotation}deg, #f44336 ${rotation}deg)`;
        }
        
        // Update metric displays
        this.updateElement('frer-percent', percentage.toFixed(2));
        this.updateElement('recovery-time', `${this.frerData.recoveryTime}ms`);
        this.updateElement('active-streams', this.frerData.activeStreams.toString());
        this.updateElement('network-usage', this.frerData.networkUsage + ' Mbps');
        this.updateElement('path-failures', this.frerData.pathFailures.toString());
        
        // Add to history for charting
        this.frerData.history.push({
            timestamp: new Date(),
            availability: percentage,
            recoveryTime: this.frerData.recoveryTime,
            throughput: this.frerData.replicationRate
        });
        
        // Keep only last 100 data points
        if (this.frerData.history.length > 100) {
            this.frerData.history.shift();
        }
        
        // Update performance chart
        this.updatePerformanceChart();
        
        // Check for alerts
        this.checkFRERAlerts(metrics);
    }
    
    updateVehicleStatus(telemetry) {
        this.vehicleData = { ...this.vehicleData, ...telemetry };
        
        // Update fleet overview
        this.updateElement('roii-count', this.vehicleData.roiiCount.toString());
        this.updateElement('coii-count', this.vehicleData.coiiCount.toString());
        this.updateElement('total-bandwidth', this.vehicleData.totalBandwidth);
        this.updateElement('safety-score', `${this.vehicleData.safetyScore.toFixed(2)}%`);
        
        // Update system status indicators
        Object.keys(this.vehicleData.systems).forEach(system => {
            this.updateSystemStatus(system, this.vehicleData.systems[system]);
        });
    }
    
    updateSystemStatus(systemName, systemData) {
        const systemElement = document.getElementById(`${systemName}-system`);
        if (systemElement) {
            // Remove existing status classes
            systemElement.classList.remove('healthy', 'warning', 'critical');
            // Add current status class
            systemElement.classList.add(systemData.status);
            
            // Update system metrics
            const metricElement = systemElement.querySelector('.system-metric');
            const latencyElement = systemElement.querySelector('.system-latency');
            
            if (metricElement && systemData.throughput) {
                metricElement.textContent = systemData.throughput;
            }
            
            if (latencyElement) {
                if (systemData.latency) {
                    latencyElement.textContent = `Latency: ${systemData.latency}`;
                } else if (systemData.fps) {
                    latencyElement.textContent = `FPS: ${systemData.fps}`;
                } else if (systemData.response) {
                    latencyElement.textContent = `Response: ${systemData.response}`;
                }
            }
        }
    }
    
    handleSystemAlert(alert) {
        this.alertHistory.unshift({
            id: Date.now(),
            timestamp: new Date(alert.timestamp || Date.now()),
            level: alert.level || 'info',
            source: alert.source || 'System',
            message: alert.message,
            vehicleId: alert.vehicleId,
            resolved: false
        });
        
        // Keep only last 50 alerts
        if (this.alertHistory.length > 50) {
            this.alertHistory.pop();
        }
        
        this.renderAlerts();
        
        // Play sound for critical alerts
        if (alert.level === 'critical') {
            this.playAlertSound();
        }
        
        // Show browser notification for important alerts
        if (alert.level === 'critical' || alert.level === 'warning') {
            this.showBrowserNotification(alert);
        }
    }
    
    initializeCharts() {
        const ctx = document.getElementById('performance-chart');
        if (ctx) {
            this.performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'FRER Recovery Time (ms)',
                        data: [],
                        borderColor: '#2a5298',
                        backgroundColor: 'rgba(42, 82, 152, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'A2Z Network Performance Metrics'
                        },
                        legend: {
                            display: true
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Recovery Time (ms)'
                            },
                            beginAtZero: true
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    }
                }
            });
        }
    }
    
    updatePerformanceChart() {
        if (!this.performanceChart || this.frerData.history.length === 0) return;
        
        const metricType = document.getElementById('metric-selector')?.value || 'latency';
        const timeRange = document.getElementById('time-range')?.value || '1h';
        
        // Filter data based on time range
        const now = new Date();
        const cutoff = new Date();
        switch(timeRange) {
            case '1h':
                cutoff.setHours(now.getHours() - 1);
                break;
            case '24h':
                cutoff.setHours(now.getHours() - 24);
                break;
            case '7d':
                cutoff.setDate(now.getDate() - 7);
                break;
        }
        
        const filteredData = this.frerData.history.filter(point => point.timestamp >= cutoff);
        
        // Update chart data based on selected metric
        let chartData, label, color;
        switch(metricType) {
            case 'latency':
                chartData = filteredData.map(point => point.recoveryTime);
                label = 'FRER Recovery Time (ms)';
                color = '#2a5298';
                break;
            case 'throughput':
                chartData = filteredData.map(point => point.throughput / 1000000);
                label = 'Network Throughput (Mbps)';
                color = '#4fc3f7';
                break;
            case 'availability':
                chartData = filteredData.map(point => point.availability);
                label = 'System Availability (%)';
                color = '#4caf50';
                break;
            default:
                chartData = filteredData.map(point => point.recoveryTime);
                label = 'FRER Recovery Time (ms)';
                color = '#2a5298';
        }
        
        this.performanceChart.data.labels = filteredData.map(point => 
            point.timestamp.toLocaleTimeString()
        );
        this.performanceChart.data.datasets[0].data = chartData;
        this.performanceChart.data.datasets[0].label = label;
        this.performanceChart.data.datasets[0].borderColor = color;
        this.performanceChart.data.datasets[0].backgroundColor = color + '20';
        
        this.performanceChart.update('none');
    }
    
    setupEventListeners() {
        // Topology view controls
        document.querySelectorAll('.control-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.control-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.switchTopologyView(e.target.dataset.view);
            });
        });
        
        // Alert filter controls
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.filterAlerts(e.target.dataset.level);
            });
        });
        
        // Chart controls
        const metricSelector = document.getElementById('metric-selector');
        const timeRangeSelector = document.getElementById('time-range');
        
        if (metricSelector) {
            metricSelector.addEventListener('change', () => {
                this.updatePerformanceChart();
            });
        }
        
        if (timeRangeSelector) {
            timeRangeSelector.addEventListener('change', () => {
                this.updatePerformanceChart();
            });
        }
        
        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    }
    
    switchTopologyView(view) {
        const frerOverlay = document.getElementById('frer-paths');
        const dataFlow = document.getElementById('data-flow');
        
        switch(view) {
            case 'logical':
                if (frerOverlay) frerOverlay.style.display = 'none';
                if (dataFlow) dataFlow.style.display = 'block';
                break;
            case 'physical':
                if (frerOverlay) frerOverlay.style.display = 'none';
                if (dataFlow) dataFlow.style.display = 'none';
                break;
            case 'frer':
                if (frerOverlay) frerOverlay.style.display = 'block';
                if (dataFlow) dataFlow.style.display = 'block';
                break;
        }
    }
    
    renderAlerts() {
        const container = document.getElementById('alerts-container');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.alertHistory.slice(0, 10).forEach(alert => {
            const alertElement = document.createElement('div');
            alertElement.className = `alert alert-${alert.level}`;
            alertElement.innerHTML = `
                <div class="alert-header">
                    <span class="alert-time">${alert.timestamp.toLocaleTimeString()}</span>
                    <span class="alert-source">${alert.source}</span>
                    ${alert.vehicleId ? `<span class="vehicle-id">Vehicle ${alert.vehicleId}</span>` : ''}
                </div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-actions">
                    <button class="alert-action-btn" onclick="monitor.acknowledgeAlert(${alert.id})">
                        Acknowledge
                    </button>
                </div>
            `;
            container.appendChild(alertElement);
        });
    }
    
    acknowledgeAlert(alertId) {
        const alert = this.alertHistory.find(a => a.id === alertId);
        if (alert) {
            alert.resolved = true;
            this.renderAlerts();
        }
    }
    
    checkFRERAlerts(metrics) {
        // Check for critical thresholds
        if (metrics.availability < 99.999) {
            this.handleSystemAlert({
                level: 'critical',
                source: 'FRER Monitor',
                message: `Network availability dropped to ${metrics.availability.toFixed(4)}%`,
                timestamp: Date.now()
            });
        }
        
        if (metrics.recoveryTime > 5.0) {
            this.handleSystemAlert({
                level: 'warning',
                source: 'FRER Monitor',
                message: `Recovery time exceeded threshold: ${metrics.recoveryTime}ms`,
                timestamp: Date.now()
            });
        }
        
        if (metrics.pathFailures > 0) {
            this.handleSystemAlert({
                level: 'critical',
                source: 'Network Infrastructure',
                message: `${metrics.pathFailures} FRER path failures detected`,
                timestamp: Date.now()
            });
        }
    }
    
    getAvailabilityColor(percentage) {
        if (percentage >= 99.999) return '#4caf50';  // Green
        if (percentage >= 99.99) return '#ff9800';   // Orange
        return '#f44336';  // Red
    }
    
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        if (!statusElement) return;
        
        const icon = statusElement.querySelector('i');
        const text = statusElement.querySelector('span');
        
        switch(status) {
            case 'connected':
                statusElement.style.background = '#4caf50';
                icon.className = 'fas fa-wifi';
                text.textContent = 'Connected';
                break;
            case 'disconnected':
                statusElement.style.background = '#f44336';
                icon.className = 'fas fa-wifi-slash';
                text.textContent = 'Disconnected';
                break;
            case 'error':
                statusElement.style.background = '#ff9800';
                icon.className = 'fas fa-exclamation-triangle';
                text.textContent = 'Connection Error';
                break;
            case 'simulation':
                statusElement.style.background = '#2196f3';
                icon.className = 'fas fa-cog';
                text.textContent = 'Simulation Mode';
                break;
        }
    }
    
    attemptReconnection() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            
            setTimeout(() => {
                this.initializeWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.log('Max reconnection attempts reached. Switching to simulation mode.');
            this.updateConnectionStatus('simulation');
        }
    }
    
    playAlertSound() {
        // Create a simple alert tone
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 1);
        
        oscillator.start();
        oscillator.stop(audioContext.currentTime + 1);
    }
    
    showBrowserNotification(alert) {
        if ('Notification' in window && Notification.permission === 'granted') {
            const notification = new Notification(`A2Z System Alert - ${alert.level.toUpperCase()}`, {
                body: alert.message,
                icon: '/favicon.ico',
                badge: '/favicon.ico'
            });
            
            notification.onclick = () => {
                window.focus();
                notification.close();
            };
            
            setTimeout(() => notification.close(), 5000);
        }
    }
    
    // Demo data simulation (remove in production)
    startDataSimulation() {
        console.log('Starting A2Z monitoring simulation...');
        
        setInterval(() => {
            // Simulate A2Z Gigabit FRER metrics updates
            const newMetrics = {
                availability: 99.97 + (Math.random() * 0.03),
                recoveryTime: 12.3 + (Math.random() * 5) - 2.5,
                activeStreams: 4,
                networkUsage: `${Math.floor(450 + Math.random() * 100)}/1000`,
                pathFailures: Math.random() < 0.005 ? 1 : 0
            };
            
            this.updateFRERMetrics(newMetrics);
            
            // Simulate occasional alerts
            if (Math.random() < 0.02) {
                const alertTypes = [
                    { level: 'info', message: 'Vehicle maintenance scheduled', source: 'Fleet Management' },
                    { level: 'warning', message: 'Network latency slightly elevated', source: 'Network Monitor' },
                    { level: 'critical', message: 'Emergency brake test completed', source: 'Safety System' }
                ];
                
                const alert = alertTypes[Math.floor(Math.random() * alertTypes.length)];
                alert.vehicleId = Math.floor(Math.random() * 20) + 1;
                
                this.handleSystemAlert(alert);
            }
        }, 2000); // Update every 2 seconds
        
        // Simulate A2Z vehicle telemetry updates
        setInterval(() => {
            this.updateVehicleStatus({
                roiiCount: 7 + Math.floor(Math.random() * 2) - 1,
                coiiCount: 4 + Math.floor(Math.random() * 2) - 1,
                totalBandwidth: `${Math.floor(450 + Math.random() * 100)} Mbps`,
                safetyScore: 99.97 + (Math.random() * 0.03) - 0.015
            });
        }, 5000); // Update every 5 seconds
    }
    
    // Export data for reporting
    exportPerformanceReport() {
        const report = {
            timestamp: new Date().toISOString(),
            frerMetrics: this.frerData,
            vehicleStatus: this.vehicleData,
            alertSummary: {
                total: this.alertHistory.length,
                critical: this.alertHistory.filter(a => a.level === 'critical').length,
                warning: this.alertHistory.filter(a => a.level === 'warning').length,
                info: this.alertHistory.filter(a => a.level === 'info').length
            }
        };
        
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `a2z-performance-report-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Initialize the monitoring system
let monitor;

document.addEventListener('DOMContentLoaded', () => {
    monitor = new A2ZNetworkMonitor();
    
    // Add export button functionality if needed
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            monitor.exportPerformanceReport();
        });
    }
    
    console.log('A2Z Network Monitoring System initialized');
});