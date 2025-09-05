// A2Z Gigabit TSN/FRER Network Topology Visualization
// Interactive D3.js implementation for real-time network monitoring

class NetworkTopology {
    constructor() {
        this.width = window.innerWidth;
        this.height = window.innerHeight - 180;
        this.animationRunning = true;
        this.showFRER = false;
        this.currentView = 'zone';
        
        // Network data structure
        this.networkData = {
            nodes: [],
            links: [],
            frerPaths: [],
            packets: []
        };
        
        this.init();
    }
    
    init() {
        // Create SVG canvas
        this.svg = d3.select('#network-canvas')
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);
        
        // Create groups for different layers
        this.linkGroup = this.svg.append('g').attr('class', 'links');
        this.frerGroup = this.svg.append('g').attr('class', 'frer-paths');
        this.nodeGroup = this.svg.append('g').attr('class', 'nodes');
        this.packetGroup = this.svg.append('g').attr('class', 'packets');
        this.labelGroup = this.svg.append('g').attr('class', 'labels');
        
        // Initialize network data
        this.initializeNetworkData();
        
        // Create force simulation
        this.simulation = d3.forceSimulation(this.networkData.nodes)
            .force('link', d3.forceLink(this.networkData.links).id(d => d.id).distance(150))
            .force('charge', d3.forceManyBody().strength(-500))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(60));
        
        // Render network
        this.render();
        
        // Start animations
        this.startAnimations();
        
        // Handle window resize
        window.addEventListener('resize', () => this.handleResize());
    }
    
    initializeNetworkData() {
        // Zone View Network Structure
        if (this.currentView === 'zone') {
            this.networkData.nodes = [
                // Central Zone
                { id: 'central-switch', name: 'LAN9692 Central', type: 'switch', zone: 'central', 
                  x: this.width / 2, y: this.height / 2, bandwidth: '66Gbps', ports: 30 },
                
                // Front Zone
                { id: 'front-switch', name: 'LAN9662 Front', type: 'switch', zone: 'front',
                  x: this.width / 2 - 200, y: this.height / 2 - 150, bandwidth: '8Gbps', ports: 8 },
                { id: 'lidar-front', name: 'Front LiDAR', type: 'sensor', zone: 'front',
                  dataRate: '100Mbps', priority: 'critical' },
                { id: 'camera-front', name: 'Front Camera Array', type: 'sensor', zone: 'front',
                  dataRate: '400Mbps', priority: 'critical' },
                { id: 'radar-front', name: 'Front Radar', type: 'sensor', zone: 'front',
                  dataRate: '50Mbps', priority: 'high' },
                
                // Rear Zone
                { id: 'rear-switch', name: 'LAN9662 Rear', type: 'switch', zone: 'rear',
                  x: this.width / 2 + 200, y: this.height / 2 - 150, bandwidth: '8Gbps', ports: 8 },
                { id: 'lidar-rear', name: 'Rear LiDAR', type: 'sensor', zone: 'rear',
                  dataRate: '100Mbps', priority: 'critical' },
                { id: 'camera-rear', name: 'Rear Camera', type: 'sensor', zone: 'rear',
                  dataRate: '100Mbps', priority: 'high' },
                
                // Control Units
                { id: 'main-ecu', name: 'Main ECU', type: 'control', zone: 'central',
                  processing: 'AI Compute', performance: '100 TOPS' },
                { id: 'safety-ecu', name: 'Safety ECU', type: 'control', zone: 'central',
                  processing: 'Emergency Control', latency: '<1ms' },
                { id: 'brake-control', name: 'Brake Controller', type: 'control', zone: 'central',
                  dataRate: '1Mbps', priority: 'critical' },
                { id: 'steering-control', name: 'Steering Controller', type: 'control', zone: 'central',
                  dataRate: '10Mbps', priority: 'critical' }
            ];
            
            this.networkData.links = [
                // Primary gigabit backbone links
                { source: 'central-switch', target: 'front-switch', bandwidth: 1000, type: 'backbone' },
                { source: 'central-switch', target: 'rear-switch', bandwidth: 1000, type: 'backbone' },
                
                // Front zone connections
                { source: 'front-switch', target: 'lidar-front', bandwidth: 100, type: 'sensor' },
                { source: 'front-switch', target: 'camera-front', bandwidth: 400, type: 'sensor' },
                { source: 'front-switch', target: 'radar-front', bandwidth: 50, type: 'sensor' },
                
                // Rear zone connections
                { source: 'rear-switch', target: 'lidar-rear', bandwidth: 100, type: 'sensor' },
                { source: 'rear-switch', target: 'camera-rear', bandwidth: 100, type: 'sensor' },
                
                // Control unit connections
                { source: 'central-switch', target: 'main-ecu', bandwidth: 1000, type: 'control' },
                { source: 'central-switch', target: 'safety-ecu', bandwidth: 100, type: 'control' },
                { source: 'central-switch', target: 'brake-control', bandwidth: 10, type: 'control' },
                { source: 'central-switch', target: 'steering-control', bandwidth: 10, type: 'control' }
            ];
            
            // FRER redundant paths
            this.networkData.frerPaths = [
                // Critical Stream 1001 - LiDAR (2-path redundancy)
                { id: 'frer-1001-p1', stream: 1001, path: ['lidar-front', 'front-switch', 'central-switch', 'main-ecu'] },
                { id: 'frer-1001-p2', stream: 1001, path: ['lidar-front', 'front-switch', 'rear-switch', 'central-switch', 'main-ecu'] },
                
                // Critical Stream 1002 - Camera (2-path redundancy)
                { id: 'frer-1002-p1', stream: 1002, path: ['camera-front', 'front-switch', 'central-switch', 'main-ecu'] },
                { id: 'frer-1002-p2', stream: 1002, path: ['camera-front', 'front-switch', 'rear-switch', 'central-switch', 'main-ecu'] },
                
                // Critical Stream 1003 - Emergency Brake (3-path redundancy)
                { id: 'frer-1003-p1', stream: 1003, path: ['brake-control', 'central-switch', 'safety-ecu'] },
                { id: 'frer-1003-p2', stream: 1003, path: ['brake-control', 'central-switch', 'front-switch', 'central-switch', 'safety-ecu'] },
                { id: 'frer-1003-p3', stream: 1003, path: ['brake-control', 'central-switch', 'rear-switch', 'central-switch', 'safety-ecu'] }
            ];
        }
    }
    
    render() {
        // Render links
        const links = this.linkGroup.selectAll('.link')
            .data(this.networkData.links)
            .enter().append('line')
            .attr('class', 'link')
            .attr('stroke', d => this.getLinkColor(d.type))
            .attr('stroke-width', d => Math.sqrt(d.bandwidth / 100))
            .attr('stroke-opacity', 0.6);
        
        // Render nodes
        const nodes = this.nodeGroup.selectAll('.node')
            .data(this.networkData.nodes)
            .enter().append('g')
            .attr('class', 'node')
            .call(this.drag());
        
        // Add node shapes based on type
        nodes.each(function(d) {
            const node = d3.select(this);
            
            if (d.type === 'switch') {
                // TSN Switch - Rectangle
                node.append('rect')
                    .attr('width', 80)
                    .attr('height', 60)
                    .attr('x', -40)
                    .attr('y', -30)
                    .attr('rx', 10)
                    .attr('fill', '#0088ff')
                    .attr('class', 'switch-node');
            } else if (d.type === 'sensor') {
                // Sensor - Circle
                node.append('circle')
                    .attr('r', 25)
                    .attr('fill', '#ffcc00')
                    .attr('class', 'sensor-node');
            } else if (d.type === 'control') {
                // Control Unit - Diamond
                node.append('rect')
                    .attr('width', 50)
                    .attr('height', 50)
                    .attr('x', -25)
                    .attr('y', -25)
                    .attr('transform', 'rotate(45)')
                    .attr('fill', '#ff4488');
            }
            
            // Add icon or symbol
            node.append('text')
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .attr('fill', 'white')
                .attr('font-size', '20px')
                .attr('font-weight', 'bold')
                .text(d => this.getNodeIcon(d));
        });
        
        // Add labels
        const labels = this.labelGroup.selectAll('.label')
            .data(this.networkData.nodes)
            .enter().append('text')
            .attr('class', 'label')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('y', d => d.type === 'switch' ? 45 : 40)
            .attr('fill', '#ffffff')
            .attr('font-size', '11px')
            .text(d => d.name);
        
        // Add hover interactions
        nodes.on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());
        
        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            links
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            nodes.attr('transform', d => `translate(${d.x}, ${d.y})`);
            labels.attr('transform', d => `translate(${d.x}, ${d.y})`);
            
            if (this.showFRER) {
                this.renderFRERPaths();
            }
        });
    }
    
    renderFRERPaths() {
        // Clear existing FRER paths
        this.frerGroup.selectAll('*').remove();
        
        this.networkData.frerPaths.forEach((frerPath, index) => {
            const pathCoords = frerPath.path.map(nodeId => {
                const node = this.networkData.nodes.find(n => n.id === nodeId);
                return node ? [node.x, node.y] : null;
            }).filter(coord => coord !== null);
            
            if (pathCoords.length > 1) {
                const lineGenerator = d3.line()
                    .x(d => d[0])
                    .y(d => d[1])
                    .curve(d3.curveCardinal.tension(0.5));
                
                this.frerGroup.append('path')
                    .attr('d', lineGenerator(pathCoords))
                    .attr('stroke', this.getFRERColor(frerPath.stream))
                    .attr('stroke-width', 2)
                    .attr('stroke-opacity', 0.5)
                    .attr('fill', 'none')
                    .attr('stroke-dasharray', index === 0 ? '' : '5,5');
            }
        });
    }
    
    startAnimations() {
        if (!this.animationRunning) return;
        
        // Animate packet flows
        setInterval(() => {
            if (this.animationRunning) {
                this.animatePackets();
            }
        }, 2000);
        
        // Update metrics
        setInterval(() => {
            this.updateMetrics();
        }, 1000);
    }
    
    animatePackets() {
        // Select random active link
        const activeLinks = this.networkData.links.filter(l => l.bandwidth > 10);
        const randomLink = activeLinks[Math.floor(Math.random() * activeLinks.length)];
        
        if (randomLink) {
            const packet = this.packetGroup.append('circle')
                .attr('r', 4)
                .attr('class', 'packet')
                .attr('cx', randomLink.source.x)
                .attr('cy', randomLink.source.y);
            
            packet.transition()
                .duration(1000)
                .attr('cx', randomLink.target.x)
                .attr('cy', randomLink.target.y)
                .on('end', function() {
                    d3.select(this).remove();
                });
        }
    }
    
    updateMetrics() {
        // Simulate metric updates
        const metrics = {
            bandwidth: 500 + Math.random() * 120,
            latency: 0.5 + Math.random() * 0.5,
            recovery: 10 + Math.random() * 5,
            packetLoss: Math.random() * 0.00005,
            uptime: 99.95 + Math.random() * 0.04
        };
        
        document.querySelector('.info-panel .metric:nth-child(1) .metric-value').innerHTML = 
            `${metrics.bandwidth.toFixed(0)}<span class="metric-unit">Mbps</span>`;
        document.querySelector('.info-panel .metric:nth-child(3) .metric-value').innerHTML = 
            `${metrics.latency.toFixed(1)}<span class="metric-unit">ms</span>`;
        document.querySelector('.info-panel .metric:nth-child(4) .metric-value').innerHTML = 
            `${metrics.recovery.toFixed(1)}<span class="metric-unit">ms</span>`;
        document.querySelector('.info-panel .metric:nth-child(5) .metric-value').innerHTML = 
            `${metrics.packetLoss.toFixed(5)}<span class="metric-unit">%</span>`;
        document.querySelector('.info-panel .metric:nth-child(6) .metric-value').innerHTML = 
            `${metrics.uptime.toFixed(2)}<span class="metric-unit">%</span>`;
    }
    
    getLinkColor(type) {
        const colors = {
            backbone: '#0088ff',
            sensor: '#ffcc00',
            control: '#ff4488',
            redundant: '#666666'
        };
        return colors[type] || '#888888';
    }
    
    getFRERColor(stream) {
        const colors = {
            1001: '#00ff88', // LiDAR
            1002: '#00ccff', // Camera
            1003: '#ff0044', // Emergency
            1004: '#ffaa00'  // Steering
        };
        return colors[stream] || '#888888';
    }
    
    getNodeIcon(node) {
        const icons = {
            switch: '⚡',
            sensor: '📡',
            control: '🎮'
        };
        return icons[node.type] || '●';
    }
    
    showTooltip(event, d) {
        const tooltip = document.getElementById('tooltip');
        let content = `<strong>${d.name}</strong><br>`;
        
        if (d.type === 'switch') {
            content += `Type: TSN Switch<br>`;
            content += `Bandwidth: ${d.bandwidth}<br>`;
            content += `Ports: ${d.ports}<br>`;
            content += `Zone: ${d.zone}`;
        } else if (d.type === 'sensor') {
            content += `Type: Sensor<br>`;
            content += `Data Rate: ${d.dataRate}<br>`;
            content += `Priority: ${d.priority}<br>`;
            content += `Zone: ${d.zone}`;
        } else if (d.type === 'control') {
            content += `Type: Control Unit<br>`;
            if (d.processing) content += `Processing: ${d.processing}<br>`;
            if (d.performance) content += `Performance: ${d.performance}<br>`;
            if (d.latency) content += `Latency: ${d.latency}`;
        }
        
        tooltip.innerHTML = content;
        tooltip.style.display = 'block';
        tooltip.style.left = (event.pageX + 10) + 'px';
        tooltip.style.top = (event.pageY - 10) + 'px';
    }
    
    hideTooltip() {
        document.getElementById('tooltip').style.display = 'none';
    }
    
    drag() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }
    
    handleResize() {
        this.width = window.innerWidth;
        this.height = window.innerHeight - 180;
        this.svg.attr('width', this.width).attr('height', this.height);
        this.simulation.force('center', d3.forceCenter(this.width / 2, this.height / 2));
        this.simulation.alpha(0.3).restart();
    }
}

// Initialize network topology
let topology = new NetworkTopology();

// Control functions
function toggleAnimation() {
    topology.animationRunning = !topology.animationRunning;
    document.getElementById('animBtn').textContent = 
        topology.animationRunning ? 'Pause Animation' : 'Start Animation';
}

function toggleFRER() {
    topology.showFRER = !topology.showFRER;
    document.getElementById('frerBtn').textContent = 
        topology.showFRER ? 'Hide FRER Paths' : 'Show FRER Paths';
    
    if (!topology.showFRER) {
        topology.frerGroup.selectAll('*').remove();
    }
}

function changeView(view) {
    topology.currentView = view;
    document.querySelectorAll('.controls button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Re-initialize with new view
    topology.initializeNetworkData();
    topology.simulation.nodes(topology.networkData.nodes);
    topology.simulation.force('link').links(topology.networkData.links);
    topology.svg.selectAll('*').remove();
    topology.init();
}