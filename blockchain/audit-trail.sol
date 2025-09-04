// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title A2Z TSN/FRER Blockchain Audit Trail
 * @dev Immutable audit trail for network events and configuration changes
 */

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

contract A2ZAuditTrail is AccessControl, ReentrancyGuard {
    using Counters for Counters.Counter;
    using ECDSA for bytes32;
    
    // Roles
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant AUDITOR_ROLE = keccak256("AUDITOR_ROLE");
    bytes32 public constant NETWORK_NODE_ROLE = keccak256("NETWORK_NODE_ROLE");
    
    // Event types
    enum EventType {
        CONFIGURATION_CHANGE,
        FRER_RECOVERY,
        ANOMALY_DETECTED,
        MAINTENANCE_ACTION,
        SECURITY_ALERT,
        PERFORMANCE_METRIC,
        FAILOVER_EVENT,
        COMPLIANCE_CHECK
    }
    
    // Severity levels
    enum Severity {
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    }
    
    // Audit event structure
    struct AuditEvent {
        uint256 id;
        uint256 timestamp;
        address reporter;
        EventType eventType;
        Severity severity;
        string switchId;
        string component;
        string description;
        bytes32 dataHash;
        string ipfsHash;
        bool verified;
        address verifier;
        uint256 blockNumber;
    }
    
    // Network metrics structure
    struct NetworkMetrics {
        uint256 timestamp;
        uint256 bandwidth;
        uint256 latency;
        uint256 packetLoss;
        uint256 availability;
        uint256 frerRecoveries;
        string switchId;
    }
    
    // Configuration change structure
    struct ConfigChange {
        uint256 id;
        uint256 timestamp;
        address initiator;
        string switchId;
        string configType;
        string oldValue;
        string newValue;
        bytes32 approvalHash;
        bool executed;
    }
    
    // State variables
    Counters.Counter private _eventIdCounter;
    Counters.Counter private _configIdCounter;
    
    mapping(uint256 => AuditEvent) public auditEvents;
    mapping(uint256 => NetworkMetrics) public metrics;
    mapping(uint256 => ConfigChange) public configChanges;
    mapping(address => bool) public authorizedNodes;
    mapping(string => uint256[]) public switchEvents;
    mapping(bytes32 => bool) public processedHashes;
    
    uint256[] public allEventIds;
    uint256[] public criticalEvents;
    
    // Statistics
    uint256 public totalEvents;
    uint256 public totalAnomalies;
    uint256 public totalRecoveries;
    uint256 public lastEventTime;
    
    // Events
    event AuditEventRecorded(
        uint256 indexed id,
        EventType indexed eventType,
        string switchId,
        address reporter
    );
    
    event ConfigurationChanged(
        uint256 indexed id,
        string switchId,
        string configType,
        address initiator
    );
    
    event MetricsRecorded(
        uint256 indexed timestamp,
        string switchId,
        uint256 bandwidth,
        uint256 latency
    );
    
    event AnomalyDetected(
        uint256 indexed id,
        Severity severity,
        string component,
        string description
    );
    
    event EmergencyAction(
        address indexed initiator,
        string action,
        string reason
    );
    
    modifier onlyAuthorizedNode() {
        require(
            authorizedNodes[msg.sender] || hasRole(NETWORK_NODE_ROLE, msg.sender),
            "Not an authorized network node"
        );
        _;
    }
    
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(AUDITOR_ROLE, msg.sender);
    }
    
    /**
     * @dev Record an audit event
     */
    function recordAuditEvent(
        EventType _eventType,
        Severity _severity,
        string memory _switchId,
        string memory _component,
        string memory _description,
        bytes32 _dataHash,
        string memory _ipfsHash
    ) external onlyAuthorizedNode nonReentrant {
        require(bytes(_switchId).length > 0, "Switch ID required");
        require(!processedHashes[_dataHash], "Event already processed");
        
        uint256 eventId = _eventIdCounter.current();
        _eventIdCounter.increment();
        
        AuditEvent memory newEvent = AuditEvent({
            id: eventId,
            timestamp: block.timestamp,
            reporter: msg.sender,
            eventType: _eventType,
            severity: _severity,
            switchId: _switchId,
            component: _component,
            description: _description,
            dataHash: _dataHash,
            ipfsHash: _ipfsHash,
            verified: false,
            verifier: address(0),
            blockNumber: block.number
        });
        
        auditEvents[eventId] = newEvent;
        allEventIds.push(eventId);
        switchEvents[_switchId].push(eventId);
        processedHashes[_dataHash] = true;
        
        if (_severity == Severity.CRITICAL) {
            criticalEvents.push(eventId);
        }
        
        // Update statistics
        totalEvents++;
        lastEventTime = block.timestamp;
        
        if (_eventType == EventType.ANOMALY_DETECTED) {
            totalAnomalies++;
            emit AnomalyDetected(eventId, _severity, _component, _description);
        } else if (_eventType == EventType.FRER_RECOVERY) {
            totalRecoveries++;
        }
        
        emit AuditEventRecorded(eventId, _eventType, _switchId, msg.sender);
    }
    
    /**
     * @dev Record network metrics
     */
    function recordMetrics(
        string memory _switchId,
        uint256 _bandwidth,
        uint256 _latency,
        uint256 _packetLoss,
        uint256 _availability,
        uint256 _frerRecoveries
    ) external onlyAuthorizedNode {
        NetworkMetrics memory newMetrics = NetworkMetrics({
            timestamp: block.timestamp,
            bandwidth: _bandwidth,
            latency: _latency,
            packetLoss: _packetLoss,
            availability: _availability,
            frerRecoveries: _frerRecoveries,
            switchId: _switchId
        });
        
        metrics[block.timestamp] = newMetrics;
        
        emit MetricsRecorded(block.timestamp, _switchId, _bandwidth, _latency);
    }
    
    /**
     * @dev Record configuration change
     */
    function recordConfigChange(
        string memory _switchId,
        string memory _configType,
        string memory _oldValue,
        string memory _newValue,
        bytes32 _approvalHash
    ) external onlyRole(ADMIN_ROLE) {
        uint256 configId = _configIdCounter.current();
        _configIdCounter.increment();
        
        ConfigChange memory change = ConfigChange({
            id: configId,
            timestamp: block.timestamp,
            initiator: msg.sender,
            switchId: _switchId,
            configType: _configType,
            oldValue: _oldValue,
            newValue: _newValue,
            approvalHash: _approvalHash,
            executed: false
        });
        
        configChanges[configId] = change;
        
        emit ConfigurationChanged(configId, _switchId, _configType, msg.sender);
    }
    
    /**
     * @dev Verify an audit event
     */
    function verifyEvent(uint256 _eventId) external onlyRole(AUDITOR_ROLE) {
        require(_eventId < _eventIdCounter.current(), "Invalid event ID");
        require(!auditEvents[_eventId].verified, "Already verified");
        
        auditEvents[_eventId].verified = true;
        auditEvents[_eventId].verifier = msg.sender;
    }
    
    /**
     * @dev Execute emergency action
     */
    function executeEmergencyAction(
        string memory _action,
        string memory _reason
    ) external onlyRole(ADMIN_ROLE) {
        emit EmergencyAction(msg.sender, _action, _reason);
        
        // Record as critical event
        recordAuditEvent(
            EventType.SECURITY_ALERT,
            Severity.CRITICAL,
            "SYSTEM",
            "EMERGENCY",
            string(abi.encodePacked("Emergency: ", _action, " - ", _reason)),
            keccak256(abi.encodePacked(_action, _reason)),
            ""
        );
    }
    
    /**
     * @dev Authorize a network node
     */
    function authorizeNode(address _node) external onlyRole(ADMIN_ROLE) {
        authorizedNodes[_node] = true;
        _grantRole(NETWORK_NODE_ROLE, _node);
    }
    
    /**
     * @dev Revoke node authorization
     */
    function revokeNode(address _node) external onlyRole(ADMIN_ROLE) {
        authorizedNodes[_node] = false;
        _revokeRole(NETWORK_NODE_ROLE, _node);
    }
    
    /**
     * @dev Get events for a specific switch
     */
    function getSwitchEvents(string memory _switchId) 
        external 
        view 
        returns (uint256[] memory) 
    {
        return switchEvents[_switchId];
    }
    
    /**
     * @dev Get all critical events
     */
    function getCriticalEvents() external view returns (uint256[] memory) {
        return criticalEvents;
    }
    
    /**
     * @dev Get events by type
     */
    function getEventsByType(EventType _eventType) 
        external 
        view 
        returns (uint256[] memory) 
    {
        uint256 count = 0;
        for (uint256 i = 0; i < allEventIds.length; i++) {
            if (auditEvents[allEventIds[i]].eventType == _eventType) {
                count++;
            }
        }
        
        uint256[] memory filteredEvents = new uint256[](count);
        uint256 index = 0;
        for (uint256 i = 0; i < allEventIds.length; i++) {
            if (auditEvents[allEventIds[i]].eventType == _eventType) {
                filteredEvents[index] = allEventIds[i];
                index++;
            }
        }
        
        return filteredEvents;
    }
    
    /**
     * @dev Get recent events
     */
    function getRecentEvents(uint256 _count) 
        external 
        view 
        returns (AuditEvent[] memory) 
    {
        uint256 totalCount = allEventIds.length;
        uint256 returnCount = _count > totalCount ? totalCount : _count;
        
        AuditEvent[] memory recentEvents = new AuditEvent[](returnCount);
        
        for (uint256 i = 0; i < returnCount; i++) {
            uint256 eventId = allEventIds[totalCount - 1 - i];
            recentEvents[i] = auditEvents[eventId];
        }
        
        return recentEvents;
    }
    
    /**
     * @dev Get system statistics
     */
    function getStatistics() 
        external 
        view 
        returns (
            uint256 _totalEvents,
            uint256 _totalAnomalies,
            uint256 _totalRecoveries,
            uint256 _criticalCount,
            uint256 _lastEventTime
        ) 
    {
        return (
            totalEvents,
            totalAnomalies,
            totalRecoveries,
            criticalEvents.length,
            lastEventTime
        );
    }
    
    /**
     * @dev Check data integrity
     */
    function verifyDataIntegrity(bytes32 _dataHash) 
        external 
        view 
        returns (bool) 
    {
        return processedHashes[_dataHash];
    }
    
    /**
     * @dev Generate audit report hash
     */
    function generateAuditReport(uint256 _fromTime, uint256 _toTime) 
        external 
        view 
        returns (bytes32) 
    {
        bytes32 reportHash = keccak256(abi.encodePacked(_fromTime, _toTime));
        
        for (uint256 i = 0; i < allEventIds.length; i++) {
            AuditEvent memory evt = auditEvents[allEventIds[i]];
            if (evt.timestamp >= _fromTime && evt.timestamp <= _toTime) {
                reportHash = keccak256(
                    abi.encodePacked(
                        reportHash,
                        evt.id,
                        evt.dataHash
                    )
                );
            }
        }
        
        return reportHash;
    }
}

/**
 * @title A2Z Compliance Oracle
 * @dev Oracle contract for compliance verification
 */
contract A2ZComplianceOracle {
    
    struct ComplianceCheck {
        uint256 timestamp;
        string standard;  // IEEE 802.1CB, ISO 26262, etc.
        bool compliant;
        string details;
        address verifier;
    }
    
    mapping(string => ComplianceCheck[]) public complianceHistory;
    mapping(string => bool) public currentCompliance;
    
    event ComplianceUpdated(
        string standard,
        bool compliant,
        address verifier
    );
    
    function updateCompliance(
        string memory _standard,
        bool _compliant,
        string memory _details
    ) external {
        ComplianceCheck memory check = ComplianceCheck({
            timestamp: block.timestamp,
            standard: _standard,
            compliant: _compliant,
            details: _details,
            verifier: msg.sender
        });
        
        complianceHistory[_standard].push(check);
        currentCompliance[_standard] = _compliant;
        
        emit ComplianceUpdated(_standard, _compliant, msg.sender);
    }
    
    function getComplianceStatus(string memory _standard) 
        external 
        view 
        returns (bool) 
    {
        return currentCompliance[_standard];
    }
}