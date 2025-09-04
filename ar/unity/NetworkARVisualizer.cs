using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.UI;
using UnityEngine.Networking;
using TMPro;
using Newtonsoft.Json;
using WebSocketSharp;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.UI;
using Microsoft.MixedReality.Toolkit.Input;

namespace A2Z.AR.Visualization
{
    /// <summary>
    /// A2Z TSN/FRER Network AR Visualizer for HoloLens 2 and Mobile AR
    /// Real-time 3D visualization of network topology and data flows
    /// </summary>
    public class NetworkARVisualizer : MonoBehaviour
    {
        [Header("AR Configuration")]
        [SerializeField] private ARSessionOrigin arOrigin;
        [SerializeField] private ARRaycastManager raycastManager;
        [SerializeField] private ARAnchorManager anchorManager;
        [SerializeField] private ARPlaneManager planeManager;
        
        [Header("Network Visualization")]
        [SerializeField] private GameObject switchPrefab;
        [SerializeField] private GameObject serverPrefab;
        [SerializeField] private GameObject dataFlowPrefab;
        [SerializeField] private GameObject alertPrefab;
        [SerializeField] private Material normalMaterial;
        [SerializeField] private Material warningMaterial;
        [SerializeField] private Material criticalMaterial;
        [SerializeField] private Material frerActiveMaterial;
        
        [Header("UI Components")]
        [SerializeField] private GameObject infoPanel;
        [SerializeField] private TextMeshProUGUI statusText;
        [SerializeField] private TextMeshProUGUI metricsText;
        [SerializeField] private Slider bandwidthSlider;
        [SerializeField] private Image latencyIndicator;
        [SerializeField] private GameObject miniMap;
        
        [Header("Connection Settings")]
        [SerializeField] private string apiUrl = "https://api.a2z-tsn.com";
        [SerializeField] private string wsUrl = "wss://ws.a2z-tsn.com";
        [SerializeField] private string apiKey = "";
        
        // Network topology
        private Dictionary<string, GameObject> switches = new Dictionary<string, GameObject>();
        private Dictionary<string, GameObject> connections = new Dictionary<string, GameObject>();
        private Dictionary<string, FRERStream> frerStreams = new Dictionary<string, FRERStream>();
        private List<ARRaycastHit> raycastHits = new List<ARRaycastHit>();
        
        // WebSocket connection
        private WebSocket webSocket;
        private Queue<Action> mainThreadActions = new Queue<Action>();
        
        // Visualization state
        private bool isVisualizationActive = false;
        private GameObject selectedObject = null;
        private float updateInterval = 1.0f;
        private float lastUpdateTime = 0;
        
        // Performance metrics
        private NetworkMetrics currentMetrics;
        private List<Alert> activeAlerts = new List<Alert>();
        
        // Holographic settings
        private bool isHoloLens = false;
        private float hologramScale = 0.5f;
        private Vector3 anchorPosition;
        
        #region Unity Lifecycle
        
        void Awake()
        {
            // Detect if running on HoloLens
            #if UNITY_WSA && !UNITY_EDITOR
            isHoloLens = true;
            SetupHoloLens();
            #else
            SetupMobileAR();
            #endif
        }
        
        void Start()
        {
            InitializeAR();
            ConnectToNetwork();
            StartCoroutine(UpdateVisualization());
        }
        
        void Update()
        {
            // Process main thread actions from WebSocket
            while (mainThreadActions.Count > 0)
            {
                mainThreadActions.Dequeue()?.Invoke();
            }
            
            // Handle user input
            HandleInput();
            
            // Update animations
            UpdateDataFlowAnimations();
            
            // Update metrics display
            if (Time.time - lastUpdateTime > updateInterval)
            {
                UpdateMetricsDisplay();
                lastUpdateTime = Time.time;
            }
        }
        
        void OnDestroy()
        {
            DisconnectWebSocket();
            StopAllCoroutines();
        }
        
        #endregion
        
        #region AR Initialization
        
        private void InitializeAR()
        {
            if (isHoloLens)
            {
                // HoloLens specific initialization
                InitializeHoloLens();
            }
            else
            {
                // Mobile AR initialization
                InitializeMobileAR();
            }
        }
        
        private void InitializeHoloLens()
        {
            // Configure spatial mapping
            var spatialMapping = gameObject.AddComponent<SpatialMapping>();
            spatialMapping.Enable();
            
            // Set up hand tracking
            var handTracking = gameObject.AddComponent<HandTracking>();
            handTracking.OnHandDetected += OnHandDetected;
            
            // Configure voice commands
            SetupVoiceCommands();
            
            Debug.Log("HoloLens AR initialized");
        }
        
        private void InitializeMobileAR()
        {
            // Configure AR plane detection
            planeManager.planesChanged += OnPlanesChanged;
            
            // Set up touch input
            Input.multiTouchEnabled = true;
            
            Debug.Log("Mobile AR initialized");
        }
        
        private void SetupHoloLens()
        {
            hologramScale = 0.3f;
            
            // Configure MRTK settings
            CoreServices.InputSystem.RegisterHandler<IMixedRealityPointerHandler>(this);
            CoreServices.InputSystem.RegisterHandler<IMixedRealityFocusHandler>(this);
        }
        
        private void SetupMobileAR()
        {
            hologramScale = 0.5f;
            
            // Configure ARCore/ARKit settings
            var config = arOrigin.GetComponent<ARSession>().subsystem.sessionConfig;
            config.planeDetection = PlaneDetectionMode.HorizontalAndVertical;
            config.lightEstimation = LightEstimation.AmbientIntensity;
        }
        
        #endregion
        
        #region Network Connection
        
        private void ConnectToNetwork()
        {
            // Connect to WebSocket for real-time updates
            webSocket = new WebSocket(wsUrl);
            
            webSocket.OnOpen += (sender, e) =>
            {
                Debug.Log("WebSocket connected");
                SendAuthentication();
                SubscribeToUpdates();
            };
            
            webSocket.OnMessage += (sender, e) =>
            {
                ProcessWebSocketMessage(e.Data);
            };
            
            webSocket.OnError += (sender, e) =>
            {
                Debug.LogError($"WebSocket error: {e.Message}");
            };
            
            webSocket.OnClose += (sender, e) =>
            {
                Debug.Log($"WebSocket closed: {e.Reason}");
                StartCoroutine(ReconnectWebSocket());
            };
            
            webSocket.Connect();
            
            // Load initial topology
            StartCoroutine(LoadNetworkTopology());
        }
        
        private void SendAuthentication()
        {
            var auth = new
            {
                type = "auth",
                apiKey = apiKey
            };
            webSocket.Send(JsonConvert.SerializeObject(auth));
        }
        
        private void SubscribeToUpdates()
        {
            var subscribe = new
            {
                type = "subscribe",
                streams = new[] { "metrics", "alerts", "frer", "topology" }
            };
            webSocket.Send(JsonConvert.SerializeObject(subscribe));
        }
        
        private void ProcessWebSocketMessage(string message)
        {
            try
            {
                dynamic data = JsonConvert.DeserializeObject(message);
                string type = data.type;
                
                mainThreadActions.Enqueue(() =>
                {
                    switch (type)
                    {
                        case "metrics":
                            UpdateMetrics(data.data);
                            break;
                        case "alert":
                            ShowAlert(data.data);
                            break;
                        case "frer_event":
                            ShowFREREvent(data.data);
                            break;
                        case "topology_change":
                            UpdateTopology(data.data);
                            break;
                    }
                });
            }
            catch (Exception e)
            {
                Debug.LogError($"Error processing WebSocket message: {e.Message}");
            }
        }
        
        private IEnumerator ReconnectWebSocket()
        {
            yield return new WaitForSeconds(5);
            
            if (webSocket.ReadyState != WebSocketState.Open)
            {
                Debug.Log("Attempting to reconnect WebSocket...");
                webSocket.Connect();
            }
        }
        
        private void DisconnectWebSocket()
        {
            if (webSocket != null && webSocket.ReadyState == WebSocketState.Open)
            {
                webSocket.Close();
            }
        }
        
        #endregion
        
        #region Network Topology Visualization
        
        private IEnumerator LoadNetworkTopology()
        {
            string url = $"{apiUrl}/v2/network/topology";
            
            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                request.SetRequestHeader("Authorization", $"Bearer {apiKey}");
                
                yield return request.SendWebRequest();
                
                if (request.result == UnityWebRequest.Result.Success)
                {
                    var topology = JsonConvert.DeserializeObject<NetworkTopology>(request.downloadHandler.text);
                    CreateNetworkVisualization(topology);
                }
                else
                {
                    Debug.LogError($"Failed to load topology: {request.error}");
                }
            }
        }
        
        private void CreateNetworkVisualization(NetworkTopology topology)
        {
            // Clear existing visualization
            ClearVisualization();
            
            // Create anchor point
            if (isHoloLens)
            {
                anchorPosition = Camera.main.transform.position + Camera.main.transform.forward * 2f;
            }
            else
            {
                // For mobile AR, place on detected plane
                anchorPosition = GetPlacementPosition();
            }
            
            // Create switches
            foreach (var switchInfo in topology.switches)
            {
                CreateSwitchObject(switchInfo);
            }
            
            // Create connections
            foreach (var connection in topology.connections)
            {
                CreateConnectionObject(connection);
            }
            
            // Create FRER streams
            foreach (var stream in topology.frerStreams)
            {
                CreateFRERStreamVisualization(stream);
            }
            
            isVisualizationActive = true;
        }
        
        private void CreateSwitchObject(SwitchInfo switchInfo)
        {
            GameObject switchObj = Instantiate(switchPrefab);
            switchObj.name = switchInfo.id;
            
            // Position based on zone
            Vector3 position = CalculateSwitchPosition(switchInfo.zone, switchInfo.index);
            switchObj.transform.position = anchorPosition + position * hologramScale;
            switchObj.transform.localScale = Vector3.one * hologramScale;
            
            // Configure switch visualization
            var visualizer = switchObj.AddComponent<SwitchVisualizer>();
            visualizer.Initialize(switchInfo);
            
            // Add interaction
            if (isHoloLens)
            {
                AddHoloLensInteraction(switchObj);
            }
            else
            {
                AddMobileInteraction(switchObj);
            }
            
            // Set material based on status
            UpdateSwitchMaterial(switchObj, switchInfo.status);
            
            // Store reference
            switches[switchInfo.id] = switchObj;
            
            // Add label
            CreateLabel(switchObj, switchInfo.id);
        }
        
        private Vector3 CalculateSwitchPosition(string zone, int index)
        {
            Vector3 basePosition = Vector3.zero;
            
            switch (zone)
            {
                case "front":
                    basePosition = new Vector3(-2f, 0, 2f);
                    break;
                case "central":
                    basePosition = new Vector3(0, 0, 0);
                    break;
                case "rear":
                    basePosition = new Vector3(2f, 0, 2f);
                    break;
            }
            
            // Add offset for multiple switches in same zone
            basePosition.x += index * 0.5f;
            basePosition.y += (index % 2) * 0.3f;
            
            return basePosition;
        }
        
        private void CreateConnectionObject(ConnectionInfo connection)
        {
            if (!switches.ContainsKey(connection.from) || !switches.ContainsKey(connection.to))
                return;
            
            GameObject fromSwitch = switches[connection.from];
            GameObject toSwitch = switches[connection.to];
            
            // Create line renderer for connection
            GameObject connectionObj = new GameObject($"Connection_{connection.from}_{connection.to}");
            LineRenderer line = connectionObj.AddComponent<LineRenderer>();
            
            // Configure line appearance
            line.material = normalMaterial;
            line.startWidth = 0.02f * hologramScale;
            line.endWidth = 0.02f * hologramScale;
            line.positionCount = 2;
            line.SetPosition(0, fromSwitch.transform.position);
            line.SetPosition(1, toSwitch.transform.position);
            
            // Add data flow animation
            var flowAnimator = connectionObj.AddComponent<DataFlowAnimator>();
            flowAnimator.Initialize(connection.bandwidth, connection.latency);
            
            connections[$"{connection.from}_{connection.to}"] = connectionObj;
        }
        
        private void CreateFRERStreamVisualization(FRERStream stream)
        {
            // Create primary path
            CreateFRERPath(stream.primaryPath, true, stream.id);
            
            // Create secondary path
            CreateFRERPath(stream.secondaryPath, false, stream.id);
            
            frerStreams[stream.id] = stream;
        }
        
        private void CreateFRERPath(List<string> path, bool isPrimary, string streamId)
        {
            if (path.Count < 2) return;
            
            GameObject pathObj = new GameObject($"FRER_{streamId}_{(isPrimary ? "Primary" : "Secondary")}");
            LineRenderer line = pathObj.AddComponent<LineRenderer>();
            
            // Configure FRER path appearance
            line.material = frerActiveMaterial;
            line.startWidth = 0.03f * hologramScale;
            line.endWidth = 0.03f * hologramScale;
            line.positionCount = path.Count;
            
            // Set path positions
            for (int i = 0; i < path.Count; i++)
            {
                if (switches.ContainsKey(path[i]))
                {
                    Vector3 pos = switches[path[i]].transform.position;
                    pos.y += isPrimary ? 0.1f : -0.1f; // Offset for visibility
                    line.SetPosition(i, pos);
                }
            }
            
            // Animate FRER data flow
            var frerAnimator = pathObj.AddComponent<FRERAnimator>();
            frerAnimator.Initialize(streamId, isPrimary);
        }
        
        #endregion
        
        #region User Interaction
        
        private void HandleInput()
        {
            if (isHoloLens)
            {
                HandleHoloLensInput();
            }
            else
            {
                HandleMobileInput();
            }
        }
        
        private void HandleHoloLensInput()
        {
            // Hand tracking and gesture recognition handled by MRTK
            // Voice commands processed separately
        }
        
        private void HandleMobileInput()
        {
            if (Input.touchCount == 0) return;
            
            Touch touch = Input.GetTouch(0);
            
            if (touch.phase == TouchPhase.Began)
            {
                // Raycast to detect object selection
                if (raycastManager.Raycast(touch.position, raycastHits))
                {
                    HandleObjectSelection(raycastHits[0].hitPose.position);
                }
            }
            else if (touch.phase == TouchPhase.Moved && Input.touchCount == 2)
            {
                // Pinch to scale
                Touch touch2 = Input.GetTouch(1);
                HandlePinchGesture(touch, touch2);
            }
        }
        
        private void HandleObjectSelection(Vector3 hitPosition)
        {
            Collider[] colliders = Physics.OverlapSphere(hitPosition, 0.1f);
            
            foreach (var collider in colliders)
            {
                GameObject obj = collider.gameObject;
                
                if (switches.ContainsValue(obj))
                {
                    SelectSwitch(obj);
                    break;
                }
            }
        }
        
        private void SelectSwitch(GameObject switchObj)
        {
            // Deselect previous
            if (selectedObject != null)
            {
                var prevVisualizer = selectedObject.GetComponent<SwitchVisualizer>();
                prevVisualizer?.SetSelected(false);
            }
            
            // Select new
            selectedObject = switchObj;
            var visualizer = switchObj.GetComponent<SwitchVisualizer>();
            visualizer?.SetSelected(true);
            
            // Show details panel
            ShowSwitchDetails(switchObj.name);
        }
        
        private void ShowSwitchDetails(string switchId)
        {
            StartCoroutine(LoadSwitchDetails(switchId));
        }
        
        private IEnumerator LoadSwitchDetails(string switchId)
        {
            string url = $"{apiUrl}/v2/network/switches/{switchId}";
            
            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                request.SetRequestHeader("Authorization", $"Bearer {apiKey}");
                
                yield return request.SendWebRequest();
                
                if (request.result == UnityWebRequest.Result.Success)
                {
                    var details = JsonConvert.DeserializeObject<SwitchDetails>(request.downloadHandler.text);
                    DisplaySwitchInfo(details);
                }
            }
        }
        
        private void DisplaySwitchInfo(SwitchDetails details)
        {
            infoPanel.SetActive(true);
            
            statusText.text = $"Switch: {details.id}\n" +
                            $"Model: {details.model}\n" +
                            $"Status: {details.status}\n" +
                            $"Uptime: {FormatUptime(details.uptime)}\n" +
                            $"Active Ports: {details.activePorts}/{details.totalPorts}";
            
            metricsText.text = $"Bandwidth: {details.bandwidth:F1} Gbps\n" +
                             $"Latency: {details.latency:F2} ms\n" +
                             $"Packet Loss: {details.packetLoss:F3}%\n" +
                             $"Temperature: {details.temperature:F1}°C";
            
            bandwidthSlider.value = details.bandwidth / 1.0f; // Normalize to 1 Gbps
            
            // Update latency indicator color
            if (details.latency < 1)
                latencyIndicator.color = Color.green;
            else if (details.latency < 5)
                latencyIndicator.color = Color.yellow;
            else
                latencyIndicator.color = Color.red;
        }
        
        #endregion
        
        #region Voice Commands (HoloLens)
        
        private void SetupVoiceCommands()
        {
            var voiceCommands = new Dictionary<string, Action>
            {
                { "show topology", () => ShowFullTopology() },
                { "show alerts", () => ShowAlerts() },
                { "show metrics", () => ShowMetrics() },
                { "show FRER streams", () => ShowFRERStreams() },
                { "hide all", () => HideAllPanels() },
                { "reset view", () => ResetVisualization() },
                { "zoom in", () => AdjustScale(1.2f) },
                { "zoom out", () => AdjustScale(0.8f) },
                { "rotate left", () => RotateVisualization(-30f) },
                { "rotate right", () => RotateVisualization(30f) }
            };
            
            foreach (var command in voiceCommands)
            {
                PhraseRecognitionSystem.CreateOrUpdateKeyword(command.Key, command.Value);
            }
        }
        
        private void OnHandDetected(Hand hand)
        {
            // Process hand gestures
            if (hand.IsPinching)
            {
                HandlePinchGesture(hand);
            }
            else if (hand.IsTapping)
            {
                HandleTapGesture(hand);
            }
        }
        
        private void HandlePinchGesture(Hand hand)
        {
            // Scale visualization based on pinch distance
            float pinchDistance = hand.GetPinchDistance();
            AdjustScale(pinchDistance);
        }
        
        private void HandleTapGesture(Hand hand)
        {
            // Select object at tap position
            Ray ray = new Ray(hand.Position, hand.Forward);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit, 5f))
            {
                HandleObjectSelection(hit.point);
            }
        }
        
        #endregion
        
        #region Real-time Updates
        
        private IEnumerator UpdateVisualization()
        {
            while (true)
            {
                if (isVisualizationActive)
                {
                    // Update switch states
                    foreach (var kvp in switches)
                    {
                        UpdateSwitchVisualization(kvp.Value);
                    }
                    
                    // Update connection states
                    foreach (var kvp in connections)
                    {
                        UpdateConnectionVisualization(kvp.Value);
                    }
                    
                    // Check for alerts
                    ProcessActiveAlerts();
                }
                
                yield return new WaitForSeconds(updateInterval);
            }
        }
        
        private void UpdateMetrics(dynamic metricsData)
        {
            currentMetrics = JsonConvert.DeserializeObject<NetworkMetrics>(metricsData.ToString());
            
            // Update visualization based on new metrics
            foreach (var switchMetric in currentMetrics.switches)
            {
                if (switches.ContainsKey(switchMetric.id))
                {
                    var visualizer = switches[switchMetric.id].GetComponent<SwitchVisualizer>();
                    visualizer?.UpdateMetrics(switchMetric);
                }
            }
        }
        
        private void ShowAlert(dynamic alertData)
        {
            Alert alert = JsonConvert.DeserializeObject<Alert>(alertData.ToString());
            activeAlerts.Add(alert);
            
            // Create visual alert
            if (switches.ContainsKey(alert.switchId))
            {
                CreateAlertVisualization(switches[alert.switchId], alert);
            }
            
            // Show notification
            ShowAlertNotification(alert);
        }
        
        private void CreateAlertVisualization(GameObject switchObj, Alert alert)
        {
            GameObject alertObj = Instantiate(alertPrefab, switchObj.transform);
            alertObj.transform.localPosition = Vector3.up * 0.5f;
            
            // Configure alert appearance based on severity
            Material alertMat = alert.severity switch
            {
                "critical" => criticalMaterial,
                "warning" => warningMaterial,
                _ => normalMaterial
            };
            
            alertObj.GetComponent<Renderer>().material = alertMat;
            
            // Animate alert
            var animator = alertObj.AddComponent<AlertAnimator>();
            animator.StartPulsing();
            
            // Auto-destroy after 30 seconds
            Destroy(alertObj, 30f);
        }
        
        private void ShowFREREvent(dynamic eventData)
        {
            FREREvent frerEvent = JsonConvert.DeserializeObject<FREREvent>(eventData.ToString());
            
            // Visualize FRER recovery
            if (frerStreams.ContainsKey(frerEvent.streamId))
            {
                StartCoroutine(AnimateFRERRecovery(frerEvent));
            }
        }
        
        private IEnumerator AnimateFRERRecovery(FREREvent frerEvent)
        {
            // Find FRER path objects
            GameObject primaryPath = GameObject.Find($"FRER_{frerEvent.streamId}_Primary");
            GameObject secondaryPath = GameObject.Find($"FRER_{frerEvent.streamId}_Secondary");
            
            if (frerEvent.pathFailed == "primary" && secondaryPath != null)
            {
                // Highlight secondary path activation
                var line = secondaryPath.GetComponent<LineRenderer>();
                Color originalColor = line.material.color;
                
                for (int i = 0; i < 5; i++)
                {
                    line.material.color = Color.green;
                    yield return new WaitForSeconds(0.2f);
                    line.material.color = originalColor;
                    yield return new WaitForSeconds(0.2f);
                }
            }
        }
        
        #endregion
        
        #region Utility Methods
        
        private void UpdateSwitchMaterial(GameObject switchObj, string status)
        {
            Renderer renderer = switchObj.GetComponent<Renderer>();
            
            renderer.material = status switch
            {
                "active" => normalMaterial,
                "warning" => warningMaterial,
                "error" => criticalMaterial,
                _ => normalMaterial
            };
        }
        
        private void UpdateDataFlowAnimations()
        {
            foreach (var connection in connections.Values)
            {
                var animator = connection.GetComponent<DataFlowAnimator>();
                animator?.UpdateAnimation();
            }
        }
        
        private void UpdateMetricsDisplay()
        {
            if (currentMetrics != null && metricsText != null)
            {
                metricsText.text = $"Network Status\n" +
                                 $"Total Bandwidth: {currentMetrics.totalBandwidth:F1} Gbps\n" +
                                 $"Avg Latency: {currentMetrics.avgLatency:F2} ms\n" +
                                 $"Active Streams: {frerStreams.Count}\n" +
                                 $"Alerts: {activeAlerts.Count}";
            }
        }
        
        private string FormatUptime(int seconds)
        {
            TimeSpan time = TimeSpan.FromSeconds(seconds);
            return $"{time.Days}d {time.Hours}h {time.Minutes}m";
        }
        
        private Vector3 GetPlacementPosition()
        {
            // Find suitable AR plane for placement
            var planes = planeManager.trackables;
            foreach (var plane in planes)
            {
                if (plane.alignment == PlaneAlignment.HorizontalUp)
                {
                    return plane.center + Vector3.up * 0.1f;
                }
            }
            
            // Default position if no plane found
            return Camera.main.transform.position + Camera.main.transform.forward * 2f;
        }
        
        private void ClearVisualization()
        {
            foreach (var switchObj in switches.Values)
            {
                Destroy(switchObj);
            }
            switches.Clear();
            
            foreach (var connection in connections.Values)
            {
                Destroy(connection);
            }
            connections.Clear();
            
            frerStreams.Clear();
        }
        
        #endregion
    }
}