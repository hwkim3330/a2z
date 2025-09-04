#!/usr/bin/env python3
"""
A2Z Real-time ML Anomaly Detection System
Advanced machine learning for TSN/FRER network anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import asyncio
import threading
from collections import deque
import pickle
import json
import logging
from enum import Enum

# Machine Learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Real-time processing
from river import anomaly
from river import preprocessing as river_prep
from river import metrics as river_metrics
import redis
import kafka
from influxdb_client import InfluxDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('A2Z-ML-Anomaly')


class AnomalyType(Enum):
    NETWORK_INTRUSION = "network_intrusion"
    BANDWIDTH_SPIKE = "bandwidth_spike"
    LATENCY_ANOMALY = "latency_anomaly"
    PACKET_LOSS = "packet_loss"
    FRER_FAILURE = "frer_failure"
    HARDWARE_FAULT = "hardware_fault"
    PROTOCOL_VIOLATION = "protocol_violation"
    TIMING_VIOLATION = "timing_violation"
    SECURITY_BREACH = "security_breach"
    UNKNOWN = "unknown"


@dataclass
class NetworkMetrics:
    """Real-time network metrics"""
    timestamp: datetime
    switch_id: str
    port_id: int
    bandwidth_mbps: float
    latency_ms: float
    jitter_ms: float
    packet_loss_rate: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    temperature: float
    frer_recovery_events: int
    queue_depth: int
    dropped_packets: int
    crc_errors: int
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector"""
        return np.array([
            self.bandwidth_mbps,
            self.latency_ms,
            self.jitter_ms,
            self.packet_loss_rate,
            self.error_rate,
            self.cpu_usage,
            self.memory_usage,
            self.temperature,
            self.frer_recovery_events,
            self.queue_depth,
            self.dropped_packets,
            self.crc_errors
        ])


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: float  # 0-1 scale
    confidence: float  # 0-1 scale
    affected_component: str
    description: str
    metrics: NetworkMetrics
    recommended_action: str
    ml_model: str
    additional_data: Dict[str, Any] = field(default_factory=dict)


class DeepAutoencoder(nn.Module):
    """PyTorch Deep Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super(DeepAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class LSTMPredictor(keras.Model):
    """TensorFlow LSTM for time series anomaly prediction"""
    
    def __init__(self, sequence_length: int, n_features: int):
        super(LSTMPredictor, self).__init__()
        
        self.lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)
        self.lstm2 = layers.LSTM(64, return_sequences=True, dropout=0.2)
        self.lstm3 = layers.LSTM(32, dropout=0.2)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(n_features)
        
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)


class TransformerAnomalyDetector(nn.Module):
    """Transformer-based anomaly detector"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 3):
        super(TransformerAnomalyDetector, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Normal vs Anomaly
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=0)
        
        # Classification
        output = self.classifier(x)
        
        return output


class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detection models"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        
        # Initialize models
        self._initialize_models()
        
        # Model performance tracking
        self.model_performance = {
            model: {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
            for model in self.models.keys()
        }
        
        # Adaptive weights based on performance
        self._update_weights()
    
    def _initialize_models(self):
        """Initialize ensemble models"""
        # Classical ML models
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        self.models['one_class_svm'] = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.1
        )
        
        self.models['dbscan'] = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        
        # Deep learning models
        self.models['autoencoder'] = DeepAutoencoder(input_dim=12)
        self.models['lstm'] = LSTMPredictor(sequence_length=24, n_features=12)
        self.models['transformer'] = TransformerAnomalyDetector(input_dim=12)
        
        # Online learning model
        self.models['online_forest'] = anomaly.HalfSpaceTrees(
            n_trees=10,
            height=8,
            window_size=250,
            seed=42
        )
    
    def _update_weights(self):
        """Update model weights based on performance"""
        total_f1 = sum(perf['f1'] for perf in self.model_performance.values())
        
        if total_f1 > 0:
            for model_name, performance in self.model_performance.items():
                self.weights[model_name] = performance['f1'] / total_f1
        else:
            # Equal weights if no performance data
            n_models = len(self.models)
            for model_name in self.models.keys():
                self.weights[model_name] = 1.0 / n_models
    
    def train(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """Train ensemble models"""
        logger.info("Training ensemble anomaly detector...")
        
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X_train)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Train classical models
        self.models['isolation_forest'].fit(X_pca)
        self.models['one_class_svm'].fit(X_pca)
        
        if y_train is not None:
            # Train supervised models if labels available
            self._train_supervised_models(X_scaled, y_train)
        
        # Train deep learning models
        self._train_deep_models(X_scaled)
        
        logger.info("Ensemble training completed")
    
    def _train_supervised_models(self, X: np.ndarray, y: np.ndarray):
        """Train supervised models when labels are available"""
        # Random Forest for comparison
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        self.models['random_forest'] = rf
    
    def _train_deep_models(self, X: np.ndarray):
        """Train deep learning models"""
        # Train PyTorch Autoencoder
        self._train_autoencoder(X)
        
        # Train TensorFlow LSTM
        self._train_lstm(X)
        
        # Train Transformer
        self._train_transformer(X)
    
    def _train_autoencoder(self, X: np.ndarray):
        """Train autoencoder model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.models['autoencoder'].to(device)
        
        # Prepare data
        tensor_X = torch.FloatTensor(X).to(device)
        dataset = TensorDataset(tensor_X, tensor_X)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        epochs = 50
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.debug(f"Autoencoder Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def _train_lstm(self, X: np.ndarray):
        """Train LSTM model"""
        # Prepare sequences
        sequence_length = 24
        sequences = self._create_sequences(X, sequence_length)
        
        if len(sequences) == 0:
            return
        
        # Split data
        train_size = int(0.8 * len(sequences))
        X_train = sequences[:train_size]
        X_val = sequences[train_size:]
        
        # Train model
        model = self.models['lstm']
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5)
        
        model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
    
    def _train_transformer(self, X: np.ndarray):
        """Train transformer model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.models['transformer'].to(device)
        
        # Prepare sequences
        sequences = self._create_sequences(X, 24)
        if len(sequences) == 0:
            return
        
        # Create labels (assuming normal data for training)
        labels = torch.zeros(len(sequences), dtype=torch.long)
        
        tensor_X = torch.FloatTensor(sequences).to(device)
        dataset = TensorDataset(tensor_X, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        epochs = 30
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.debug(f"Transformer Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for time series models"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i+sequence_length])
        return np.array(sequences)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble prediction"""
        # Preprocess
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        predictions = {}
        scores = {}
        
        # Isolation Forest
        iso_pred = self.models['isolation_forest'].predict(X_pca)
        predictions['isolation_forest'] = (iso_pred == -1).astype(int)
        scores['isolation_forest'] = -self.models['isolation_forest'].score_samples(X_pca)
        
        # One-Class SVM
        svm_pred = self.models['one_class_svm'].predict(X_pca)
        predictions['one_class_svm'] = (svm_pred == -1).astype(int)
        scores['one_class_svm'] = -self.models['one_class_svm'].score_samples(X_pca)
        
        # Autoencoder (PyTorch)
        ae_scores = self._autoencoder_predict(X_scaled)
        ae_threshold = np.percentile(ae_scores, 90)
        predictions['autoencoder'] = (ae_scores > ae_threshold).astype(int)
        scores['autoencoder'] = ae_scores
        
        # Weighted ensemble
        ensemble_scores = np.zeros(len(X))
        for model_name, model_scores in scores.items():
            if model_name in self.weights:
                # Normalize scores to 0-1 range
                normalized = (model_scores - model_scores.min()) / (model_scores.max() - model_scores.min() + 1e-10)
                ensemble_scores += self.weights[model_name] * normalized
        
        # Final prediction based on ensemble scores
        threshold = np.percentile(ensemble_scores, 90)
        ensemble_pred = (ensemble_scores > threshold).astype(int)
        
        return ensemble_pred, ensemble_scores
    
    def _autoencoder_predict(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores from autoencoder"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.models['autoencoder'].to(device)
        model.eval()
        
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).to(device)
            reconstructed = model(tensor_X)
            mse = ((tensor_X - reconstructed) ** 2).mean(dim=1).cpu().numpy()
        
        return mse
    
    def update_online(self, x: np.ndarray) -> float:
        """Update online learning model with new data"""
        # Update Half-Space Trees
        score = self.models['online_forest'].score_one(dict(enumerate(x.flatten())))
        self.models['online_forest'].learn_one(dict(enumerate(x.flatten())))
        
        return score


class RealtimeAnomalyEngine:
    """Real-time anomaly detection engine"""
    
    def __init__(self, redis_host: str = 'localhost', kafka_host: str = 'localhost:9092'):
        self.ensemble = EnsembleAnomalyDetector()
        self.buffer = deque(maxlen=10000)
        self.anomaly_history = deque(maxlen=1000)
        
        # Real-time data sources
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        self.kafka_producer = None
        self.influx_client = None
        
        # Metrics tracking
        self.metrics = {
            'total_processed': 0,
            'total_anomalies': 0,
            'false_positives': 0,
            'true_positives': 0,
            'processing_time_ms': deque(maxlen=1000)
        }
        
        # Start background threads
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_stream)
        self.processing_thread.start()
    
    def _process_stream(self):
        """Process real-time data stream"""
        while self.running:
            try:
                # Get data from Redis stream
                messages = self.redis_client.xread({'network_metrics': '$'}, block=1000, count=10)
                
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        self._process_message(data)
                
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                time.sleep(1)
    
    def _process_message(self, data: Dict[str, Any]):
        """Process individual message"""
        start_time = time.time()
        
        try:
            # Parse metrics
            metrics = NetworkMetrics(
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                switch_id=data.get('switch_id', 'unknown'),
                port_id=int(data.get('port_id', 0)),
                bandwidth_mbps=float(data.get('bandwidth_mbps', 0)),
                latency_ms=float(data.get('latency_ms', 0)),
                jitter_ms=float(data.get('jitter_ms', 0)),
                packet_loss_rate=float(data.get('packet_loss_rate', 0)),
                error_rate=float(data.get('error_rate', 0)),
                cpu_usage=float(data.get('cpu_usage', 0)),
                memory_usage=float(data.get('memory_usage', 0)),
                temperature=float(data.get('temperature', 0)),
                frer_recovery_events=int(data.get('frer_recovery_events', 0)),
                queue_depth=int(data.get('queue_depth', 0)),
                dropped_packets=int(data.get('dropped_packets', 0)),
                crc_errors=int(data.get('crc_errors', 0))
            )
            
            # Add to buffer
            self.buffer.append(metrics)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(metrics)
            
            # Process detected anomalies
            for anomaly in anomalies:
                self._handle_anomaly(anomaly)
            
            # Update metrics
            self.metrics['total_processed'] += 1
            self.metrics['processing_time_ms'].append((time.time() - start_time) * 1000)
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def detect_anomalies(self, metrics: NetworkMetrics) -> List[AnomalyDetection]:
        """Detect anomalies in network metrics"""
        anomalies = []
        
        # Convert to feature vector
        feature_vector = metrics.to_vector().reshape(1, -1)
        
        # Get ensemble prediction
        predictions, scores = self.ensemble.predict(feature_vector)
        
        if predictions[0] == 1:
            # Anomaly detected
            anomaly_type = self._classify_anomaly_type(metrics)
            severity = min(scores[0], 1.0)
            
            anomaly = AnomalyDetection(
                timestamp=metrics.timestamp,
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=self._calculate_confidence(scores[0]),
                affected_component=f"{metrics.switch_id}:{metrics.port_id}",
                description=self._generate_description(anomaly_type, metrics),
                metrics=metrics,
                recommended_action=self._recommend_action(anomaly_type, severity),
                ml_model="ensemble",
                additional_data={'raw_score': float(scores[0])}
            )
            
            anomalies.append(anomaly)
        
        # Specific checks
        anomalies.extend(self._check_specific_anomalies(metrics))
        
        return anomalies
    
    def _classify_anomaly_type(self, metrics: NetworkMetrics) -> AnomalyType:
        """Classify type of anomaly based on metrics"""
        # Rule-based classification
        if metrics.bandwidth_mbps > 900:  # Near gigabit limit
            return AnomalyType.BANDWIDTH_SPIKE
        elif metrics.latency_ms > 50:
            return AnomalyType.LATENCY_ANOMALY
        elif metrics.packet_loss_rate > 0.01:
            return AnomalyType.PACKET_LOSS
        elif metrics.frer_recovery_events > 10:
            return AnomalyType.FRER_FAILURE
        elif metrics.temperature > 80:
            return AnomalyType.HARDWARE_FAULT
        elif metrics.crc_errors > 100:
            return AnomalyType.PROTOCOL_VIOLATION
        else:
            return AnomalyType.UNKNOWN
    
    def _check_specific_anomalies(self, metrics: NetworkMetrics) -> List[AnomalyDetection]:
        """Check for specific anomaly patterns"""
        anomalies = []
        
        # FRER timing violation
        if metrics.latency_ms > 1.0 and metrics.jitter_ms > 0.5:
            anomalies.append(AnomalyDetection(
                timestamp=metrics.timestamp,
                anomaly_type=AnomalyType.TIMING_VIOLATION,
                severity=0.8,
                confidence=0.9,
                affected_component=f"{metrics.switch_id}:{metrics.port_id}",
                description="FRER timing requirements violated",
                metrics=metrics,
                recommended_action="Check time synchronization and network load",
                ml_model="rule_based",
                additional_data={}
            ))
        
        # Security breach detection
        if metrics.error_rate > 0.1 and metrics.dropped_packets > 1000:
            anomalies.append(AnomalyDetection(
                timestamp=metrics.timestamp,
                anomaly_type=AnomalyType.SECURITY_BREACH,
                severity=0.9,
                confidence=0.7,
                affected_component=f"{metrics.switch_id}:{metrics.port_id}",
                description="Potential DDoS or security breach detected",
                metrics=metrics,
                recommended_action="Enable rate limiting and check firewall rules",
                ml_model="rule_based",
                additional_data={}
            ))
        
        return anomalies
    
    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence level from anomaly score"""
        # Sigmoid transformation
        return 1 / (1 + np.exp(-2 * (score - 0.5)))
    
    def _generate_description(self, anomaly_type: AnomalyType, metrics: NetworkMetrics) -> str:
        """Generate human-readable description"""
        descriptions = {
            AnomalyType.BANDWIDTH_SPIKE: f"Bandwidth spike detected: {metrics.bandwidth_mbps:.1f} Mbps",
            AnomalyType.LATENCY_ANOMALY: f"High latency detected: {metrics.latency_ms:.2f} ms",
            AnomalyType.PACKET_LOSS: f"Packet loss rate: {metrics.packet_loss_rate:.4%}",
            AnomalyType.FRER_FAILURE: f"FRER recovery events: {metrics.frer_recovery_events}",
            AnomalyType.HARDWARE_FAULT: f"Temperature critical: {metrics.temperature}°C",
            AnomalyType.PROTOCOL_VIOLATION: f"CRC errors detected: {metrics.crc_errors}",
            AnomalyType.TIMING_VIOLATION: "TSN timing requirements not met",
            AnomalyType.SECURITY_BREACH: "Potential security threat detected",
            AnomalyType.UNKNOWN: "Unknown anomaly pattern detected"
        }
        return descriptions.get(anomaly_type, "Anomaly detected")
    
    def _recommend_action(self, anomaly_type: AnomalyType, severity: float) -> str:
        """Recommend action based on anomaly type and severity"""
        if severity > 0.8:
            prefix = "URGENT: "
        elif severity > 0.5:
            prefix = "RECOMMENDED: "
        else:
            prefix = "SUGGESTED: "
        
        actions = {
            AnomalyType.BANDWIDTH_SPIKE: "Enable traffic shaping and check for bandwidth hogs",
            AnomalyType.LATENCY_ANOMALY: "Check network congestion and QoS settings",
            AnomalyType.PACKET_LOSS: "Inspect physical connections and switch buffers",
            AnomalyType.FRER_FAILURE: "Review FRER configuration and path redundancy",
            AnomalyType.HARDWARE_FAULT: "Check cooling system and consider hardware replacement",
            AnomalyType.PROTOCOL_VIOLATION: "Verify protocol compliance and cable quality",
            AnomalyType.TIMING_VIOLATION: "Reconfigure PTP/gPTP and check time synchronization",
            AnomalyType.SECURITY_BREACH: "Activate security protocols and review access logs",
            AnomalyType.UNKNOWN: "Perform detailed system diagnostics"
        }
        
        return prefix + actions.get(anomaly_type, "Monitor and investigate")
    
    def _handle_anomaly(self, anomaly: AnomalyDetection):
        """Handle detected anomaly"""
        # Add to history
        self.anomaly_history.append(anomaly)
        self.metrics['total_anomalies'] += 1
        
        # Log anomaly
        logger.warning(f"Anomaly detected: {anomaly.anomaly_type.value} - {anomaly.description}")
        
        # Send alert
        self._send_alert(anomaly)
        
        # Store in database
        self._store_anomaly(anomaly)
        
        # Trigger automated response if critical
        if anomaly.severity > 0.8:
            self._trigger_automated_response(anomaly)
    
    def _send_alert(self, anomaly: AnomalyDetection):
        """Send anomaly alert"""
        alert_data = {
            'timestamp': anomaly.timestamp.isoformat(),
            'type': anomaly.anomaly_type.value,
            'severity': anomaly.severity,
            'component': anomaly.affected_component,
            'description': anomaly.description,
            'action': anomaly.recommended_action
        }
        
        # Publish to Redis
        self.redis_client.publish('anomaly_alerts', json.dumps(alert_data))
        
        # Send to Kafka if available
        if self.kafka_producer:
            self.kafka_producer.send('anomalies', value=alert_data)
    
    def _store_anomaly(self, anomaly: AnomalyDetection):
        """Store anomaly in database"""
        # Store in InfluxDB if available
        if self.influx_client:
            point = {
                "measurement": "anomalies",
                "tags": {
                    "type": anomaly.anomaly_type.value,
                    "component": anomaly.affected_component,
                    "model": anomaly.ml_model
                },
                "time": anomaly.timestamp,
                "fields": {
                    "severity": anomaly.severity,
                    "confidence": anomaly.confidence,
                    "bandwidth": anomaly.metrics.bandwidth_mbps,
                    "latency": anomaly.metrics.latency_ms,
                    "packet_loss": anomaly.metrics.packet_loss_rate
                }
            }
            # Write to InfluxDB
    
    def _trigger_automated_response(self, anomaly: AnomalyDetection):
        """Trigger automated response for critical anomalies"""
        logger.critical(f"Triggering automated response for {anomaly.anomaly_type.value}")
        
        # Implement automated responses
        if anomaly.anomaly_type == AnomalyType.BANDWIDTH_SPIKE:
            # Enable rate limiting
            self._enable_rate_limiting(anomaly.affected_component)
        elif anomaly.anomaly_type == AnomalyType.SECURITY_BREACH:
            # Isolate affected component
            self._isolate_component(anomaly.affected_component)
        elif anomaly.anomaly_type == AnomalyType.HARDWARE_FAULT:
            # Initiate failover
            self._initiate_failover(anomaly.affected_component)
    
    def _enable_rate_limiting(self, component: str):
        """Enable rate limiting on component"""
        command = {
            'action': 'enable_rate_limiting',
            'component': component,
            'limit': '100mbps'
        }
        self.redis_client.publish('network_commands', json.dumps(command))
    
    def _isolate_component(self, component: str):
        """Isolate component from network"""
        command = {
            'action': 'isolate',
            'component': component
        }
        self.redis_client.publish('network_commands', json.dumps(command))
    
    def _initiate_failover(self, component: str):
        """Initiate failover for component"""
        command = {
            'action': 'failover',
            'component': component
        }
        self.redis_client.publish('network_commands', json.dumps(command))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        avg_processing_time = np.mean(self.metrics['processing_time_ms']) if self.metrics['processing_time_ms'] else 0
        
        return {
            'total_processed': self.metrics['total_processed'],
            'total_anomalies': self.metrics['total_anomalies'],
            'detection_rate': self.metrics['total_anomalies'] / max(self.metrics['total_processed'], 1),
            'avg_processing_time_ms': avg_processing_time,
            'buffer_size': len(self.buffer),
            'anomaly_history_size': len(self.anomaly_history),
            'model_weights': self.ensemble.weights
        }
    
    def train_incremental(self):
        """Incremental training with recent data"""
        if len(self.buffer) < 100:
            return
        
        # Get recent data
        recent_data = list(self.buffer)[-1000:]
        X = np.array([m.to_vector() for m in recent_data])
        
        # Update ensemble
        self.ensemble.train(X)
        
        logger.info("Incremental training completed")
    
    def shutdown(self):
        """Shutdown engine"""
        self.running = False
        self.processing_thread.join()
        logger.info("Anomaly detection engine shutdown")


def main():
    """Main execution"""
    logger.info("Starting A2Z Real-time ML Anomaly Detection System")
    
    # Initialize engine
    engine = RealtimeAnomalyEngine()
    
    # Train initial models with historical data
    # Load historical data here
    # engine.ensemble.train(historical_data)
    
    try:
        # Run for demonstration
        time.sleep(60)
        
        # Get statistics
        stats = engine.get_statistics()
        logger.info(f"Statistics: {stats}")
        
        # Perform incremental training
        engine.train_incremental()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()