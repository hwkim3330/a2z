#!/usr/bin/env python3
"""
A2Z Real-time Edge AI Inference Engine
엣지 디바이스에서 실시간 AI 추론을 위한 초경량 엔진
"""

import asyncio
import numpy as np
import cv2
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque
import logging

# 엣지 AI 라이브러리
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# OpenVINO (Intel)
try:
    from openvino.inference_engine import IECore
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# Coral Edge TPU
try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
    from pycoral.adapters import classify
    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False

class InferenceBackend(Enum):
    """추론 백엔드 종류"""
    TENSORRT = "tensorrt"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    OPENVINO = "openvino"
    CORAL_TPU = "coral"
    CPU_OPTIMIZED = "cpu"

@dataclass
class InferenceResult:
    """추론 결과"""
    model_name: str
    inference_time: float
    confidence: float
    predictions: Dict[str, Any]
    timestamp: float
    frame_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """모델 구성 정보"""
    name: str
    path: str
    backend: InferenceBackend
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    preprocessing: Dict[str, Any]
    postprocessing: Dict[str, Any]
    batch_size: int = 1
    precision: str = "fp16"
    device: str = "cuda:0"

class TensorRTInferenceEngine:
    """TensorRT 추론 엔진"""
    
    def __init__(self, model_path: str, logger: Optional[logging.Logger] = None):
        self.model_path = model_path
        self.logger = logger or logging.getLogger(__name__)
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        if TENSORRT_AVAILABLE:
            self._load_engine()
    
    def _load_engine(self):
        """TensorRT 엔진 로드"""
        try:
            # TensorRT 로거 설정
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # 엔진 파일 로드
            with open(self.model_path, 'rb') as f:
                self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            # 입출력 바인딩 설정
            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                size = trt.volume(self.context.get_binding_shape(binding_idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                
                # GPU 메모리 할당
                device_mem = cuda.mem_alloc(size * dtype().itemsize)
                
                self.bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(binding):
                    self.inputs.append({
                        'binding': binding,
                        'device_mem': device_mem,
                        'size': size,
                        'dtype': dtype
                    })
                else:
                    self.outputs.append({
                        'binding': binding,
                        'device_mem': device_mem,
                        'size': size,
                        'dtype': dtype
                    })
            
            # CUDA 스트림 생성
            self.stream = cuda.Stream()
            
            self.logger.info(f"TensorRT 엔진 로드 완료: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"TensorRT 엔진 로드 실패: {e}")
            raise
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """추론 실행"""
        try:
            # 입력 데이터를 GPU로 복사
            cuda.memcpy_htod_async(
                self.inputs[0]['device_mem'],
                input_data,
                self.stream
            )
            
            # 추론 실행
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )
            
            # 결과를 CPU로 복사
            output = np.empty(
                self.outputs[0]['size'],
                dtype=self.outputs[0]['dtype']
            )
            
            cuda.memcpy_dtoh_async(
                output,
                self.outputs[0]['device_mem'],
                self.stream
            )
            
            # 스트림 동기화
            self.stream.synchronize()
            
            return output
            
        except Exception as e:
            self.logger.error(f"TensorRT 추론 실패: {e}")
            raise

class EdgeAIInferenceEngine:
    """통합 엣지 AI 추론 엔진"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.inference_queue = asyncio.Queue(maxsize=100)
        self.result_queue = asyncio.Queue(maxsize=100)
        self.logger = self._setup_logger()
        self.stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'average_inference_time': 0.0,
            'throughput': 0.0  # FPS
        }
        self.performance_history = deque(maxlen=1000)
        
        # 하드웨어 감지
        self.available_backends = self._detect_backends()
        self.logger.info(f"사용 가능한 백엔드: {list(self.available_backends.keys())}")
    
    def _setup_logger(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('EdgeAI')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _detect_backends(self) -> Dict[InferenceBackend, bool]:
        """사용 가능한 추론 백엔드 감지"""
        backends = {}
        
        # TensorRT 감지
        if TENSORRT_AVAILABLE:
            try:
                cuda.init()
                device_count = cuda.Device.count()
                if device_count > 0:
                    backends[InferenceBackend.TENSORRT] = True
                    self.logger.info(f"TensorRT 사용 가능 (GPU {device_count}개)")
            except:
                backends[InferenceBackend.TENSORRT] = False
        
        # PyTorch 감지
        if TORCH_AVAILABLE:
            backends[InferenceBackend.PYTORCH] = True
            cuda_available = torch.cuda.is_available()
            self.logger.info(f"PyTorch 사용 가능 (CUDA: {cuda_available})")
        
        # ONNX Runtime 감지
        if ONNX_AVAILABLE:
            providers = ort.get_available_providers()
            backends[InferenceBackend.ONNX] = True
            self.logger.info(f"ONNX Runtime 사용 가능 (Providers: {providers})")
        
        # OpenVINO 감지
        if OPENVINO_AVAILABLE:
            try:
                ie = IECore()
                devices = ie.available_devices
                backends[InferenceBackend.OPENVINO] = True
                self.logger.info(f"OpenVINO 사용 가능 (Devices: {devices})")
            except:
                backends[InferenceBackend.OPENVINO] = False
        
        # Coral TPU 감지
        if CORAL_AVAILABLE:
            try:
                tpu_devices = edgetpu.list_edge_tpus()
                if tpu_devices:
                    backends[InferenceBackend.CORAL_TPU] = True
                    self.logger.info(f"Coral TPU 사용 가능 ({len(tpu_devices)}개)")
            except:
                backends[InferenceBackend.CORAL_TPU] = False
        
        # CPU 최적화는 항상 사용 가능
        backends[InferenceBackend.CPU_OPTIMIZED] = True
        
        return backends
    
    def load_model(self, config: ModelConfig) -> bool:
        """모델 로드"""
        try:
            if config.backend not in self.available_backends:
                raise ValueError(f"백엔드 {config.backend}는 사용할 수 없습니다")
            
            if config.backend == InferenceBackend.TENSORRT:
                model = TensorRTInferenceEngine(config.path, self.logger)
            
            elif config.backend == InferenceBackend.PYTORCH:
                if TORCH_AVAILABLE:
                    model = torch.jit.load(config.path)
                    if torch.cuda.is_available() and 'cuda' in config.device:
                        model = model.cuda()
                    model.eval()
                else:
                    raise ImportError("PyTorch가 설치되지 않았습니다")
            
            elif config.backend == InferenceBackend.ONNX:
                if ONNX_AVAILABLE:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    model = ort.InferenceSession(config.path, providers=providers)
                else:
                    raise ImportError("ONNX Runtime이 설치되지 않았습니다")
            
            elif config.backend == InferenceBackend.OPENVINO:
                if OPENVINO_AVAILABLE:
                    ie = IECore()
                    net = ie.read_network(model=config.path)
                    model = ie.load_network(network=net, device_name='CPU')
                else:
                    raise ImportError("OpenVINO가 설치되지 않았습니다")
            
            elif config.backend == InferenceBackend.CORAL_TPU:
                if CORAL_AVAILABLE:
                    model = edgetpu.make_interpreter(config.path)
                    model.allocate_tensors()
                else:
                    raise ImportError("PyCoral이 설치되지 않았습니다")
            
            else:
                # CPU 최적화 버전 (NumPy 기반)
                model = self._load_cpu_model(config.path)
            
            self.models[config.name] = {
                'model': model,
                'config': config,
                'backend': config.backend
            }
            
            self.logger.info(f"모델 '{config.name}' 로드 완료 ({config.backend})")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 '{config.name}' 로드 실패: {e}")
            return False
    
    def _load_cpu_model(self, model_path: str):
        """CPU 최적화 모델 로드 (예시)"""
        # 실제로는 더 복잡한 CPU 최적화 로직이 필요
        return {
            'weights': np.load(model_path),
            'type': 'cpu_optimized'
        }
    
    async def infer_async(self, model_name: str, input_data: np.ndarray, 
                         frame_id: int = 0) -> Optional[InferenceResult]:
        """비동기 추론"""
        try:
            if model_name not in self.models:
                raise ValueError(f"모델 '{model_name}'이 로드되지 않았습니다")
            
            model_info = self.models[model_name]
            model = model_info['model']
            config = model_info['config']
            backend = model_info['backend']
            
            # 전처리
            preprocessed_data = self._preprocess(input_data, config.preprocessing)
            
            # 추론 실행
            start_time = time.time()
            
            if backend == InferenceBackend.TENSORRT:
                raw_output = model.infer(preprocessed_data)
            
            elif backend == InferenceBackend.PYTORCH:
                with torch.no_grad():
                    input_tensor = torch.from_numpy(preprocessed_data)
                    if torch.cuda.is_available() and 'cuda' in config.device:
                        input_tensor = input_tensor.cuda()
                    raw_output = model(input_tensor).cpu().numpy()
            
            elif backend == InferenceBackend.ONNX:
                input_name = model.get_inputs()[0].name
                raw_output = model.run(None, {input_name: preprocessed_data})[0]
            
            elif backend == InferenceBackend.OPENVINO:
                input_blob = next(iter(model.input_info))
                raw_output = model.infer(inputs={input_blob: preprocessed_data})
                raw_output = list(raw_output.values())[0]
            
            elif backend == InferenceBackend.CORAL_TPU:
                model.set_tensor(model.get_input_details()[0]['index'], preprocessed_data)
                model.invoke()
                raw_output = model.get_tensor(model.get_output_details()[0]['index'])
            
            else:
                # CPU 최적화 버전
                raw_output = self._cpu_inference(model, preprocessed_data)
            
            inference_time = time.time() - start_time
            
            # 후처리
            predictions = self._postprocess(raw_output, config.postprocessing)
            
            # 결과 생성
            result = InferenceResult(
                model_name=model_name,
                inference_time=inference_time,
                confidence=predictions.get('confidence', 0.0),
                predictions=predictions,
                timestamp=time.time(),
                frame_id=frame_id
            )
            
            # 통계 업데이트
            self._update_stats(inference_time, True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"추론 실패 ({model_name}): {e}")
            self._update_stats(0, False)
            return None
    
    def _preprocess(self, data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """입력 데이터 전처리"""
        try:
            processed = data.copy()
            
            # 리사이징
            if 'resize' in config:
                target_size = config['resize']
                processed = cv2.resize(processed, target_size)
            
            # 정규화
            if 'normalize' in config:
                norm_config = config['normalize']
                processed = processed.astype(np.float32)
                processed = (processed - norm_config.get('mean', 0)) / norm_config.get('std', 1)
            
            # 채널 순서 변경 (HWC -> CHW)
            if config.get('channels_first', False) and len(processed.shape) == 3:
                processed = np.transpose(processed, (2, 0, 1))
            
            # 배치 차원 추가
            if config.get('add_batch_dim', True):
                processed = np.expand_dims(processed, axis=0)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"전처리 실패: {e}")
            return data
    
    def _postprocess(self, raw_output: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """출력 데이터 후처리"""
        try:
            predictions = {}
            
            if config.get('type') == 'classification':
                # 분류 결과 처리
                probabilities = self._softmax(raw_output.flatten())
                predicted_class = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class])
                
                predictions.update({
                    'class_id': int(predicted_class),
                    'confidence': confidence,
                    'probabilities': probabilities.tolist()
                })
                
                # 클래스 라벨 매핑
                if 'class_labels' in config:
                    predictions['class_name'] = config['class_labels'][predicted_class]
            
            elif config.get('type') == 'detection':
                # 객체 탐지 결과 처리
                detections = self._parse_detection_output(
                    raw_output, 
                    config.get('confidence_threshold', 0.5),
                    config.get('nms_threshold', 0.4)
                )
                
                predictions.update({
                    'detections': detections,
                    'num_detections': len(detections)
                })
            
            elif config.get('type') == 'segmentation':
                # 세그멘테이션 결과 처리
                mask = np.argmax(raw_output, axis=1 if len(raw_output.shape) == 4 else 0)
                predictions.update({
                    'mask': mask,
                    'unique_classes': np.unique(mask).tolist()
                })
            
            else:
                # 일반적인 회귀 출력
                predictions['output'] = raw_output.tolist()
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"후처리 실패: {e}")
            return {'raw_output': raw_output.tolist()}
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """소프트맥스 함수"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _parse_detection_output(self, output: np.ndarray, conf_threshold: float, 
                               nms_threshold: float) -> List[Dict[str, Any]]:
        """객체 탐지 출력 파싱 (YOLO 형식 예시)"""
        detections = []
        
        # 실제로는 특정 모델 형식에 맞게 구현해야 함
        # 여기서는 간단한 예시만 제공
        
        return detections
    
    def _cpu_inference(self, model: Dict[str, Any], input_data: np.ndarray) -> np.ndarray:
        """CPU 최적화 추론 (예시)"""
        # 실제로는 더 복잡한 CPU 최적화 로직 구현 필요
        weights = model['weights']
        # 간단한 행렬 곱셈 예시
        output = np.dot(input_data.flatten(), weights)
        return output
    
    def _update_stats(self, inference_time: float, success: bool):
        """성능 통계 업데이트"""
        self.stats['total_inferences'] += 1
        
        if success:
            self.stats['successful_inferences'] += 1
            self.performance_history.append(inference_time)
            
            # 평균 추론 시간 계산
            if len(self.performance_history) > 0:
                self.stats['average_inference_time'] = np.mean(list(self.performance_history))
                self.stats['throughput'] = 1.0 / self.stats['average_inference_time']
        else:
            self.stats['failed_inferences'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        return {
            **self.stats,
            'success_rate': (self.stats['successful_inferences'] / 
                           max(self.stats['total_inferences'], 1)) * 100,
            'available_backends': list(self.available_backends.keys()),
            'loaded_models': list(self.models.keys())
        }
    
    async def benchmark_model(self, model_name: str, num_runs: int = 100) -> Dict[str, float]:
        """모델 벤치마크"""
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이 로드되지 않았습니다")
        
        config = self.models[model_name]['config']
        dummy_input = np.random.randn(*config.input_shape).astype(np.float32)
        
        inference_times = []
        
        # 워밍업
        for _ in range(10):
            await self.infer_async(model_name, dummy_input)
        
        # 벤치마크 실행
        start_time = time.time()
        for i in range(num_runs):
            result = await self.infer_async(model_name, dummy_input, frame_id=i)
            if result:
                inference_times.append(result.inference_time)
        
        total_time = time.time() - start_time
        
        return {
            'average_time': np.mean(inference_times),
            'min_time': np.min(inference_times),
            'max_time': np.max(inference_times),
            'std_time': np.std(inference_times),
            'throughput': num_runs / total_time,
            'total_runs': num_runs,
            'successful_runs': len(inference_times)
        }

class RealTimeVideoProcessor:
    """실시간 비디오 처리기"""
    
    def __init__(self, inference_engine: EdgeAIInferenceEngine):
        self.inference_engine = inference_engine
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.frame_queue = asyncio.Queue(maxsize=10)
        self.result_queue = asyncio.Queue(maxsize=10)
    
    async def process_video_stream(self, source: str, model_name: str, 
                                  output_callback=None):
        """비디오 스트림 실시간 처리"""
        try:
            # 비디오 캡처 초기화
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"비디오 소스를 열 수 없습니다: {source}")
            
            self.running = True
            frame_id = 0
            
            # 비디오 처리 루프
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 추론 실행
                result = await self.inference_engine.infer_async(
                    model_name, frame, frame_id
                )
                
                if result and output_callback:
                    await output_callback(frame, result)
                
                frame_id += 1
                
                # 프레임 레이트 조절
                await asyncio.sleep(0.033)  # ~30 FPS
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"비디오 처리 실패: {e}")
        finally:
            self.running = False
    
    def stop(self):
        """처리 중단"""
        self.running = False

# 사용 예시 및 테스트
async def main():
    """메인 실행 함수"""
    print("A2Z 엣지 AI 추론 엔진 시작...")
    
    # 추론 엔진 초기화
    engine = EdgeAIInferenceEngine()
    
    # 성능 통계 출력
    stats = engine.get_performance_stats()
    print(f"성능 통계: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 모델 로드 예시 (실제 모델 파일 필요)
    model_config = ModelConfig(
        name="object_detection",
        path="models/yolo_v8_trt.engine",  # TensorRT 모델
        backend=InferenceBackend.TENSORRT,
        input_shape=(1, 3, 640, 640),
        output_shape=(1, 25200, 85),
        preprocessing={
            'resize': (640, 640),
            'normalize': {'mean': 0, 'std': 255},
            'channels_first': True
        },
        postprocessing={
            'type': 'detection',
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4
        }
    )
    
    # 모델 로드 시도
    if engine.load_model(model_config):
        print(f"모델 '{model_config.name}' 로드 성공")
        
        # 벤치마크 실행
        try:
            benchmark_results = await engine.benchmark_model("object_detection", 50)
            print(f"벤치마크 결과: {json.dumps(benchmark_results, indent=2)}")
        except Exception as e:
            print(f"벤치마크 실패: {e}")
    else:
        print(f"모델 '{model_config.name}' 로드 실패")
    
    # 실시간 비디오 처리 예시 (웹캠)
    if False:  # 실제 테스트 시 True로 변경
        async def result_callback(frame, result):
            print(f"프레임 {result.frame_id}: "
                  f"추론시간 {result.inference_time:.3f}초, "
                  f"신뢰도 {result.confidence:.3f}")
        
        processor = RealTimeVideoProcessor(engine)
        await processor.process_video_stream(0, "object_detection", result_callback)
    
    print("엣지 AI 추론 엔진 테스트 완료")

if __name__ == "__main__":
    asyncio.run(main())