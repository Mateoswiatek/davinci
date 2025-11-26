"""
YOLO Object Detection Processor for Raspberry Pi 5

Async YOLO processing that doesn't block the main streaming pipeline.
Supports multiple backends: PyTorch, ONNX, NCNN, TFLite (EdgeTPU), Hailo.

Author: Claude Code
License: MIT
"""

import time
import logging
import threading
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any, Callable
from enum import Enum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class YOLOBackend(Enum):
    """Supported YOLO backends."""
    PYTORCH = "pytorch"      # Default ultralytics
    ONNX = "onnx"            # ONNX Runtime
    NCNN = "ncnn"            # Tencent NCNN (fastest CPU)
    TFLITE = "tflite"        # TensorFlow Lite
    EDGETPU = "edgetpu"      # Google Coral EdgeTPU
    HAILO = "hailo"          # Hailo-8L (fastest overall)
    OPENCV_DNN = "opencv_dnn"  # OpenCV DNN with Darknet (lightweight, no PyTorch needed)


class FrameSkipStrategy(Enum):
    """Strategies for frame skipping."""
    FIXED = "fixed"          # Process every N-th frame
    ADAPTIVE = "adaptive"    # Adapt based on processing time
    QUEUE = "queue"          # Use queue, drop old frames
    NONE = "none"            # Process every frame (not recommended)


@dataclass
class Detection:
    """Single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int

    def to_dict(self) -> dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area
        }


@dataclass
class DetectionResult:
    """Container for detection results with metadata."""
    detections: List[Detection]
    frame_id: int
    timestamp_ns: int
    inference_time_ms: float
    model_name: str
    input_size: Tuple[int, int]

    @property
    def count(self) -> int:
        return len(self.detections)

    def filter_by_confidence(self, min_confidence: float) -> 'DetectionResult':
        """Filter detections by minimum confidence."""
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectionResult(
            detections=filtered,
            frame_id=self.frame_id,
            timestamp_ns=self.timestamp_ns,
            inference_time_ms=self.inference_time_ms,
            model_name=self.model_name,
            input_size=self.input_size
        )

    def filter_by_class(self, class_ids: List[int]) -> 'DetectionResult':
        """Filter detections by class IDs."""
        filtered = [d for d in self.detections if d.class_id in class_ids]
        return DetectionResult(
            detections=filtered,
            frame_id=self.frame_id,
            timestamp_ns=self.timestamp_ns,
            inference_time_ms=self.inference_time_ms,
            model_name=self.model_name,
            input_size=self.input_size
        )


@dataclass
class YOLOConfig:
    """YOLO processor configuration."""
    model_path: str = "yolov8n.pt"
    backend: YOLOBackend = YOLOBackend.PYTORCH
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    device: str = "cpu"  # cpu, cuda, mps

    # Frame skipping
    skip_strategy: FrameSkipStrategy = FrameSkipStrategy.FIXED
    skip_n_frames: int = 3  # Process every 3rd frame = 10 FPS @ 30 FPS input

    # Adaptive strategy parameters
    target_inference_time_ms: float = 100.0
    min_skip: int = 1
    max_skip: int = 10

    # CPU pinning
    cpu_affinity: Optional[List[int]] = None  # e.g., [3] to pin to CPU 3

    # OpenCV DNN specific (for OPENCV_DNN backend)
    opencv_config_path: Optional[str] = None  # Path to .cfg file
    opencv_weights_path: Optional[str] = None  # Path to .weights file
    opencv_names_path: Optional[str] = None  # Path to .names file (class names)
    filter_classes: Optional[List[int]] = None  # Filter only specific classes (e.g., [0] for persons only)


class YOLOProcessor:
    """
    Async YOLO processor that doesn't block the main pipeline.

    Key features:
    - Frame skipping to maintain FPS
    - Cached detections for skipped frames
    - Multiple backend support
    - CPU pinning for real-time performance
    """

    # COCO class names (80 classes)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    def __init__(self, config: Optional[YOLOConfig] = None):
        self.config = config or YOLOConfig()
        self.model = None
        self.frame_counter = 0
        self.last_result: Optional[DetectionResult] = None

        # Statistics
        self._stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_inference_time_ms': 0.0,
            'avg_inference_time_ms': 0.0,
            'detections_per_frame': 0.0,
            'inference_times': deque(maxlen=100)
        }

        # Adaptive skip
        self._current_skip = self.config.skip_n_frames

    def initialize(self) -> bool:
        """
        Initialize the YOLO model.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Set CPU affinity if specified
            if self.config.cpu_affinity:
                import os
                os.sched_setaffinity(0, set(self.config.cpu_affinity))
                logger.info(f"Set CPU affinity to: {self.config.cpu_affinity}")

            # Load model based on backend
            if self.config.backend == YOLOBackend.PYTORCH:
                self._load_pytorch()
            elif self.config.backend == YOLOBackend.ONNX:
                self._load_onnx()
            elif self.config.backend == YOLOBackend.NCNN:
                self._load_ncnn()
            elif self.config.backend == YOLOBackend.TFLITE:
                self._load_tflite()
            elif self.config.backend == YOLOBackend.EDGETPU:
                self._load_edgetpu()
            elif self.config.backend == YOLOBackend.HAILO:
                self._load_hailo()
            elif self.config.backend == YOLOBackend.OPENCV_DNN:
                self._load_opencv_dnn()
            else:
                raise ValueError(f"Unsupported backend: {self.config.backend}")

            logger.info(f"YOLO model loaded: {self.config.model_path} ({self.config.backend.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            return False

    def _load_pytorch(self):
        """Load PyTorch/Ultralytics model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.config.model_path)
            self.model.to(self.config.device)
            logger.info("Loaded PyTorch/Ultralytics model")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

    def _load_onnx(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            # Use CPU execution provider for Raspberry Pi
            providers = ['CPUExecutionProvider']
            self.model = ort.InferenceSession(self.config.model_path, providers=providers)
            self._onnx_input_name = self.model.get_inputs()[0].name
            logger.info("Loaded ONNX model")
        except ImportError:
            logger.error("onnxruntime not installed. Run: pip install onnxruntime")
            raise

    def _load_ncnn(self):
        """Load NCNN model (fastest on CPU)."""
        try:
            # NCNN requires separate param and bin files
            param_path = self.config.model_path.replace('.bin', '.param')
            bin_path = self.config.model_path

            import ncnn
            self.model = ncnn.Net()
            self.model.opt.use_vulkan_compute = False  # CPU only on Pi
            self.model.opt.num_threads = 4
            self.model.load_param(param_path)
            self.model.load_model(bin_path)
            logger.info("Loaded NCNN model")
        except ImportError:
            logger.error("ncnn not installed. See: https://github.com/Tencent/ncnn")
            raise

    def _load_tflite(self):
        """Load TFLite model."""
        try:
            import tflite_runtime.interpreter as tflite
            self.model = tflite.Interpreter(model_path=self.config.model_path)
            self.model.allocate_tensors()
            self._tflite_input_details = self.model.get_input_details()
            self._tflite_output_details = self.model.get_output_details()
            logger.info("Loaded TFLite model")
        except ImportError:
            logger.error("tflite_runtime not installed")
            raise

    def _load_edgetpu(self):
        """Load EdgeTPU (Coral) model."""
        try:
            import tflite_runtime.interpreter as tflite
            from pycoral.utils.edgetpu import make_interpreter
            self.model = make_interpreter(self.config.model_path)
            self.model.allocate_tensors()
            self._tflite_input_details = self.model.get_input_details()
            self._tflite_output_details = self.model.get_output_details()
            logger.info("Loaded EdgeTPU model")
        except ImportError:
            logger.error("pycoral not installed. See: https://coral.ai/docs/accelerator/get-started/")
            raise

    def _load_hailo(self):
        """Load Hailo HEF model."""
        try:
            from hailo_platform import HailoRTClient
            self.model = HailoRTClient()
            self.model.load_hef(self.config.model_path)
            logger.info("Loaded Hailo HEF model")
        except ImportError:
            logger.error("hailo_platform not installed. See: https://hailo.ai/developer-zone/")
            raise

    def _load_opencv_dnn(self):
        """
        Load YOLOv4-tiny model using OpenCV DNN.

        This is the lightweight, production-proven approach that works well on Raspberry Pi.
        NO PyTorch or Ultralytics required - just OpenCV!

        Expected files:
        - .cfg file (network configuration)
        - .weights file (pre-trained weights)
        - .names file (class names)
        """
        import cv2
        import os

        # Determine file paths
        config_path = self.config.opencv_config_path or self.config.model_path.replace('.weights', '.cfg')
        weights_path = self.config.opencv_weights_path or self.config.model_path
        names_path = self.config.opencv_names_path

        # Auto-detect names file if not specified
        if not names_path:
            base_dir = os.path.dirname(weights_path) or '.'
            names_path = os.path.join(base_dir, 'coco.names')

        # Validate files exist
        for path, name in [(config_path, 'config'), (weights_path, 'weights')]:
            if not os.path.exists(path):
                logger.error(f"OpenCV DNN {name} file not found: {path}")
                logger.error("Download YOLOv4-tiny files:")
                logger.error("  wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg")
                logger.error("  wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights")
                logger.error("  wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
                raise FileNotFoundError(f"OpenCV DNN {name} file not found: {path}")

        logger.info(f"Loading OpenCV DNN model...")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Weights: {weights_path}")
        logger.info(f"  Names: {names_path}")

        # Load network using OpenCV DNN
        self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

        # Configure for CPU (optimal for Raspberry Pi)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get output layer names
        layer_names = self.model.getLayerNames()
        try:
            # OpenCV 4.x format
            self._opencv_output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
        except (TypeError, IndexError):
            # OpenCV 3.x format
            self._opencv_output_layers = [layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

        # Load class names (use file if exists, otherwise use COCO_CLASSES)
        if os.path.exists(names_path):
            with open(names_path, 'r') as f:
                self._opencv_classes = [line.strip() for line in f.readlines()]
            logger.info(f"  Loaded {len(self._opencv_classes)} classes from {names_path}")
        else:
            self._opencv_classes = self.COCO_CLASSES
            logger.info(f"  Using built-in COCO classes ({len(self._opencv_classes)} classes)")

        logger.info(f"  Output layers: {self._opencv_output_layers}")
        logger.info(f"  Input size: {self.config.input_size[0]}x{self.config.input_size[1]}")
        logger.info("OpenCV DNN model loaded successfully!")

    def _infer_opencv_dnn(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference using OpenCV DNN backend.

        This is optimized for Raspberry Pi performance:
        - Uses smaller input size (320x320 recommended)
        - Efficient blob creation
        - Built-in NMS from OpenCV
        """
        import cv2

        height, width = frame.shape[:2]
        input_size = self.config.input_size[0]  # Assume square input

        # Create blob from image (this is the preprocessing step)
        blob = cv2.dnn.blobFromImage(
            frame,
            1/255.0,                    # Scale factor (normalize to 0-1)
            (input_size, input_size),   # Resize to input size
            swapRB=True,                # Convert BGR -> RGB
            crop=False                  # Don't crop, resize with padding
        )

        # Set input and run forward pass
        self.model.setInput(blob)
        outputs = self.model.forward(self._opencv_output_layers)

        # Parse detections
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]  # Class probabilities start at index 5
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])

                if confidence > self.config.confidence_threshold:
                    # Check class filter if specified
                    if self.config.filter_classes and class_id not in self.config.filter_classes:
                        continue

                    # Scale bounding box back to original image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression using OpenCV's built-in function
        detections = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                self.config.confidence_threshold,
                self.config.iou_threshold
            )

            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]

                    # Ensure coordinates are within image bounds
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(width, x + w)
                    y2 = min(height, y + h)

                    cls_id = class_ids[i]
                    cls_name = self._opencv_classes[cls_id] if cls_id < len(self._opencv_classes) else f"class_{cls_id}"

                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=confidences[i],
                        bbox=(x1, y1, x2, y2),
                        center=((x1 + x2) // 2, (y1 + y2) // 2),
                        area=(x2 - x1) * (y2 - y1)
                    ))

        return detections[:self.config.max_detections]

    def process(self, frame: np.ndarray, frame_id: int = 0) -> Optional[DetectionResult]:
        """
        Process a frame for object detection.

        Uses frame skipping to maintain streaming performance.

        Args:
            frame: Input frame (BGR or RGB numpy array)
            frame_id: Frame identifier

        Returns:
            DetectionResult or None if frame was skipped
        """
        self.frame_counter += 1

        # Check if we should skip this frame
        if not self._should_process():
            self._stats['frames_skipped'] += 1
            return self.last_result  # Return cached result

        # Process frame
        start_time = time.perf_counter()

        try:
            detections = self._run_inference(frame)
            inference_time_ms = (time.perf_counter() - start_time) * 1000

            result = DetectionResult(
                detections=detections,
                frame_id=frame_id,
                timestamp_ns=time.time_ns(),
                inference_time_ms=inference_time_ms,
                model_name=self.config.model_path,
                input_size=self.config.input_size
            )

            # Update stats
            self._update_stats(inference_time_ms, len(detections))

            # Adapt skip rate if using adaptive strategy
            if self.config.skip_strategy == FrameSkipStrategy.ADAPTIVE:
                self._adapt_skip_rate(inference_time_ms)

            self.last_result = result
            return result

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self.last_result

    def _should_process(self) -> bool:
        """Determine if current frame should be processed."""
        if self.config.skip_strategy == FrameSkipStrategy.NONE:
            return True
        elif self.config.skip_strategy == FrameSkipStrategy.FIXED:
            return self.frame_counter % self._current_skip == 0
        elif self.config.skip_strategy == FrameSkipStrategy.ADAPTIVE:
            return self.frame_counter % self._current_skip == 0
        elif self.config.skip_strategy == FrameSkipStrategy.QUEUE:
            return True  # Queue strategy handles differently
        return True

    def _adapt_skip_rate(self, inference_time_ms: float):
        """Adapt skip rate based on inference time."""
        if inference_time_ms > self.config.target_inference_time_ms * 1.5:
            # Too slow, skip more frames
            self._current_skip = min(self._current_skip + 1, self.config.max_skip)
        elif inference_time_ms < self.config.target_inference_time_ms * 0.5:
            # Fast enough, process more frames
            self._current_skip = max(self._current_skip - 1, self.config.min_skip)

    def _run_inference(self, frame: np.ndarray) -> List[Detection]:
        """Run inference based on backend."""
        if self.config.backend == YOLOBackend.PYTORCH:
            return self._infer_pytorch(frame)
        elif self.config.backend == YOLOBackend.ONNX:
            return self._infer_onnx(frame)
        elif self.config.backend == YOLOBackend.NCNN:
            return self._infer_ncnn(frame)
        elif self.config.backend in (YOLOBackend.TFLITE, YOLOBackend.EDGETPU):
            return self._infer_tflite(frame)
        elif self.config.backend == YOLOBackend.HAILO:
            return self._infer_hailo(frame)
        elif self.config.backend == YOLOBackend.OPENCV_DNN:
            return self._infer_opencv_dnn(frame)
        return []

    def _infer_pytorch(self, frame: np.ndarray) -> List[Detection]:
        """Run PyTorch/Ultralytics inference."""
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.COCO_CLASSES[cls_id] if cls_id < len(self.COCO_CLASSES) else f"class_{cls_id}"

            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                area=(x2 - x1) * (y2 - y1)
            ))

        return detections

    def _infer_onnx(self, frame: np.ndarray) -> List[Detection]:
        """Run ONNX inference."""
        import cv2

        # Preprocess
        input_size = self.config.input_size
        img = cv2.resize(frame, input_size)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        # Run inference
        outputs = self.model.run(None, {self._onnx_input_name: img})

        # Postprocess (simplified - actual implementation depends on model output format)
        return self._postprocess_yolo_output(outputs[0], frame.shape[:2])

    def _infer_ncnn(self, frame: np.ndarray) -> List[Detection]:
        """Run NCNN inference."""
        import ncnn
        import cv2

        # Preprocess
        input_size = self.config.input_size
        img = cv2.resize(frame, input_size)

        mat = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_BGR, input_size[0], input_size[1])

        # Normalize
        mean_vals = [0, 0, 0]
        norm_vals = [1/255.0, 1/255.0, 1/255.0]
        mat.substract_mean_normalize(mean_vals, norm_vals)

        # Run inference
        ex = self.model.create_extractor()
        ex.input("in0", mat)
        ret, output = ex.extract("out0")

        # Postprocess
        return self._postprocess_yolo_output(np.array(output), frame.shape[:2])

    def _infer_tflite(self, frame: np.ndarray) -> List[Detection]:
        """Run TFLite/EdgeTPU inference."""
        import cv2

        # Preprocess
        input_size = self.config.input_size
        img = cv2.resize(frame, input_size)
        img = np.expand_dims(img, 0).astype(np.uint8)

        # Set input
        self.model.set_tensor(self._tflite_input_details[0]['index'], img)

        # Run inference
        self.model.invoke()

        # Get output
        output = self.model.get_tensor(self._tflite_output_details[0]['index'])

        # Postprocess
        return self._postprocess_yolo_output(output, frame.shape[:2])

    def _infer_hailo(self, frame: np.ndarray) -> List[Detection]:
        """Run Hailo inference."""
        import cv2

        # Preprocess
        input_size = self.config.input_size
        img = cv2.resize(frame, input_size)

        # Run inference
        outputs = self.model.infer(img)

        # Postprocess
        return self._postprocess_yolo_output(outputs[0], frame.shape[:2])

    def _postprocess_yolo_output(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> List[Detection]:
        """
        Postprocess YOLO output to extract detections.

        This is a generic implementation - actual format depends on model.
        """
        detections = []
        h_orig, w_orig = original_size
        h_input, w_input = self.config.input_size

        # Scale factors
        scale_x = w_orig / w_input
        scale_y = h_orig / h_input

        # Output format: [batch, num_detections, 85] for YOLOv8
        # 85 = 4 (bbox) + 1 (objectness) + 80 (class probs)

        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension

        for detection in output:
            if len(detection) >= 85:
                x, y, w, h = detection[:4]
                objectness = detection[4]
                class_probs = detection[5:85]

                if objectness < self.config.confidence_threshold:
                    continue

                cls_id = np.argmax(class_probs)
                confidence = objectness * class_probs[cls_id]

                if confidence < self.config.confidence_threshold:
                    continue

                # Convert to xyxy
                x1 = int((x - w/2) * scale_x)
                y1 = int((y - h/2) * scale_y)
                x2 = int((x + w/2) * scale_x)
                y2 = int((y + h/2) * scale_y)

                # Clamp to image bounds
                x1 = max(0, min(x1, w_orig))
                y1 = max(0, min(y1, h_orig))
                x2 = max(0, min(x2, w_orig))
                y2 = max(0, min(y2, h_orig))

                cls_name = self.COCO_CLASSES[cls_id] if cls_id < len(self.COCO_CLASSES) else f"class_{cls_id}"

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    area=(x2 - x1) * (y2 - y1)
                ))

        # Apply NMS
        detections = self._non_max_suppression(detections)

        return detections[:self.config.max_detections]

    def _non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if self._iou(best.bbox, d.bbox) < self.config.iou_threshold
            ]

        return keep

    def _iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _update_stats(self, inference_time_ms: float, detection_count: int):
        """Update processing statistics."""
        self._stats['frames_processed'] += 1
        self._stats['total_inference_time_ms'] += inference_time_ms
        self._stats['inference_times'].append(inference_time_ms)

        if self._stats['frames_processed'] > 0:
            self._stats['avg_inference_time_ms'] = (
                self._stats['total_inference_time_ms'] / self._stats['frames_processed']
            )

        # Running average of detections
        alpha = 0.1
        self._stats['detections_per_frame'] = (
            alpha * detection_count + (1 - alpha) * self._stats['detections_per_frame']
        )

    def get_stats(self) -> dict:
        """Get processing statistics."""
        stats = self._stats.copy()
        if stats['inference_times']:
            times = list(stats['inference_times'])
            stats['p50_inference_ms'] = np.percentile(times, 50)
            stats['p95_inference_ms'] = np.percentile(times, 95)
            stats['p99_inference_ms'] = np.percentile(times, 99)
        stats['current_skip'] = self._current_skip
        return stats


class AsyncYOLOProcessor:
    """
    Multiprocessing-based async YOLO processor.

    Runs YOLO in a separate process to never block the main pipeline.
    """

    def __init__(self, config: Optional[YOLOConfig] = None):
        self.config = config or YOLOConfig()
        self._process: Optional[mp.Process] = None
        self._input_queue: Optional[mp.Queue] = None
        self._output_queue: Optional[mp.Queue] = None
        self._running = mp.Value('b', False)
        self.last_result: Optional[DetectionResult] = None

    def start(self):
        """Start the async processor."""
        self._input_queue = mp.Queue(maxsize=2)
        self._output_queue = mp.Queue(maxsize=2)
        self._running.value = True

        self._process = mp.Process(
            target=self._worker_loop,
            args=(self.config, self._input_queue, self._output_queue, self._running),
            daemon=True,
            name="YOLOProcessor"
        )
        self._process.start()
        logger.info("Async YOLO processor started")

    @staticmethod
    def _worker_loop(config, input_queue, output_queue, running):
        """Worker process main loop."""
        # Set CPU affinity if specified
        if config.cpu_affinity:
            import os
            os.sched_setaffinity(0, set(config.cpu_affinity))

        processor = YOLOProcessor(config)
        if not processor.initialize():
            logger.error("Failed to initialize YOLO in worker")
            return

        while running.value:
            try:
                # Get frame from queue (with timeout)
                frame_data = input_queue.get(timeout=0.1)
                if frame_data is None:
                    continue

                frame, frame_id = frame_data
                result = processor.process(frame, frame_id)

                # Put result in output queue (non-blocking)
                try:
                    output_queue.put_nowait(result)
                except:
                    pass  # Queue full, skip this result

            except Exception:
                continue

    def submit(self, frame: np.ndarray, frame_id: int = 0):
        """Submit a frame for processing (non-blocking)."""
        if self._input_queue is None:
            return

        try:
            # Non-blocking put - if queue full, drop oldest
            if self._input_queue.full():
                try:
                    self._input_queue.get_nowait()
                except:
                    pass
            self._input_queue.put_nowait((frame.copy(), frame_id))
        except:
            pass  # Queue full, skip this frame

    def get_result(self) -> Optional[DetectionResult]:
        """Get latest detection result (non-blocking)."""
        if self._output_queue is None:
            return self.last_result

        try:
            # Get all available results, keep only the latest
            while not self._output_queue.empty():
                self.last_result = self._output_queue.get_nowait()
        except:
            pass

        return self.last_result

    def stop(self):
        """Stop the async processor."""
        self._running.value = False
        if self._process:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
        logger.info("Async YOLO processor stopped")


def draw_detections(
    frame: np.ndarray,
    result: Optional[DetectionResult],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw detection boxes and labels on frame.

    Args:
        frame: Input frame (will be modified in-place)
        result: Detection result
        color: Box color (BGR)
        thickness: Line thickness
        font_scale: Text scale

    Returns:
        Frame with drawn detections
    """
    if result is None or not result.detections:
        return frame

    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for drawing")
        return frame

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label = f"{det.class_name}: {det.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1
        )

        # Label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1
        )

    # Draw inference time
    if result.inference_time_ms > 0:
        info = f"YOLO: {result.inference_time_ms:.1f}ms | {result.count} objects"
        cv2.putText(
            frame,
            info,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== YOLO Processor Test ===")

    # Create test config
    config = YOLOConfig(
        model_path="yolov8n.pt",
        backend=YOLOBackend.PYTORCH,
        skip_strategy=FrameSkipStrategy.FIXED,
        skip_n_frames=3,
        confidence_threshold=0.5
    )

    # Test synchronous processor
    print("\n--- Synchronous Processor ---")
    processor = YOLOProcessor(config)

    if processor.initialize():
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        for i in range(10):
            result = processor.process(test_frame, frame_id=i)
            if result:
                print(f"Frame {i}: {result.count} detections, {result.inference_time_ms:.1f}ms")
            else:
                print(f"Frame {i}: skipped (cached: {processor.last_result.count if processor.last_result else 0} detections)")

        print("\nStats:", processor.get_stats())

    # Test async processor
    print("\n--- Async Processor ---")
    async_processor = AsyncYOLOProcessor(config)
    async_processor.start()

    import time
    for i in range(20):
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        async_processor.submit(test_frame, frame_id=i)

        result = async_processor.get_result()
        if result:
            print(f"Frame {i}: Got result for frame {result.frame_id}, {result.count} detections")
        else:
            print(f"Frame {i}: No result yet")

        time.sleep(0.033)  # ~30 FPS

    async_processor.stop()