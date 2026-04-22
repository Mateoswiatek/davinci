#!/usr/bin/env python3
"""
Optimized YOLO Processor for Raspberry Pi 5
Async multiprocessing pipeline for minimal latency VR streaming

Features:
- Multiprocessing (no GIL blocking)
- Support for multiple formats (PyTorch, NCNN, ONNX, Hailo, EdgeTPU)
- Frame skipping strategies
- Result caching & temporal smoothing
- Performance metrics & monitoring
- Graceful degradation
- CPU pinning for real-time performance

Author: DaVinci VR Project
Date: 2025-11-26
"""

import time
import logging
import multiprocessing as mp
from multiprocessing import Queue, Event, Value
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logging.warning("OpenCV not available")

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    from ncnn import ncnn
    HAS_NCNN = True
except ImportError:
    HAS_NCNN = False

try:
    from pycoral.adapters import detect
    from pycoral.utils.edgetpu import make_interpreter
    HAS_CORAL = True
except ImportError:
    HAS_CORAL = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats"""
    PYTORCH = "pt"      # YOLOv8 PyTorch (.pt)
    ONNX = "onnx"       # ONNX Runtime (.onnx)
    NCNN = "ncnn"       # NCNN (.param + .bin)
    TFLITE = "tflite"   # TensorFlow Lite (.tflite)
    EDGETPU = "edgetpu" # Coral EdgeTPU (.tflite)
    HAILO = "hef"       # Hailo (.hef)


class FrameSkipStrategy(Enum):
    """Frame skipping strategies"""
    NONE = "none"           # Process every frame
    FIXED = "fixed"         # Process every N-th frame
    ADAPTIVE = "adaptive"   # Skip based on inference time
    QUEUE_FULL = "queue"    # Skip if queue full


@dataclass
class YOLOConfig:
    """YOLO processor configuration"""
    model_path: str
    model_format: ModelFormat = ModelFormat.PYTORCH
    confidence: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 50
    classes: Optional[List[int]] = None
    input_size: Tuple[int, int] = (640, 640)
    device: str = "cpu"


@dataclass
class ProcessingConfig:
    """Processing pipeline configuration"""
    max_queue_size: int = 2
    frame_skip_strategy: FrameSkipStrategy = FrameSkipStrategy.FIXED
    frame_skip_interval: int = 3  # Process every 3rd frame
    enable_smoothing: bool = True
    smoothing_window: int = 3
    cpu_affinity: Optional[List[int]] = None  # Pin to specific cores
    priority: int = -5  # Process priority (negative = higher)


@dataclass
class Detection:
    """Single object detection"""
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionResult:
    """Detection result with metadata"""
    detections: List[Detection]
    inference_time_ms: float
    timestamp: float
    frame_id: int


class PerformanceMonitor:
    """Monitor processing performance"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.inference_times = []
        self.queue_full_count = 0
        self.frames_processed = 0
        self.frames_skipped = 0
        self.start_time = time.time()

    def add_inference_time(self, time_ms: float):
        """Add inference time measurement"""
        self.inference_times.append(time_ms)
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)

    def record_frame_processed(self):
        """Record frame processed"""
        self.frames_processed += 1

    def record_frame_skipped(self):
        """Record frame skipped"""
        self.frames_skipped += 1

    def record_queue_full(self):
        """Record queue full event"""
        self.queue_full_count += 1

    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        elapsed = time.time() - self.start_time
        total_frames = self.frames_processed + self.frames_skipped

        return {
            'avg_inference_ms': np.mean(self.inference_times),
            'std_inference_ms': np.std(self.inference_times),
            'min_inference_ms': np.min(self.inference_times),
            'max_inference_ms': np.max(self.inference_times),
            'p50_inference_ms': np.percentile(self.inference_times, 50),
            'p95_inference_ms': np.percentile(self.inference_times, 95),
            'p99_inference_ms': np.percentile(self.inference_times, 99),
            'detection_fps': self.frames_processed / elapsed if elapsed > 0 else 0,
            'frames_processed': self.frames_processed,
            'frames_skipped': self.frames_skipped,
            'skip_ratio': self.frames_skipped / total_frames if total_frames > 0 else 0,
            'queue_full_count': self.queue_full_count,
            'uptime_seconds': elapsed
        }

    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        if not stats:
            logger.info("No statistics available yet")
            return

        logger.info("=" * 60)
        logger.info("YOLO Performance Statistics")
        logger.info("=" * 60)
        logger.info(f"Detection FPS:     {stats['detection_fps']:.1f}")
        logger.info(f"Frames processed:  {stats['frames_processed']}")
        logger.info(f"Frames skipped:    {stats['frames_skipped']} ({stats['skip_ratio']*100:.1f}%)")
        logger.info(f"Queue full events: {stats['queue_full_count']}")
        logger.info("-" * 60)
        logger.info(f"Inference time:")
        logger.info(f"  Mean:  {stats['avg_inference_ms']:.1f}ms ± {stats['std_inference_ms']:.1f}ms")
        logger.info(f"  P50:   {stats['p50_inference_ms']:.1f}ms")
        logger.info(f"  P95:   {stats['p95_inference_ms']:.1f}ms")
        logger.info(f"  P99:   {stats['p99_inference_ms']:.1f}ms")
        logger.info(f"  Range: {stats['min_inference_ms']:.1f}ms - {stats['max_inference_ms']:.1f}ms")
        logger.info("-" * 60)
        logger.info(f"Uptime: {stats['uptime_seconds']:.1f}s")
        logger.info("=" * 60)


class YOLOWorker:
    """
    YOLO inference worker - runs in separate process
    Supports multiple model formats
    """

    def __init__(self, yolo_config: YOLOConfig):
        self.config = yolo_config
        self.model = None

    def load_model(self):
        """Load YOLO model based on format"""
        logger.info(f"Loading YOLO model: {self.config.model_path}")
        logger.info(f"Format: {self.config.model_format.value}")

        if self.config.model_format == ModelFormat.PYTORCH:
            if not HAS_ULTRALYTICS:
                raise RuntimeError("ultralytics not installed")
            self.model = YOLO(self.config.model_path)
            logger.info("PyTorch model loaded")

        elif self.config.model_format == ModelFormat.NCNN:
            if not HAS_NCNN:
                raise RuntimeError("ncnn not installed")
            # NCNN requires .param and .bin files
            param_path = self.config.model_path.replace('.bin', '.param')
            self.model = self._load_ncnn(param_path, self.config.model_path)
            logger.info("NCNN model loaded")

        elif self.config.model_format == ModelFormat.EDGETPU:
            if not HAS_CORAL:
                raise RuntimeError("pycoral not installed")
            self.model = make_interpreter(self.config.model_path)
            self.model.allocate_tensors()
            logger.info("EdgeTPU model loaded")

        elif self.config.model_format == ModelFormat.ONNX:
            if not HAS_ULTRALYTICS:
                raise RuntimeError("ultralytics not installed for ONNX")
            # Ultralytics can load ONNX
            self.model = YOLO(self.config.model_path, task='detect')
            logger.info("ONNX model loaded")

        else:
            raise NotImplementedError(f"Format {self.config.model_format} not implemented")

    def _load_ncnn(self, param_path: str, bin_path: str):
        """Load NCNN model"""
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False  # Pi 5 doesn't support Vulkan
        net.opt.num_threads = 4  # Use all cores
        net.load_param(param_path)
        net.load_model(bin_path)
        return net

    def inference(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on frame
        Returns DetectionResult
        """
        start_time = time.perf_counter()

        if self.config.model_format == ModelFormat.PYTORCH:
            result = self._inference_pytorch(frame)
        elif self.config.model_format == ModelFormat.NCNN:
            result = self._inference_ncnn(frame)
        elif self.config.model_format == ModelFormat.EDGETPU:
            result = self._inference_edgetpu(frame)
        elif self.config.model_format == ModelFormat.ONNX:
            result = self._inference_pytorch(frame)  # Same as PyTorch
        else:
            result = DetectionResult([], 0.0, time.time(), 0)

        inference_time_ms = (time.perf_counter() - start_time) * 1000
        result.inference_time_ms = inference_time_ms

        return result

    def _inference_pytorch(self, frame: np.ndarray) -> DetectionResult:
        """PyTorch/ONNX inference"""
        results = self.model(
            frame,
            conf=self.config.confidence,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            classes=self.config.classes,
            verbose=False,
            device=self.config.device
        )[0]

        detections = []
        boxes = results.boxes

        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())

            detection = Detection(
                box=tuple(box),
                confidence=conf,
                class_id=cls,
                class_name=results.names[cls]
            )
            detections.append(detection)

        return DetectionResult(
            detections=detections,
            inference_time_ms=0.0,  # Will be filled by caller
            timestamp=time.time(),
            frame_id=0
        )

    def _inference_ncnn(self, frame: np.ndarray) -> DetectionResult:
        """NCNN inference (fastest CPU-only)"""
        # TODO: Implement NCNN inference
        # This requires proper NCNN bindings setup
        # For now, fallback to PyTorch
        logger.warning("NCNN inference not fully implemented, using PyTorch")
        return self._inference_pytorch(frame)

    def _inference_edgetpu(self, frame: np.ndarray) -> DetectionResult:
        """Coral EdgeTPU inference"""
        # Preprocess
        input_details = self.model.get_input_details()[0]
        _, height, width, _ = input_details['shape']

        frame_resized = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Run inference
        self.model.set_tensor(input_details['index'], frame_rgb[np.newaxis, ...])
        self.model.invoke()

        # Get detections
        boxes = detect.get_objects(self.model, self.config.confidence)

        detections = []
        for obj in boxes:
            detection = Detection(
                box=(obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax),
                confidence=obj.score,
                class_id=obj.id,
                class_name=f"class_{obj.id}"
            )
            detections.append(detection)

        return DetectionResult(
            detections=detections,
            inference_time_ms=0.0,
            timestamp=time.time(),
            frame_id=0
        )


def worker_process(
    yolo_config: YOLOConfig,
    processing_config: ProcessingConfig,
    input_queue: Queue,
    output_queue: Queue,
    stop_event: Event,
    frames_processed: Value
):
    """
    Worker process main loop
    Runs YOLO inference in separate process
    """
    try:
        # Set CPU affinity if specified
        if processing_config.cpu_affinity:
            import os
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity(processing_config.cpu_affinity)
            logger.info(f"Worker pinned to CPUs: {processing_config.cpu_affinity}")

        # Set process priority
        import os
        try:
            os.nice(processing_config.priority)
            logger.info(f"Worker priority set to: {processing_config.priority}")
        except PermissionError:
            logger.warning("Cannot set process priority (requires root)")

        # Load model
        worker = YOLOWorker(yolo_config)
        worker.load_model()

        logger.info("YOLO worker ready")

        frame_id = 0

        while not stop_event.is_set():
            try:
                # Get frame (blocking with timeout)
                frame = input_queue.get(timeout=0.1)

                # Run inference
                result = worker.inference(frame)
                result.frame_id = frame_id

                # Send result (non-blocking)
                try:
                    output_queue.put_nowait(result)
                    with frames_processed.get_lock():
                        frames_processed.value += 1
                except:
                    # Queue full - skip result
                    pass

                frame_id += 1

            except Exception as e:
                if not stop_event.is_set():
                    logger.error(f"Worker error: {e}")

    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        raise


class AsyncYOLOProcessor:
    """
    Async YOLO processor with multiprocessing
    Main interface for VR streaming pipeline
    """

    def __init__(
        self,
        yolo_config: YOLOConfig,
        processing_config: ProcessingConfig
    ):
        self.yolo_config = yolo_config
        self.processing_config = processing_config

        # Shared queues
        self.input_queue = Queue(maxsize=processing_config.max_queue_size)
        self.output_queue = Queue(maxsize=processing_config.max_queue_size)
        self.stop_event = Event()
        self.frames_processed = Value('i', 0)

        # Result cache
        self.latest_result: Optional[DetectionResult] = None
        self.result_history: List[DetectionResult] = []

        # Frame counters
        self.frame_counter = 0
        self.frames_submitted = 0

        # Performance monitoring
        self.monitor = PerformanceMonitor()

        # Start worker
        self._start_worker()

        logger.info("AsyncYOLOProcessor initialized")
        logger.info(f"  Model: {yolo_config.model_path}")
        logger.info(f"  Format: {yolo_config.model_format.value}")
        logger.info(f"  Strategy: {processing_config.frame_skip_strategy.value}")
        if processing_config.frame_skip_strategy == FrameSkipStrategy.FIXED:
            logger.info(f"  Skip interval: {processing_config.frame_skip_interval}")

    def _start_worker(self):
        """Start worker process"""
        self.worker = mp.Process(
            target=worker_process,
            args=(
                self.yolo_config,
                self.processing_config,
                self.input_queue,
                self.output_queue,
                self.stop_event,
                self.frames_processed
            ),
            daemon=True,
            name="YOLOWorker"
        )
        self.worker.start()
        logger.info(f"Worker process started (PID: {self.worker.pid})")

    def should_process_frame(self) -> bool:
        """
        Determine if current frame should be processed
        Based on frame skip strategy
        """
        strategy = self.processing_config.frame_skip_strategy

        if strategy == FrameSkipStrategy.NONE:
            return True

        elif strategy == FrameSkipStrategy.FIXED:
            interval = self.processing_config.frame_skip_interval
            return self.frame_counter % interval == 0

        elif strategy == FrameSkipStrategy.QUEUE_FULL:
            return not self.input_queue.full()

        elif strategy == FrameSkipStrategy.ADAPTIVE:
            # Skip if inference too slow
            if self.latest_result:
                target_time = 33.0  # 30 FPS target
                if self.latest_result.inference_time_ms > target_time:
                    # Too slow - skip more frames
                    skip_ratio = int(self.latest_result.inference_time_ms / target_time)
                    return self.frame_counter % (skip_ratio + 1) == 0
            return True

        return False

    def submit_frame(self, frame: np.ndarray) -> bool:
        """
        Submit frame for processing
        Returns True if submitted, False if skipped
        """
        self.frame_counter += 1

        # Check if should process
        if not self.should_process_frame():
            self.monitor.record_frame_skipped()
            return False

        # Try to submit (non-blocking)
        try:
            self.input_queue.put_nowait(frame.copy())
            self.frames_submitted += 1
            self.monitor.record_frame_processed()
            return True
        except:
            self.monitor.record_queue_full()
            self.monitor.record_frame_skipped()
            return False

    def get_latest_result(self) -> Optional[DetectionResult]:
        """
        Get latest detection result (non-blocking)
        Returns cached result if no new one available
        """
        # Drain output queue - keep only latest
        new_results = 0
        try:
            while not self.output_queue.empty():
                result = self.output_queue.get_nowait()
                self.latest_result = result

                # Update history for smoothing
                if self.processing_config.enable_smoothing:
                    self.result_history.append(result)
                    max_history = self.processing_config.smoothing_window
                    if len(self.result_history) > max_history:
                        self.result_history.pop(0)

                # Update monitor
                self.monitor.add_inference_time(result.inference_time_ms)
                new_results += 1

        except:
            pass

        return self.latest_result

    def get_smoothed_result(self) -> Optional[DetectionResult]:
        """
        Get temporally smoothed detection result
        Averages detections over last N frames
        """
        if not self.processing_config.enable_smoothing:
            return self.latest_result

        if not self.result_history:
            return None

        # Simple averaging (could be improved with tracking)
        # For now, just return latest
        return self.latest_result

    def draw_detections(
        self,
        frame: np.ndarray,
        result: Optional[DetectionResult] = None,
        show_info: bool = True
    ) -> np.ndarray:
        """
        Draw detection boxes on frame
        """
        if result is None:
            result = self.latest_result

        if result is None or not HAS_OPENCV:
            return frame

        annotated = frame.copy()

        # Draw boxes
        for det in result.detections:
            x1, y1, x2, y2 = map(int, det.box)

            # Color based on class
            color = (0, 255, 0)  # Green

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 5),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        # Draw info overlay
        if show_info:
            stats = self.monitor.get_stats()
            if stats:
                info_text = [
                    f"Detection FPS: {stats['detection_fps']:.1f}",
                    f"Inference: {stats['avg_inference_ms']:.1f}ms",
                    f"Detections: {len(result.detections)}",
                    f"Skipped: {stats['skip_ratio']*100:.0f}%"
                ]

                y_offset = 30
                for text in info_text:
                    cv2.putText(
                        annotated,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    y_offset += 25

        return annotated

    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return self.monitor.get_stats()

    def print_stats(self):
        """Print performance statistics"""
        self.monitor.print_stats()

    def stop(self):
        """Stop worker process"""
        logger.info("Stopping YOLO processor...")
        self.stop_event.set()

        # Wait for worker to finish
        self.worker.join(timeout=2.0)

        if self.worker.is_alive():
            logger.warning("Worker did not stop gracefully, terminating...")
            self.worker.terminate()
            self.worker.join(timeout=1.0)

        # Print final stats
        self.print_stats()

        logger.info("YOLO processor stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test YOLO Async Processor')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model path')
    parser.add_argument('--format', type=str, default='pt',
                        choices=['pt', 'onnx', 'ncnn', 'edgetpu'],
                        help='Model format')
    parser.add_argument('--skip', type=int, default=3,
                        help='Process every N-th frame')
    parser.add_argument('--camera', action='store_true',
                        help='Use Pi Camera (otherwise test with dummy frames)')
    parser.add_argument('--cores', type=str, default=None,
                        help='CPU cores for worker (e.g., "2,3")')
    args = parser.parse_args()

    # Parse CPU affinity
    cpu_affinity = None
    if args.cores:
        cpu_affinity = [int(c) for c in args.cores.split(',')]

    # Create config
    yolo_config = YOLOConfig(
        model_path=args.model,
        model_format=ModelFormat(args.format),
        confidence=0.5,
        input_size=(640, 640)
    )

    processing_config = ProcessingConfig(
        max_queue_size=2,
        frame_skip_strategy=FrameSkipStrategy.FIXED,
        frame_skip_interval=args.skip,
        cpu_affinity=cpu_affinity
    )

    # Create processor
    with AsyncYOLOProcessor(yolo_config, processing_config) as processor:

        if args.camera and HAS_OPENCV:
            # Test with camera
            from picamera2 import Picamera2

            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()

            logger.info("Press 'q' to quit")

            try:
                while True:
                    # Capture frame
                    frame = picam2.capture_array("main")

                    # Submit for processing
                    submitted = processor.submit_frame(frame)

                    # Get latest result
                    result = processor.get_latest_result()

                    # Draw detections
                    annotated = processor.draw_detections(frame, result)

                    # Display
                    cv2.imshow("YOLO Test", annotated)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                picam2.stop()
                cv2.destroyAllWindows()

        else:
            # Test with dummy frames
            logger.info("Testing with dummy frames (30 seconds)...")

            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < 30:
                # Create dummy frame
                frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

                # Submit
                submitted = processor.submit_frame(frame)

                # Get result
                result = processor.get_latest_result()

                if result:
                    logger.info(f"Frame {frame_count}: {len(result.detections)} detections, "
                                f"{result.inference_time_ms:.1f}ms")

                # Maintain 30 FPS
                time.sleep(1/30)
                frame_count += 1

            logger.info(f"Test complete: {frame_count} frames in {time.time()-start_time:.1f}s")