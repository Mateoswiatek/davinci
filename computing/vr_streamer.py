#!/usr/bin/env python3
"""
VR Streamer - Main Pipeline

High-performance VR streaming pipeline combining:
- Camera capture (Picamera2)
- YOLO object detection (async, optional)
- Multiple streaming protocols (UDP, WebSocket)

Designed for Raspberry Pi 5 → Oculus Quest streaming.

Author: Claude Code
License: MIT

Usage:
    python vr_streamer.py --help
    python vr_streamer.py --protocol websocket --port 8000
    python vr_streamer.py --protocol udp --target 192.168.1.100 --port 5000
    python vr_streamer.py --protocol both --yolo --yolo-skip 3
"""

import asyncio
import argparse
import logging
import signal
import time
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import multiprocessing as mp

import numpy as np

# Local imports
from camera_capture import CameraCapture, CameraConfig, CameraProfile, CapturedFrame
from yolo_processor import (
    YOLOProcessor, AsyncYOLOProcessor, YOLOConfig,
    YOLOBackend, FrameSkipStrategy, DetectionResult, draw_detections
)
from protocols import (
    StreamConfig, UDPStreamer, WebSocketStreamer,
    MultiProtocolStreamer
)
from protocols.udp_streamer import UDPConfig
from protocols.websocket_streamer import WebSocketConfig
from protocols.base import CompressionFormat

logger = logging.getLogger(__name__)


class StreamProtocol(Enum):
    """Available streaming protocols."""
    UDP = "udp"
    WEBSOCKET = "websocket"
    BOTH = "both"


@dataclass
class VRStreamerConfig:
    """Main VR streamer configuration."""
    # Camera
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    camera_stereo: bool = False
    camera_profile: CameraProfile = CameraProfile.LOW_LATENCY

    # Streaming
    protocol: StreamProtocol = StreamProtocol.WEBSOCKET
    host: str = "0.0.0.0"
    port: int = 8000
    target_host: str = ""  # For UDP
    target_port: int = 5000  # For UDP
    jpeg_quality: int = 85

    # YOLO
    yolo_enabled: bool = False
    yolo_model: str = "yolov8n.pt"
    yolo_backend: YOLOBackend = YOLOBackend.OPENCV_DNN  # Default to OpenCV DNN (lightweight!)
    yolo_skip_frames: int = 10  # Default to 10 for better performance
    yolo_confidence: float = 0.3  # Lower default for better detection
    yolo_async: bool = True
    yolo_draw: bool = True
    yolo_input_size: int = 320  # Smaller = faster (320 for speed, 416 for accuracy)
    yolo_filter_classes: Optional[List[int]] = None  # Filter specific classes (e.g., [0] for persons only)

    # OpenCV DNN specific paths (for OPENCV_DNN backend)
    yolo_opencv_config: Optional[str] = None  # Path to .cfg file
    yolo_opencv_weights: Optional[str] = None  # Path to .weights file
    yolo_opencv_names: Optional[str] = None  # Path to .names file

    # Performance
    cpu_affinity_camera: Optional[List[int]] = None
    cpu_affinity_yolo: Optional[List[int]] = None
    cpu_affinity_network: Optional[List[int]] = None

    # Debug
    show_stats: bool = True
    stats_interval: float = 5.0
    log_level: str = "INFO"


class VRStreamer:
    """
    Main VR streaming pipeline.

    Architecture:
    - Main process: Camera capture + streaming
    - Subprocess (optional): YOLO detection

    This ensures streaming is never blocked by YOLO.
    """

    def __init__(self, config: VRStreamerConfig):
        self.config = config
        self._running = False

        # Components
        self.camera: Optional[CameraCapture] = None
        self.yolo: Optional[AsyncYOLOProcessor] = None
        self.streamer: Optional[MultiProtocolStreamer] = None

        # State
        self._frame_count = 0
        self._last_detection: Optional[DetectionResult] = None
        self._stats_task: Optional[asyncio.Task] = None

        # Stats
        self._stats = {
            'frames_captured': 0,
            'frames_streamed': 0,
            'frames_processed_yolo': 0,
            'avg_capture_ms': 0.0,
            'avg_stream_ms': 0.0,
            'avg_total_ms': 0.0,
            'clients': 0,
            'detections': 0
        }

    def _setup_logging(self):
        """Configure logging."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )

    def _setup_cpu_affinity(self):
        """Set CPU affinity for main process."""
        if self.config.cpu_affinity_camera:
            try:
                os.sched_setaffinity(0, set(self.config.cpu_affinity_camera))
                logger.info(f"Main process CPU affinity: {self.config.cpu_affinity_camera}")
            except Exception as e:
                logger.warning(f"Failed to set CPU affinity: {e}")

    async def initialize(self) -> bool:
        """Initialize all components."""
        self._setup_logging()
        self._setup_cpu_affinity()

        logger.info("Initializing VR Streamer...")

        # 1. Initialize camera
        camera_config = CameraConfig(
            width=self.config.camera_width,
            height=self.config.camera_height,
            fps=self.config.camera_fps,
            stereo=self.config.camera_stereo,
            profile=self.config.camera_profile
        )

        self.camera = CameraCapture(camera_config)
        if not self.camera.initialize():
            logger.error("Failed to initialize camera")
            return False

        logger.info(f"Camera initialized: {self.config.camera_width}x{self.config.camera_height} @ {self.config.camera_fps}fps")

        # 2. Initialize YOLO (if enabled)
        if self.config.yolo_enabled:
            yolo_config = YOLOConfig(
                model_path=self.config.yolo_model,
                backend=self.config.yolo_backend,
                input_size=(self.config.yolo_input_size, self.config.yolo_input_size),
                confidence_threshold=self.config.yolo_confidence,
                skip_strategy=FrameSkipStrategy.FIXED,
                skip_n_frames=self.config.yolo_skip_frames,
                cpu_affinity=self.config.cpu_affinity_yolo,
                filter_classes=self.config.yolo_filter_classes,
                # OpenCV DNN specific paths
                opencv_config_path=self.config.yolo_opencv_config,
                opencv_weights_path=self.config.yolo_opencv_weights,
                opencv_names_path=self.config.yolo_opencv_names,
            )

            logger.info(f"Initializing YOLO with backend: {self.config.yolo_backend.value}")

            if self.config.yolo_async:
                self.yolo = AsyncYOLOProcessor(yolo_config)
                self.yolo.start()
                logger.info("Async YOLO processor started")
            else:
                self.yolo = YOLOProcessor(yolo_config)
                if not self.yolo.initialize():
                    logger.warning("YOLO initialization failed - continuing without detection")
                    self.yolo = None
                else:
                    logger.info("Sync YOLO processor initialized")

        # 3. Initialize streaming protocols
        self.streamer = MultiProtocolStreamer()

        if self.config.protocol in (StreamProtocol.WEBSOCKET, StreamProtocol.BOTH):
            ws_config = WebSocketConfig(
                host=self.config.host,
                port=self.config.port,
                compression=CompressionFormat.JPEG,
                jpeg_quality=self.config.jpeg_quality,
                include_detections=True
            )
            ws_streamer = WebSocketStreamer(ws_config)
            if await ws_streamer.start():
                self.streamer.add_protocol("websocket", ws_streamer)
                logger.info(f"WebSocket server: http://{self.config.host}:{self.config.port}")
            else:
                logger.error("Failed to start WebSocket server")

        if self.config.protocol in (StreamProtocol.UDP, StreamProtocol.BOTH):
            udp_config = UDPConfig(
                host=self.config.host,
                port=self.config.port + 1 if self.config.protocol == StreamProtocol.BOTH else self.config.port,
                target_host=self.config.target_host,
                target_port=self.config.target_port,
                compression=CompressionFormat.JPEG,
                jpeg_quality=self.config.jpeg_quality
            )
            udp_streamer = UDPStreamer(udp_config)
            if await udp_streamer.start():
                self.streamer.add_protocol("udp", udp_streamer)
                logger.info(f"UDP streamer: sending to {self.config.target_host}:{self.config.target_port}")
            else:
                logger.error("Failed to start UDP streamer")

        if not self.streamer.protocols:
            logger.error("No streaming protocols initialized")
            return False

        logger.info("VR Streamer initialized successfully")
        return True

    async def start(self):
        """Start the streaming pipeline."""
        if not self.camera:
            raise RuntimeError("Camera not initialized")

        self._running = True
        self.camera.start()

        # Start stats printer
        if self.config.show_stats:
            self._stats_task = asyncio.create_task(self._print_stats_loop())

        logger.info("VR Streamer started - streaming...")

        await self._main_loop()

    async def _main_loop(self):
        """Main streaming loop."""
        frame_interval = 1.0 / self.config.camera_fps
        last_frame_time = time.time()

        while self._running:
            loop_start = time.perf_counter()

            try:
                # 1. Capture frame
                capture_start = time.perf_counter()
                frame = self.camera.capture_frame()
                capture_time = (time.perf_counter() - capture_start) * 1000

                if frame is None:
                    await asyncio.sleep(0.001)
                    continue

                self._frame_count += 1
                self._stats['frames_captured'] += 1

                # 2. YOLO processing (async - doesn't block)
                metadata = None
                if self.yolo:
                    if isinstance(self.yolo, AsyncYOLOProcessor):
                        # Submit frame (non-blocking)
                        self.yolo.submit(frame.data, frame.frame_id)
                        # Get latest result (non-blocking)
                        detection = self.yolo.get_result()
                    else:
                        # Sync processing
                        detection = self.yolo.process(frame.data, frame.frame_id)

                    if detection:
                        self._last_detection = detection
                        self._stats['frames_processed_yolo'] += 1
                        self._stats['detections'] = detection.count

                        # Prepare metadata for streaming (ensure JSON serializable)
                        metadata = {
                            'detections': [d.to_dict() for d in detection.detections],
                            'inference_time_ms': float(detection.inference_time_ms)
                        }

                # 3. Draw detections on frame (if enabled)
                display_frame = frame.data
                if self.config.yolo_draw and self._last_detection:
                    display_frame = draw_detections(frame.data.copy(), self._last_detection)

                # 4. Stream frame
                stream_start = time.perf_counter()
                results = await self.streamer.broadcast_to_all(
                    display_frame,
                    frame_id=frame.frame_id,
                    timestamp=time.time(),
                    metadata=metadata
                )
                stream_time = (time.perf_counter() - stream_start) * 1000

                self._stats['frames_streamed'] += 1
                self._stats['clients'] = self.streamer.get_total_clients()

                # Update timing stats
                total_time = (time.perf_counter() - loop_start) * 1000
                alpha = 0.1
                self._stats['avg_capture_ms'] = alpha * capture_time + (1 - alpha) * self._stats['avg_capture_ms']
                self._stats['avg_stream_ms'] = alpha * stream_time + (1 - alpha) * self._stats['avg_stream_ms']
                self._stats['avg_total_ms'] = alpha * total_time + (1 - alpha) * self._stats['avg_total_ms']

                # 5. Frame rate control
                elapsed = time.time() - last_frame_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                last_frame_time = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                await asyncio.sleep(0.01)

    async def _print_stats_loop(self):
        """Periodically print statistics."""
        while self._running:
            await asyncio.sleep(self.config.stats_interval)

            if not self._running:
                break

            stats = self._stats
            fps = stats['frames_captured'] / self.config.stats_interval if self.config.stats_interval > 0 else 0
            self._stats['frames_captured'] = 0  # Reset counter

            logger.info(
                f"FPS: {fps:.1f} | "
                f"Capture: {stats['avg_capture_ms']:.1f}ms | "
                f"Stream: {stats['avg_stream_ms']:.1f}ms | "
                f"Total: {stats['avg_total_ms']:.1f}ms | "
                f"Clients: {stats['clients']} | "
                f"Detections: {stats['detections']}"
            )

    async def stop(self):
        """Stop the streaming pipeline."""
        logger.info("Stopping VR Streamer...")
        self._running = False

        # Cancel stats task
        if self._stats_task:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass

        # Stop components
        if self.yolo and isinstance(self.yolo, AsyncYOLOProcessor):
            self.yolo.stop()

        if self.streamer:
            await self.streamer.stop_all()

        if self.camera:
            self.camera.close()

        logger.info("VR Streamer stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self._stats.copy()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VR Streamer - High-performance video streaming for VR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Camera settings
    camera_group = parser.add_argument_group("Camera")
    camera_group.add_argument("--width", type=int, default=1280, help="Camera width")
    camera_group.add_argument("--height", type=int, default=720, help="Camera height")
    camera_group.add_argument("--fps", type=int, default=30, help="Camera FPS")
    camera_group.add_argument("--stereo", action="store_true", help="Enable stereo camera mode")
    camera_group.add_argument(
        "--camera-profile",
        choices=["ultra_low_latency", "low_latency", "balanced", "high_quality"],
        default="low_latency",
        help="Camera latency profile"
    )

    # Streaming settings
    stream_group = parser.add_argument_group("Streaming")
    stream_group.add_argument(
        "--protocol",
        choices=["udp", "websocket", "both"],
        default="websocket",
        help="Streaming protocol"
    )
    stream_group.add_argument("--host", default="0.0.0.0", help="Server bind address")
    stream_group.add_argument("--port", type=int, default=8000, help="Server port")
    stream_group.add_argument("--target", default="", help="UDP target IP address")
    stream_group.add_argument("--target-port", type=int, default=5000, help="UDP target port")
    stream_group.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100)")

    # YOLO settings
    yolo_group = parser.add_argument_group("YOLO Detection")
    yolo_group.add_argument("--yolo", action="store_true", help="Enable YOLO detection")
    yolo_group.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model path")
    yolo_group.add_argument(
        "--yolo-backend",
        choices=["opencv_dnn", "pytorch", "onnx", "ncnn", "tflite", "edgetpu", "hailo"],
        default="opencv_dnn",
        help="YOLO backend (opencv_dnn recommended for Pi)"
    )
    yolo_group.add_argument("--yolo-skip", type=int, default=10, help="Process every N-th frame (10 recommended)")
    yolo_group.add_argument("--yolo-conf", type=float, default=0.3, help="Detection confidence threshold")
    yolo_group.add_argument("--yolo-input-size", type=int, default=320, help="YOLO input size (320 for speed)")
    yolo_group.add_argument("--yolo-sync", action="store_true", help="Use synchronous YOLO (not recommended)")
    yolo_group.add_argument("--no-draw", action="store_true", help="Don't draw detections on frame")
    yolo_group.add_argument("--yolo-persons-only", action="store_true", help="Detect only persons (class 0)")

    # Performance settings
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--cpu-camera", type=int, nargs="+", help="CPU affinity for camera (e.g., 2)")
    perf_group.add_argument("--cpu-yolo", type=int, nargs="+", help="CPU affinity for YOLO (e.g., 3)")
    perf_group.add_argument("--cpu-network", type=int, nargs="+", help="CPU affinity for network")

    # Debug settings
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument("--no-stats", action="store_true", help="Disable stats output")
    debug_group.add_argument("--stats-interval", type=float, default=5.0, help="Stats print interval")
    debug_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    return parser.parse_args()


def create_config_from_args(args) -> VRStreamerConfig:
    """Create configuration from command line arguments."""
    profile_map = {
        "ultra_low_latency": CameraProfile.ULTRA_LOW_LATENCY,
        "low_latency": CameraProfile.LOW_LATENCY,
        "balanced": CameraProfile.BALANCED,
        "high_quality": CameraProfile.HIGH_QUALITY
    }

    protocol_map = {
        "udp": StreamProtocol.UDP,
        "websocket": StreamProtocol.WEBSOCKET,
        "both": StreamProtocol.BOTH
    }

    backend_map = {
        "opencv_dnn": YOLOBackend.OPENCV_DNN,
        "pytorch": YOLOBackend.PYTORCH,
        "onnx": YOLOBackend.ONNX,
        "ncnn": YOLOBackend.NCNN,
        "tflite": YOLOBackend.TFLITE,
        "edgetpu": YOLOBackend.EDGETPU,
        "hailo": YOLOBackend.HAILO
    }

    # Auto-detect OpenCV DNN paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_dir = os.path.join(script_dir, "..", "yolo")

    yolo_backend = backend_map.get(args.yolo_backend, YOLOBackend.OPENCV_DNN)

    # Set OpenCV DNN paths if using that backend
    opencv_config = None
    opencv_weights = None
    opencv_names = None
    if yolo_backend == YOLOBackend.OPENCV_DNN:
        opencv_config = os.path.join(yolo_dir, "yolov4-tiny.cfg")
        opencv_weights = os.path.join(yolo_dir, "yolov4-tiny.weights")
        opencv_names = os.path.join(yolo_dir, "coco.names")

    # Filter classes if persons-only
    filter_classes = [0] if hasattr(args, 'yolo_persons_only') and args.yolo_persons_only else None

    return VRStreamerConfig(
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        camera_stereo=args.stereo,
        camera_profile=profile_map.get(args.camera_profile, CameraProfile.LOW_LATENCY),

        protocol=protocol_map.get(args.protocol, StreamProtocol.WEBSOCKET),
        host=args.host,
        port=args.port,
        target_host=args.target,
        target_port=args.target_port,
        jpeg_quality=args.quality,

        yolo_enabled=args.yolo,
        yolo_model=opencv_weights if yolo_backend == YOLOBackend.OPENCV_DNN else args.yolo_model,
        yolo_backend=yolo_backend,
        yolo_skip_frames=args.yolo_skip,
        yolo_confidence=args.yolo_conf,
        yolo_input_size=args.yolo_input_size if hasattr(args, 'yolo_input_size') else 320,
        yolo_async=not args.yolo_sync,
        yolo_draw=not args.no_draw,
        yolo_filter_classes=filter_classes,
        yolo_opencv_config=opencv_config,
        yolo_opencv_weights=opencv_weights,
        yolo_opencv_names=opencv_names,

        cpu_affinity_camera=args.cpu_camera,
        cpu_affinity_yolo=args.cpu_yolo,
        cpu_affinity_network=args.cpu_network,

        show_stats=not args.no_stats,
        stats_interval=args.stats_interval,
        log_level=args.log_level
    )


async def main():
    """Main entry point."""
    args = parse_args()
    config = create_config_from_args(args)

    streamer = VRStreamer(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(streamer.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Initialize and start
    if not await streamer.initialize():
        logger.error("Failed to initialize streamer")
        sys.exit(1)

    try:
        await streamer.start()
    except KeyboardInterrupt:
        pass
    finally:
        await streamer.stop()


if __name__ == "__main__":
    asyncio.run(main())