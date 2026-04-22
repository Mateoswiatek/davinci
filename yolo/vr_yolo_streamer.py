#!/usr/bin/env python3
"""
VR Streaming with YOLO Object Detection
Integrates YOLOv8 detection with ultra-low-latency UDP streaming

Target latency: <50ms (capture + YOLO + streaming)
- Capture: ~2-5ms
- YOLO inference: ~20-30ms (YOLOv8n on Pi 5)
- UDP send: ~10-15ms
Total: ~40-50ms (acceptable for VR)

Author: DaVinci VR Project
Date: 2025-11-26
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from picamera2 import Picamera2

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logging.warning("ultralytics not installed. YOLO detection disabled.")

from vr_udp_streamer import UDPFrameStreamer, StreamConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class YOLOConfig:
    """YOLO detection configuration"""
    model_path: str = "yolov8n.pt"  # Nano - fastest
    confidence: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 50
    classes: Optional[list] = None  # None = all classes
    device: str = "cpu"  # Pi 5 doesn't have CUDA


class YOLODetector:
    """YOLOv8 object detector optimized for VR"""

    def __init__(self, config: YOLOConfig):
        if not HAS_YOLO:
            raise RuntimeError("ultralytics package not installed")

        self.config = config
        self.model = YOLO(config.model_path)

        # Inference statistics
        self.inference_times = []
        self.max_history = 100

        logger.info(f"YOLO model loaded: {config.model_path}")
        logger.info(f"Device: {config.device}")

    def detect(self, frame: np.ndarray, annotate: bool = True) -> np.ndarray:
        """
        Run object detection on frame

        Args:
            frame: Input frame (height, width, 3)
            annotate: If True, draw bounding boxes on frame

        Returns:
            Annotated frame if annotate=True, else original frame
        """
        start_time = time.time()

        # Run inference
        results = self.model(
            frame,
            conf=self.config.confidence,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            classes=self.config.classes,
            verbose=False,
            device=self.config.device
        )

        # Track inference time
        inference_ms = (time.time() - start_time) * 1000
        self.inference_times.append(inference_ms)
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)

        # Annotate if requested
        if annotate:
            annotated = results[0].plot()
            return annotated
        else:
            return frame

    def get_avg_inference_time(self) -> float:
        """Get average inference time in ms"""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)


class VRYOLOStreamer:
    """
    Complete VR pipeline with YOLO detection
    Optimized for minimal latency while running object detection
    """

    def __init__(
        self,
        stream_config: StreamConfig,
        yolo_config: YOLOConfig,
        enable_yolo: bool = True
    ):
        self.stream_config = stream_config
        self.enable_yolo = enable_yolo

        # Initialize components
        self.streamer = UDPFrameStreamer(stream_config)

        if enable_yolo and HAS_YOLO:
            self.detector = YOLODetector(yolo_config)
        else:
            self.detector = None
            if enable_yolo:
                logger.warning("YOLO requested but not available. Streaming without detection.")

        # Initialize camera
        self.picam2 = Picamera2()
        self._configure_camera()

        # Performance metrics
        self.fps = 0.0
        self.capture_ms = 0.0
        self.yolo_ms = 0.0
        self.stream_ms = 0.0
        self.total_ms = 0.0

        self.frames_sent = 0
        self.last_metric_time = time.time()
        self.last_metric_frames = 0

    def _configure_camera(self):
        """Configure camera for VR capture"""
        frame_duration_us = int(1_000_000 / self.stream_config.fps)

        config = self.picam2.create_video_configuration(
            main={
                "size": (self.stream_config.width, self.stream_config.height),
                "format": self.stream_config.format
            },
            buffer_count=2,
            controls={
                "FrameDurationLimits": (frame_duration_us, frame_duration_us),
                "ExposureTime": 10000,
                "AnalogueGain": 2.0,
            }
        )

        self.picam2.configure(config)
        logger.info(f"Camera configured: {self.stream_config.width}x{self.stream_config.height} @ {self.stream_config.fps}fps")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optional YOLO processing"""
        if self.detector is None:
            return frame

        return self.detector.detect(frame, annotate=True)

    def _update_metrics(self):
        """Update and log performance metrics"""
        current_time = time.time()
        elapsed = current_time - self.last_metric_time

        if elapsed >= 5.0:
            frames = self.frames_sent - self.last_metric_frames
            self.fps = frames / elapsed

            log_msg = (
                f"Performance: {self.fps:.1f} FPS | "
                f"Total: {self.total_ms:.1f}ms "
                f"(Capture: {self.capture_ms:.1f}ms"
            )

            if self.detector:
                avg_yolo = self.detector.get_avg_inference_time()
                log_msg += f", YOLO: {avg_yolo:.1f}ms"

            log_msg += f", Stream: {self.stream_ms:.1f}ms)"
            logger.info(log_msg)

            # Latency warning
            if self.total_ms > 50:
                logger.warning(f"Total latency {self.total_ms:.1f}ms exceeds VR target (50ms)!")

            self.last_metric_time = current_time
            self.last_metric_frames = self.frames_sent

    def run(self):
        """
        Main pipeline loop

        Pipeline stages:
        1. Capture (zero-copy)
        2. YOLO detection (optional)
        3. UDP streaming
        4. FPS maintenance
        """
        self.picam2.start()
        logger.info("VR YOLO streaming started")

        frame_time = 1.0 / self.stream_config.fps

        try:
            while True:
                loop_start = time.time()

                # Stage 1: Capture
                capture_start = time.time()
                frame = self.picam2.capture_array("main")
                self.capture_ms = (time.time() - capture_start) * 1000

                # Stage 2: YOLO processing (if enabled)
                if self.detector:
                    yolo_start = time.time()
                    processed_frame = self._process_frame(frame)
                    self.yolo_ms = (time.time() - yolo_start) * 1000
                else:
                    processed_frame = frame
                    self.yolo_ms = 0.0

                # Stage 3: UDP streaming
                stream_start = time.time()
                self.streamer.send_frame(processed_frame)
                self.stream_ms = (time.time() - stream_start) * 1000

                # Total latency
                self.total_ms = (time.time() - loop_start) * 1000

                # Update counters
                self.frames_sent += 1

                # Stage 4: Maintain FPS
                elapsed = time.time() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                self._update_metrics()

        except KeyboardInterrupt:
            logger.info("Streaming stopped by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.picam2.stop()
        self.streamer.close()

        logger.info("=" * 60)
        logger.info("Final Statistics:")
        logger.info(f"  Total frames: {self.frames_sent}")
        logger.info(f"  Final FPS: {self.fps:.1f}")
        logger.info(f"  Avg capture: {self.capture_ms:.1f}ms")
        if self.detector:
            logger.info(f"  Avg YOLO: {self.detector.get_avg_inference_time():.1f}ms")
        logger.info(f"  Avg stream: {self.stream_ms:.1f}ms")
        logger.info(f"  Avg total: {self.total_ms:.1f}ms")
        logger.info("=" * 60)


def main():
    """Entry point with CLI arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='VR YOLO Streaming Server')

    # Streaming options
    parser.add_argument('--host', type=str, default='192.168.1.100',
                        help='VR headset IP address')
    parser.add_argument('--port', type=int, default=5000,
                        help='UDP port')
    parser.add_argument('--width', type=int, default=2560,
                        help='Frame width')
    parser.add_argument('--height', type=int, default=800,
                        help='Frame height')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS')

    # YOLO options
    parser.add_argument('--no-yolo', action='store_true',
                        help='Disable YOLO detection')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')

    args = parser.parse_args()

    # Create configurations
    stream_config = StreamConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        format="RGB888",
        vr_host=args.host,
        vr_port=args.port,
        max_packet_size=8192
    )

    yolo_config = YOLOConfig(
        model_path=args.model,
        confidence=args.conf,
        device="cpu"
    )

    # Display configuration
    logger.info("=" * 60)
    logger.info("VR YOLO Streaming Server")
    logger.info("=" * 60)
    logger.info(f"Resolution: {stream_config.width}x{stream_config.height}")
    logger.info(f"FPS: {stream_config.fps}")
    logger.info(f"Destination: {stream_config.vr_host}:{stream_config.vr_port}")
    logger.info(f"YOLO: {'DISABLED' if args.no_yolo else f'ENABLED ({args.model})'}")

    # Calculate bandwidth
    bytes_per_frame = stream_config.width * stream_config.height * 3
    mbps = (bytes_per_frame * stream_config.fps * 8) / 1_000_000
    logger.info(f"Bandwidth: ~{mbps:.0f} Mbps")

    if mbps > 900:
        logger.warning("Bandwidth exceeds Gigabit Ethernet capacity!")
        logger.warning("Consider lowering resolution or FPS")

    logger.info("=" * 60)

    # Latency estimation
    estimated_latency = 5  # Capture
    if not args.no_yolo:
        estimated_latency += 25  # YOLO on Pi 5
    estimated_latency += 12  # UDP send

    logger.info(f"Estimated latency: ~{estimated_latency}ms")

    if estimated_latency > 50:
        logger.warning(f"Estimated latency ({estimated_latency}ms) exceeds VR target (50ms)!")
        logger.warning("Consider disabling YOLO (--no-yolo) or lowering resolution")

    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Start streaming
    streamer = VRYOLOStreamer(
        stream_config=stream_config,
        yolo_config=yolo_config,
        enable_yolo=not args.no_yolo
    )
    streamer.run()


if __name__ == "__main__":
    main()