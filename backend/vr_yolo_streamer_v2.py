#!/usr/bin/env python3
"""
VR Streaming with Optimized YOLO Detection V2
Production-ready pipeline with async YOLO processing

Performance targets:
- Streaming: 30 FPS (1920x1080)
- Detection: 10 FPS (every 3rd frame)
- Total latency: <50ms

Author: DaVinci VR Project
Date: 2025-11-26
"""

import time
import logging
import signal
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from picamera2 import Picamera2

from yolo_processor_optimized import (
    AsyncYOLOProcessor,
    YOLOConfig,
    ProcessingConfig,
    ModelFormat,
    FrameSkipStrategy
)
from vr_udp_streamer import UDPFrameStreamer, StreamConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    fps: float = 0.0
    capture_ms: float = 0.0
    yolo_ms: float = 0.0
    annotate_ms: float = 0.0
    stream_ms: float = 0.0
    total_ms: float = 0.0
    frames_sent: int = 0


class OptimizedVRYOLOStreamer:
    """
    Production VR streaming pipeline with async YOLO

    Architecture:
    1. Main thread: Camera capture + UDP streaming (30 FPS)
    2. Worker process: YOLO inference (10 FPS, async)
    3. Results cached and annotated in main thread

    This ensures streaming never blocks on YOLO!
    """

    def __init__(
        self,
        stream_config: StreamConfig,
        yolo_config: YOLOConfig,
        processing_config: ProcessingConfig,
        enable_yolo: bool = True,
        enable_annotation: bool = True
    ):
        self.stream_config = stream_config
        self.enable_yolo = enable_yolo
        self.enable_annotation = enable_annotation

        # Initialize components
        logger.info("Initializing VR YOLO Streamer V2...")

        # UDP streamer
        self.streamer = UDPFrameStreamer(stream_config)
        logger.info(f"UDP streamer initialized: {stream_config.vr_host}:{stream_config.vr_port}")

        # YOLO processor (async)
        if enable_yolo:
            self.yolo_processor = AsyncYOLOProcessor(yolo_config, processing_config)
            logger.info("YOLO processor initialized (async)")
        else:
            self.yolo_processor = None
            logger.info("YOLO disabled")

        # Camera
        self.picam2 = Picamera2()
        self._configure_camera()

        # Metrics
        self.metrics = PipelineMetrics()
        self.last_metric_time = time.time()
        self.last_metric_frames = 0

        # Shutdown handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = False

        logger.info("Initialization complete")

    def _configure_camera(self):
        """Configure Pi Camera for low-latency capture"""
        frame_duration_us = int(1_000_000 / self.stream_config.fps)

        config = self.picam2.create_video_configuration(
            main={
                "size": (self.stream_config.width, self.stream_config.height),
                "format": self.stream_config.format
            },
            buffer_count=2,  # Minimal buffering
            controls={
                "FrameDurationLimits": (frame_duration_us, frame_duration_us),
                "ExposureTime": 10000,  # 10ms exposure (low latency)
                "AnalogueGain": 2.0,
                "AeEnable": False,  # Disable auto-exposure (saves time)
                "AwbEnable": False,  # Disable auto-white-balance
            }
        )

        self.picam2.configure(config)
        logger.info(
            f"Camera configured: {self.stream_config.width}x{self.stream_config.height} "
            f"@ {self.stream_config.fps} FPS"
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _update_metrics(self):
        """Update and log performance metrics"""
        current_time = time.time()
        elapsed = current_time - self.last_metric_time

        if elapsed >= 5.0:  # Log every 5 seconds
            frames = self.metrics.frames_sent - self.last_metric_frames
            self.metrics.fps = frames / elapsed

            # Build log message
            log_parts = [
                f"Performance: {self.metrics.fps:.1f} FPS",
                f"Total: {self.metrics.total_ms:.1f}ms",
                f"(Capture: {self.metrics.capture_ms:.1f}ms",
            ]

            if self.yolo_processor:
                yolo_stats = self.yolo_processor.get_stats()
                if yolo_stats:
                    log_parts.append(
                        f"YOLO: {yolo_stats['avg_inference_ms']:.1f}ms "
                        f"@ {yolo_stats['detection_fps']:.1f} det/s"
                    )
                if self.enable_annotation:
                    log_parts.append(f"Annotate: {self.metrics.annotate_ms:.1f}ms")

            log_parts.append(f"Stream: {self.metrics.stream_ms:.1f}ms)")

            logger.info(" ".join(log_parts))

            # Warnings
            if self.metrics.total_ms > 50:
                logger.warning(
                    f"Total latency {self.metrics.total_ms:.1f}ms exceeds VR target (50ms)!"
                )

            if self.metrics.fps < self.stream_config.fps * 0.9:
                logger.warning(
                    f"FPS {self.metrics.fps:.1f} below target {self.stream_config.fps}!"
                )

            # Reset counters
            self.last_metric_time = current_time
            self.last_metric_frames = self.metrics.frames_sent

    def run(self):
        """
        Main pipeline loop

        Pipeline stages (all in main thread):
        1. Capture frame (3-5ms)
        2. Submit to YOLO worker (non-blocking, <0.1ms)
        3. Get latest YOLO result (non-blocking, <0.1ms)
        4. Annotate frame if enabled (2-3ms)
        5. Stream via UDP (8-10ms)
        6. Maintain target FPS

        Total: ~15-20ms per frame (well under 50ms target!)
        """
        self.running = True
        self.picam2.start()

        logger.info("=" * 60)
        logger.info("VR YOLO Streaming V2 - STARTED")
        logger.info("=" * 60)
        logger.info(f"Resolution: {self.stream_config.width}x{self.stream_config.height}")
        logger.info(f"Target FPS: {self.stream_config.fps}")
        logger.info(f"YOLO: {'ENABLED' if self.enable_yolo else 'DISABLED'}")
        logger.info(f"Annotation: {'ENABLED' if self.enable_annotation else 'DISABLED'}")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        frame_time = 1.0 / self.stream_config.fps

        try:
            while self.running:
                loop_start = time.perf_counter()

                # === Stage 1: Capture ===
                capture_start = time.perf_counter()
                frame = self.picam2.capture_array("main")
                self.metrics.capture_ms = (time.perf_counter() - capture_start) * 1000

                # === Stage 2: Submit to YOLO (non-blocking) ===
                if self.yolo_processor:
                    self.yolo_processor.submit_frame(frame)

                # === Stage 3: Get latest YOLO result (non-blocking) ===
                yolo_result = None
                if self.yolo_processor:
                    yolo_result = self.yolo_processor.get_latest_result()

                # === Stage 4: Annotate (optional) ===
                if self.enable_annotation and yolo_result:
                    annotate_start = time.perf_counter()
                    frame_to_send = self.yolo_processor.draw_detections(
                        frame,
                        yolo_result,
                        show_info=True
                    )
                    self.metrics.annotate_ms = (time.perf_counter() - annotate_start) * 1000
                else:
                    frame_to_send = frame
                    self.metrics.annotate_ms = 0.0

                # === Stage 5: UDP Stream ===
                stream_start = time.perf_counter()
                self.streamer.send_frame(frame_to_send)
                self.metrics.stream_ms = (time.perf_counter() - stream_start) * 1000

                # === Stage 6: Maintain FPS ===
                self.metrics.total_ms = (time.perf_counter() - loop_start) * 1000
                self.metrics.frames_sent += 1

                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                # Update metrics
                self._update_metrics()

        except KeyboardInterrupt:
            logger.info("Streaming stopped by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")

        if self.picam2:
            self.picam2.stop()

        if self.streamer:
            self.streamer.close()

        if self.yolo_processor:
            self.yolo_processor.stop()

        # Final statistics
        logger.info("=" * 60)
        logger.info("Final Statistics")
        logger.info("=" * 60)
        logger.info(f"Total frames sent: {self.metrics.frames_sent}")
        logger.info(f"Final FPS: {self.metrics.fps:.1f}")
        logger.info(f"Avg capture: {self.metrics.capture_ms:.1f}ms")
        if self.yolo_processor:
            self.yolo_processor.print_stats()
        if self.enable_annotation:
            logger.info(f"Avg annotate: {self.metrics.annotate_ms:.1f}ms")
        logger.info(f"Avg stream: {self.metrics.stream_ms:.1f}ms")
        logger.info(f"Avg total: {self.metrics.total_ms:.1f}ms")
        logger.info("=" * 60)


def main():
    """Entry point with CLI"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimized VR YOLO Streaming Server V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CPU-only with NCNN (fastest CPU-only)
  python vr_yolo_streamer_v2.py --model yolov8n.ncnn --format ncnn --skip 3

  # Coral EdgeTPU
  python vr_yolo_streamer_v2.py --model yolov8n_edgetpu.tflite --format edgetpu --skip 1

  # Hailo-8L (if available)
  python vr_yolo_streamer_v2.py --model yolov8n.hef --format hef --skip 1

  # No YOLO (streaming only)
  python vr_yolo_streamer_v2.py --no-yolo
        """
    )

    # Streaming options
    parser.add_argument('--host', type=str, default='192.168.1.100',
                        help='VR headset IP address')
    parser.add_argument('--port', type=int, default=5000,
                        help='UDP port')
    parser.add_argument('--width', type=int, default=1920,
                        help='Frame width')
    parser.add_argument('--height', type=int, default=1080,
                        help='Frame height')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS')

    # YOLO options
    parser.add_argument('--no-yolo', action='store_true',
                        help='Disable YOLO detection')
    parser.add_argument('--no-annotation', action='store_true',
                        help='Disable drawing boxes (detection still runs)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model path')
    parser.add_argument('--format', type=str, default='pt',
                        choices=['pt', 'onnx', 'ncnn', 'edgetpu', 'hef'],
                        help='Model format')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--skip', type=int, default=3,
                        help='Process every N-th frame (1=all, 3=every 3rd)')
    parser.add_argument('--strategy', type=str, default='fixed',
                        choices=['none', 'fixed', 'adaptive', 'queue'],
                        help='Frame skip strategy')

    # Advanced options
    parser.add_argument('--queue-size', type=int, default=2,
                        help='Max queue size for YOLO worker')
    parser.add_argument('--cores', type=str, default=None,
                        help='CPU cores for YOLO worker (e.g., "2,3")')
    parser.add_argument('--priority', type=int, default=-5,
                        help='YOLO worker priority (negative=higher, requires root)')

    args = parser.parse_args()

    # Validate
    if args.no_yolo and args.no_annotation:
        logger.warning("Both YOLO and annotation disabled - running streaming only")

    # Parse CPU affinity
    cpu_affinity = None
    if args.cores:
        try:
            cpu_affinity = [int(c.strip()) for c in args.cores.split(',')]
        except ValueError:
            logger.error(f"Invalid CPU cores format: {args.cores}")
            sys.exit(1)

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
        model_format=ModelFormat(args.format),
        confidence=args.conf,
        input_size=(640, 640),
        device="cpu"
    )

    processing_config = ProcessingConfig(
        max_queue_size=args.queue_size,
        frame_skip_strategy=FrameSkipStrategy(args.strategy),
        frame_skip_interval=args.skip,
        enable_smoothing=True,
        cpu_affinity=cpu_affinity,
        priority=args.priority
    )

    # Display configuration
    logger.info("=" * 60)
    logger.info("Configuration")
    logger.info("=" * 60)
    logger.info(f"Resolution: {stream_config.width}x{stream_config.height}")
    logger.info(f"Target FPS: {stream_config.fps}")
    logger.info(f"Destination: {stream_config.vr_host}:{stream_config.vr_port}")
    logger.info("-" * 60)

    if not args.no_yolo:
        logger.info(f"YOLO Model: {args.model}")
        logger.info(f"Format: {args.format}")
        logger.info(f"Skip strategy: {args.strategy}")
        if args.strategy == 'fixed':
            logger.info(f"Skip interval: every {args.skip} frames")
        logger.info(f"Confidence: {args.conf}")
        logger.info(f"Queue size: {args.queue_size}")
        if cpu_affinity:
            logger.info(f"CPU affinity: {cpu_affinity}")
        logger.info(f"Priority: {args.priority}")
    else:
        logger.info("YOLO: DISABLED")

    logger.info("-" * 60)

    # Calculate bandwidth
    bytes_per_frame = stream_config.width * stream_config.height * 3
    mbps = (bytes_per_frame * stream_config.fps * 8) / 1_000_000
    logger.info(f"Bandwidth: ~{mbps:.0f} Mbps")

    if mbps > 900:
        logger.warning("Bandwidth exceeds Gigabit Ethernet capacity!")
        logger.warning("Consider lowering resolution or FPS")

    # Estimate latency
    est_latency = 5  # Capture
    if not args.no_yolo:
        # Estimate based on format
        if args.format == 'hef':  # Hailo
            est_latency += 3
        elif args.format == 'edgetpu':  # Coral
            est_latency += 17
        elif args.format == 'ncnn':  # NCNN
            est_latency += 80 / args.skip  # Amortized
        else:  # PyTorch
            est_latency += 150 / args.skip
    if not args.no_annotation:
        est_latency += 2
    est_latency += 10  # UDP

    logger.info(f"Estimated latency: ~{est_latency:.0f}ms")

    if est_latency > 50:
        logger.warning(f"Estimated latency ({est_latency:.0f}ms) may exceed VR target (50ms)")
        logger.info("Consider:")
        logger.info("  - Increasing --skip value")
        logger.info("  - Using faster model format (NCNN/EdgeTPU/Hailo)")
        logger.info("  - Disabling annotation (--no-annotation)")

    logger.info("=" * 60)
    logger.info("Starting in 3 seconds...")
    logger.info("=" * 60)

    time.sleep(3)

    # Start streaming
    streamer = OptimizedVRYOLOStreamer(
        stream_config=stream_config,
        yolo_config=yolo_config,
        processing_config=processing_config,
        enable_yolo=not args.no_yolo,
        enable_annotation=not args.no_annotation
    )

    streamer.run()


if __name__ == "__main__":
    main()