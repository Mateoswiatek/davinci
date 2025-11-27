#!/usr/bin/env python3
"""
Quick Start Server

Simple launcher for the VR streaming server with sensible defaults.

Usage:
    python run_server.py                      # WebSocket on port 8000
    python run_server.py --with-udp           # WebSocket + UDP
    python run_server.py --with-yolo          # With YOLO detection (OpenCV DNN - lightweight!)
    python run_server.py --with-yolo-pytorch  # With YOLO detection (PyTorch - heavier)
    python run_server.py --target 192.168.1.X # UDP target for VR headset
    python run_server.py --yolo-persons-only  # Detect only persons (class 0)

YOLO Backend Options:
    --yolo-backend opencv_dnn   # OpenCV DNN + YOLOv4-tiny (RECOMMENDED for Pi)
    --yolo-backend pytorch      # PyTorch/Ultralytics YOLOv8 (requires more RAM)
    --yolo-backend onnx         # ONNX Runtime
    --yolo-backend ncnn         # NCNN (fastest CPU)
"""

import asyncio
import argparse
import logging
import signal
import sys
import os

# Setup path
sys.path.insert(0, str(__file__).rsplit('/', 1)[0])

from vr_streamer import VRStreamer, VRStreamerConfig, StreamProtocol
from camera_capture import CameraProfile
from yolo_processor import YOLOBackend

# Default paths for YOLO models (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YOLO_DIR = os.path.join(SCRIPT_DIR, "..", "yolo")
DEFAULT_OPENCV_CONFIG = os.path.join(DEFAULT_YOLO_DIR, "yolov4-tiny.cfg")
DEFAULT_OPENCV_WEIGHTS = os.path.join(DEFAULT_YOLO_DIR, "yolov4-tiny.weights")
DEFAULT_OPENCV_NAMES = os.path.join(DEFAULT_YOLO_DIR, "coco.names")


def main():
    parser = argparse.ArgumentParser(description="VR Streaming Server")

    # Quick options
    parser.add_argument("--with-udp", action="store_true",
                        help="Enable UDP streaming alongside WebSocket")
    parser.add_argument("--with-yolo", action="store_true",
                        help="Enable YOLO detection (uses OpenCV DNN by default - lightweight!)")
    parser.add_argument("--with-yolo-pytorch", action="store_true",
                        help="Enable YOLO detection with PyTorch/Ultralytics (heavier, needs more RAM)")
    parser.add_argument("--target", default="",
                        help="UDP target IP (e.g., 192.168.1.100)")

    # YOLO options
    parser.add_argument("--yolo-backend", default="opencv_dnn",
                        choices=["opencv_dnn", "pytorch", "onnx", "ncnn", "tflite", "edgetpu", "hailo"],
                        help="YOLO backend (default: opencv_dnn - recommended for Pi)")
    parser.add_argument("--yolo-model", default="",
                        help="Path to YOLO model (auto-detected based on backend)")
    parser.add_argument("--yolo-skip", type=int, default=10,
                        help="Process YOLO every N frames (default: 10 for better performance)")
    parser.add_argument("--yolo-confidence", type=float, default=0.3,
                        help="YOLO confidence threshold (default: 0.3)")
    parser.add_argument("--yolo-input-size", type=int, default=320,
                        help="YOLO input size (default: 320 for speed, use 416 for accuracy)")
    parser.add_argument("--yolo-persons-only", action="store_true",
                        help="Detect only persons (class 0) - improves performance")

    # Overrides
    parser.add_argument("--port", type=int, default=8000,
                        help="WebSocket port")
    parser.add_argument("--width", type=int, default=1280,
                        help="Camera width")
    parser.add_argument("--height", type=int, default=720,
                        help="Camera height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Camera FPS")
    parser.add_argument("--quality", type=int, default=85,
                        help="JPEG quality")

    # CPU pinning for performance
    parser.add_argument("--cpu-camera", type=int, nargs="+",
                        help="CPU cores for camera capture (e.g., --cpu-camera 2)")
    parser.add_argument("--cpu-yolo", type=int, nargs="+",
                        help="CPU cores for YOLO processing (e.g., --cpu-yolo 3)")
    parser.add_argument("--cpu-network", type=int, nargs="+",
                        help="CPU cores for network streaming")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Configure logging - BEFORE any imports that use logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True  # Override any existing config
    )

    # Enable all loggers
    logging.getLogger('vr_streamer').setLevel(level)
    logging.getLogger('camera_capture').setLevel(level)
    logging.getLogger('protocols').setLevel(level)

    # Determine protocol
    if args.with_udp or args.target:
        protocol = StreamProtocol.BOTH
    else:
        protocol = StreamProtocol.WEBSOCKET

    # Determine YOLO settings
    yolo_enabled = args.with_yolo or args.with_yolo_pytorch

    # Select backend based on flags
    if args.with_yolo_pytorch:
        yolo_backend = YOLOBackend.PYTORCH
    else:
        yolo_backend = YOLOBackend(args.yolo_backend)

    # Auto-detect model path based on backend
    if args.yolo_model:
        yolo_model = args.yolo_model
    elif yolo_backend == YOLOBackend.OPENCV_DNN:
        yolo_model = DEFAULT_OPENCV_WEIGHTS
    elif yolo_backend == YOLOBackend.PYTORCH:
        yolo_model = "yolov8n.pt"
    else:
        yolo_model = args.yolo_model or "yolov8n.pt"

    # Filter classes (persons only = class 0)
    filter_classes = [0] if args.yolo_persons_only else None

    # Create config
    config = VRStreamerConfig(
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        camera_profile=CameraProfile.LOW_LATENCY,

        protocol=protocol,
        host="0.0.0.0",
        port=args.port,
        target_host=args.target,
        target_port=5000,
        jpeg_quality=args.quality,

        yolo_enabled=yolo_enabled,
        yolo_model=yolo_model,
        yolo_backend=yolo_backend,
        yolo_skip_frames=args.yolo_skip,
        yolo_async=True,
        yolo_draw=True,
        yolo_confidence=args.yolo_confidence,
        yolo_input_size=args.yolo_input_size,
        yolo_filter_classes=filter_classes,

        # OpenCV DNN specific paths
        yolo_opencv_config=DEFAULT_OPENCV_CONFIG if yolo_backend == YOLOBackend.OPENCV_DNN else None,
        yolo_opencv_weights=DEFAULT_OPENCV_WEIGHTS if yolo_backend == YOLOBackend.OPENCV_DNN else None,
        yolo_opencv_names=DEFAULT_OPENCV_NAMES if yolo_backend == YOLOBackend.OPENCV_DNN else None,

        # CPU pinning
        cpu_affinity_camera=args.cpu_camera,
        cpu_affinity_yolo=args.cpu_yolo,
        cpu_affinity_network=args.cpu_network,

        show_stats=True,
        stats_interval=5.0,
        log_level="DEBUG" if args.verbose else "INFO"
    )

    # Print startup info
    print("\n" + "="*60)
    print("  VR STREAMING SERVER")
    print("="*60)
    print(f"  Camera: {config.camera_width}x{config.camera_height} @ {config.camera_fps}fps")
    print(f"  Protocol: {config.protocol.value}")
    print(f"  WebSocket: http://0.0.0.0:{config.port}")
    if protocol in (StreamProtocol.UDP, StreamProtocol.BOTH):
        print(f"  UDP Target: {config.target_host}:{config.target_port}")
    if config.yolo_enabled:
        print(f"  YOLO Backend: {yolo_backend.value}")
        print(f"  YOLO Model: {yolo_model}")
        print(f"  YOLO Skip: every {config.yolo_skip_frames} frames")
        print(f"  YOLO Input: {args.yolo_input_size}x{args.yolo_input_size}")
        print(f"  YOLO Confidence: {args.yolo_confidence}")
        if args.yolo_persons_only:
            print(f"  YOLO Filter: persons only (class 0)")
    if args.cpu_camera or args.cpu_yolo or args.cpu_network:
        print(f"  CPU Pinning:")
        if args.cpu_camera:
            print(f"    Camera: CPU {args.cpu_camera}")
        if args.cpu_yolo:
            print(f"    YOLO: CPU {args.cpu_yolo}")
        if args.cpu_network:
            print(f"    Network: CPU {args.cpu_network}")
    print("="*60)
    print("  Press Ctrl+C to stop")
    print("="*60 + "\n")

    # Create and run streamer
    streamer = VRStreamer(config)

    async def run():
        if not await streamer.initialize():
            print("Failed to initialize streamer!")
            return

        try:
            await streamer.start()
        except KeyboardInterrupt:
            pass
        finally:
            await streamer.stop()

    # Handle signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(streamer.stop()))

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()