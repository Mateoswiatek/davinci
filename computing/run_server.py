#!/usr/bin/env python3
"""
Quick Start Server

Simple launcher for the VR streaming server with sensible defaults.

Usage:
    python run_server.py                      # WebSocket on port 8000
    python run_server.py --with-udp           # WebSocket + UDP
    python run_server.py --with-yolo          # With YOLO detection
    python run_server.py --target 192.168.1.X # UDP target for VR headset
"""

import asyncio
import argparse
import logging
import signal
import sys

# Setup path
sys.path.insert(0, str(__file__).rsplit('/', 1)[0])

from vr_streamer import VRStreamer, VRStreamerConfig, StreamProtocol
from camera_capture import CameraProfile
from yolo_processor import YOLOBackend


def main():
    parser = argparse.ArgumentParser(description="VR Streaming Server")

    # Quick options
    parser.add_argument("--with-udp", action="store_true",
                        help="Enable UDP streaming alongside WebSocket")
    parser.add_argument("--with-yolo", action="store_true",
                        help="Enable YOLO detection")
    parser.add_argument("--target", default="",
                        help="UDP target IP (e.g., 192.168.1.100)")

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

        yolo_enabled=args.with_yolo,
        yolo_model="yolov8n.pt",
        yolo_skip_frames=3,
        yolo_async=True,
        yolo_draw=True,

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
        print(f"  YOLO: {config.yolo_model} (every {config.yolo_skip_frames} frames)")
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