#!/usr/bin/env python3
"""
VR Streamer — Main Pipeline

Camera capture (Picamera2) → JPEG compression → WebSocket/UDP streaming.
Designed for Raspberry Pi 5 → Oculus Quest.

Usage:
    python vr_streamer.py                          # WebSocket, port 8000
    python vr_streamer.py --protocol udp --target 192.168.1.100
    python vr_streamer.py --width 1920 --height 1080 --fps 60
    python vr_streamer.py --help
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

import numpy as np

from camera_capture import CameraCapture, CameraConfig, CameraProfile, CapturedFrame
from protocols import StreamConfig, UDPStreamer, WebSocketStreamer, MultiProtocolStreamer
from protocols.udp_streamer import UDPConfig
from protocols.websocket_streamer import WebSocketConfig
from protocols.base import CompressionFormat

logger = logging.getLogger(__name__)


class StreamProtocol(Enum):
    UDP = "udp"
    WEBSOCKET = "websocket"
    BOTH = "both"


@dataclass
class VRStreamerConfig:
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
    target_host: str = ""
    target_port: int = 5000
    jpeg_quality: int = 85

    # Debug
    show_stats: bool = True
    stats_interval: float = 5.0
    log_level: str = "INFO"


class VRStreamer:
    """
    Main VR streaming pipeline.

    Camera capture runs in a background thread (CameraCapture).
    Streaming runs in the asyncio event loop.
    """

    def __init__(self, config: VRStreamerConfig):
        self.config = config
        self._running = False
        self.camera: Optional[CameraCapture] = None
        self.streamer: Optional[MultiProtocolStreamer] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._stats = {
            'frames_captured': 0,
            'frames_streamed': 0,
            'avg_capture_ms': 0.0,
            'avg_stream_ms': 0.0,
            'avg_total_ms': 0.0,
            'clients': 0,
        }

    async def initialize(self) -> bool:
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S',
        )

        camera_config = CameraConfig(
            width=self.config.camera_width,
            height=self.config.camera_height,
            fps=self.config.camera_fps,
            stereo=self.config.camera_stereo,
        )
        self.camera = CameraCapture(camera_config)
        if not self.camera.initialize():
            logger.error("Failed to initialize camera")
            return False
        logger.info(f"Camera: {self.config.camera_width}x{self.config.camera_height} @ {self.config.camera_fps}fps")

        self.streamer = MultiProtocolStreamer()

        if self.config.protocol in (StreamProtocol.WEBSOCKET, StreamProtocol.BOTH):
            ws_config = WebSocketConfig(
                host=self.config.host,
                port=self.config.port,
                compression=CompressionFormat.JPEG,
                jpeg_quality=self.config.jpeg_quality,
            )
            ws_streamer = WebSocketStreamer(ws_config)
            if await ws_streamer.start():
                self.streamer.add_protocol("websocket", ws_streamer)
                logger.info(f"WebSocket: ws://{self.config.host}:{self.config.port}")
            else:
                logger.error("Failed to start WebSocket server")
                return False

        if self.config.protocol in (StreamProtocol.UDP, StreamProtocol.BOTH):
            udp_port = self.config.port + 1 if self.config.protocol == StreamProtocol.BOTH else self.config.port
            udp_config = UDPConfig(
                host=self.config.host,
                port=udp_port,
                target_host=self.config.target_host,
                target_port=self.config.target_port,
                compression=CompressionFormat.JPEG,
                jpeg_quality=self.config.jpeg_quality,
            )
            udp_streamer = UDPStreamer(udp_config)
            if await udp_streamer.start():
                self.streamer.add_protocol("udp", udp_streamer)
                logger.info(f"UDP: {self.config.target_host}:{self.config.target_port}")
            else:
                logger.error("Failed to start UDP streamer")
                return False

        if not self.streamer.protocols:
            logger.error("No streaming protocols initialized")
            return False

        return True

    async def start(self):
        self._running = True
        self.camera.start()

        if self.config.show_stats:
            self._stats_task = asyncio.create_task(self._stats_loop())

        logger.info("Streaming started")
        await self._main_loop()

    async def _main_loop(self):
        frame_interval = 1.0 / self.config.camera_fps
        last_frame_time = time.time()
        alpha = 0.1

        while self._running:
            loop_start = time.perf_counter()

            try:
                capture_start = time.perf_counter()
                frame = self.camera.capture_frame()
                capture_ms = (time.perf_counter() - capture_start) * 1000

                if frame is None:
                    await asyncio.sleep(0.001)
                    continue

                self._stats['frames_captured'] += 1

                stream_start = time.perf_counter()
                await self.streamer.broadcast_to_all(
                    frame.data,
                    frame_id=frame.frame_id,
                    timestamp=time.time(),
                )
                stream_ms = (time.perf_counter() - stream_start) * 1000

                self._stats['frames_streamed'] += 1
                self._stats['clients'] = self.streamer.get_total_clients()

                total_ms = (time.perf_counter() - loop_start) * 1000
                self._stats['avg_capture_ms'] = alpha * capture_ms + (1 - alpha) * self._stats['avg_capture_ms']
                self._stats['avg_stream_ms'] = alpha * stream_ms + (1 - alpha) * self._stats['avg_stream_ms']
                self._stats['avg_total_ms'] = alpha * total_ms + (1 - alpha) * self._stats['avg_total_ms']

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

    async def _stats_loop(self):
        while self._running:
            await asyncio.sleep(self.config.stats_interval)
            if not self._running:
                break
            s = self._stats
            fps = s['frames_captured'] / self.config.stats_interval
            self._stats['frames_captured'] = 0
            logger.info(
                f"FPS: {fps:.1f} | Capture: {s['avg_capture_ms']:.1f}ms | "
                f"Stream: {s['avg_stream_ms']:.1f}ms | Total: {s['avg_total_ms']:.1f}ms | "
                f"Clients: {s['clients']}"
            )

    async def stop(self):
        logger.info("Stopping...")
        self._running = False

        if self._stats_task:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass

        if self.streamer:
            await self.streamer.stop_all()

        if self.camera:
            self.camera.close()

        logger.info("Stopped")

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()


def main():
    parser = argparse.ArgumentParser(description="VR Streamer")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--protocol", choices=["udp", "websocket", "both"], default="websocket")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--target", default="", help="UDP target IP")
    parser.add_argument("--target-port", type=int, default=5000)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--stats-interval", type=float, default=5.0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument(
        "--camera-profile",
        choices=["ultra_low_latency", "low_latency", "balanced", "high_quality"],
        default="low_latency",
    )

    args = parser.parse_args()

    profile_map = {
        "ultra_low_latency": CameraProfile.ULTRA_LOW_LATENCY,
        "low_latency": CameraProfile.LOW_LATENCY,
        "balanced": CameraProfile.BALANCED,
        "high_quality": CameraProfile.HIGH_QUALITY,
    }
    protocol_map = {
        "udp": StreamProtocol.UDP,
        "websocket": StreamProtocol.WEBSOCKET,
        "both": StreamProtocol.BOTH,
    }

    config = VRStreamerConfig(
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        camera_stereo=args.stereo,
        camera_profile=profile_map[args.camera_profile],
        protocol=protocol_map[args.protocol],
        host=args.host,
        port=args.port,
        target_host=args.target,
        target_port=args.target_port,
        jpeg_quality=args.quality,
        show_stats=not args.no_stats,
        stats_interval=args.stats_interval,
        log_level=args.log_level,
    )

    streamer = VRStreamer(config)

    async def run():
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(streamer.stop()))
        if not await streamer.initialize():
            sys.exit(1)
        try:
            await streamer.start()
        finally:
            await streamer.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
