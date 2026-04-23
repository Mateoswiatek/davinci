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

try:
    from servo_manager import ServoConfig, ServoManager, ConnectionManager
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False

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

    # Web viewer — path to web/ directory (serves index.html at /)
    web_dir: Optional[str] = None

    # SSL — set both to enable HTTPS/WSS (required for WebXR on Oculus)
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    # Servo control
    servo_enabled: bool = False
    servo_pan_pin: int = 14
    servo_tilt_pin: int = 15
    servo_roll_pin: int = 18
    servo_step: float = 0.25        # Quantization step in degrees
    angle_send_hz: float = 20.0     # Head angle update rate from client
    send_servo_position: bool = True  # Broadcast servo position in frame metadata
    servo_initial_pan: float = 0.0   # Startup position in degrees
    servo_initial_tilt: float = 0.0
    servo_initial_roll: float = 0.0

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
        self._ws_streamer: Optional[WebSocketStreamer] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._stats = {
            'frames_captured': 0,
            'frames_streamed': 0,
            'avg_capture_ms': 0.0,
            'avg_stream_ms': 0.0,
            'avg_total_ms': 0.0,
            'clients': 0,
            'angles_received': 0,
        }
        self.servo_manager: Optional[ServoManager] = None
        self.connection_manager: Optional[ConnectionManager] = None
        self._ws_map: Dict[str, Any] = {}  # ws_id → WebSocketResponse
        self._first_angle_logged = False

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
                static_path=self.config.web_dir,
                ssl_certfile=self.config.ssl_certfile,
                ssl_keyfile=self.config.ssl_keyfile,
            )
            ws_streamer = WebSocketStreamer(ws_config)

            if self.config.servo_enabled:
                if not SERVO_AVAILABLE:
                    logger.error("servo_enabled=True but servo_manager.py not found")
                    return False
                servo_cfg = ServoConfig(
                    pan_pin=self.config.servo_pan_pin,
                    tilt_pin=self.config.servo_tilt_pin,
                    roll_pin=self.config.servo_roll_pin,
                    step=self.config.servo_step,
                    initial_pan=self.config.servo_initial_pan,
                    initial_tilt=self.config.servo_initial_tilt,
                    initial_roll=self.config.servo_initial_roll,
                )
                self.servo_manager = ServoManager(servo_cfg)
                self.servo_manager.initialize()
                self.connection_manager = ConnectionManager()
                ws_streamer.connect_handler = self._on_ws_connect
                ws_streamer.disconnect_handler = self._on_ws_disconnect
                ws_streamer.message_handler = self._on_ws_message
                logger.info(
                    f"Servo control enabled — step={self.config.servo_step}°, "
                    f"angle_hz={self.config.angle_send_hz}"
                )

            if await ws_streamer.start():
                self.streamer.add_protocol("websocket", ws_streamer)
                self._ws_streamer = ws_streamer
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

                metadata = None
                if self.servo_manager and self.config.send_servo_position:
                    metadata = {'servo': self.servo_manager.get_angles()}

                stream_start = time.perf_counter()
                await self.streamer.broadcast_to_all(
                    frame.data,
                    frame_id=frame.frame_id,
                    timestamp=time.time(),
                    metadata=metadata,
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
            angles = s['angles_received']
            self._stats['frames_captured'] = 0
            self._stats['angles_received'] = 0
            servo_part = f" | Angles/s: {angles / self.config.stats_interval:.1f}" if self.config.servo_enabled else ""
            logger.info(
                f"FPS: {fps:.1f} | Capture: {s['avg_capture_ms']:.1f}ms | "
                f"Stream: {s['avg_stream_ms']:.1f}ms | Total: {s['avg_total_ms']:.1f}ms | "
                f"Clients: {s['clients']}{servo_part}"
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

        if self.servo_manager:
            self.servo_manager.close()

        if self.camera:
            self.camera.close()

        logger.info("Stopped")

    # ── Servo / role event handlers ───────────────────────────────────────────

    async def _on_ws_connect(self, ws_id: str, ws):
        self._ws_map[ws_id] = ws
        if not self.connection_manager or not self._ws_streamer:
            return
        role = await self.connection_manager.on_connect(ws_id)
        msg = {
            'type': 'role',
            'role': role,
            'controller_available': (
                role == 'observer' and self.connection_manager.controller_available()
            ),
            'angle_send_hz': self.config.angle_send_hz,
            'servo_step': self.config.servo_step,
        }
        if self.servo_manager and self.config.send_servo_position:
            msg['servo_position'] = self.servo_manager.get_angles()
        await self._ws_streamer.send_to_ws(ws, msg)

    async def _on_ws_disconnect(self, ws_id: str):
        self._ws_map.pop(ws_id, None)
        if not self.connection_manager or not self._ws_streamer:
            return
        was_controller = await self.connection_manager.on_disconnect(ws_id)
        if was_controller:
            await self._ws_streamer.send_message({
                'type': 'role_update',
                'controller_available': True,
            })

    async def _on_ws_message(self, ws_id: str, data: dict):
        msg_type = data.get('type')
        logger.debug(f"WS msg from {ws_id[:8]}: type={msg_type}")

        if msg_type == 'head_angles':
            if not self.servo_manager:
                logger.warning("head_angles received but servo_manager is None (servo_enabled=False?)")
                return
            if not (self.connection_manager and self.connection_manager.is_controller(ws_id)):
                logger.debug(f"head_angles ignored — sender {ws_id[:8]} is not controller")
                return

            pitch = data.get('pitch')
            yaw   = data.get('yaw')
            roll  = data.get('roll')

            if not self._first_angle_logged:
                logger.info(f"First head_angles received: pitch={pitch} yaw={yaw} roll={roll}")
                self._first_angle_logged = True

            self._stats['angles_received'] += 1
            self.servo_manager.move(pitch=pitch, yaw=yaw, roll=roll)

        elif msg_type == 'take_control' and self.connection_manager:
            success = await self.connection_manager.request_control(ws_id)
            ws = self._ws_map.get(ws_id)
            if ws and self._ws_streamer:
                await self._ws_streamer.send_to_ws(ws, {
                    'type': 'control_result',
                    'success': success,
                    'role': self.connection_manager.get_role(ws_id),
                })

        elif msg_type == 'release_control' and self.connection_manager:
            await self.connection_manager.release_control(ws_id)
            ws = self._ws_map.get(ws_id)
            if ws and self._ws_streamer:
                await self._ws_streamer.send_to_ws(ws, {
                    'type': 'control_result',
                    'success': True,
                    'role': 'observer',
                })
            if self._ws_streamer:
                await self._ws_streamer.send_message({
                    'type': 'role_update',
                    'controller_available': True,
                })

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
