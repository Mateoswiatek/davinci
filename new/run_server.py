#!/usr/bin/env python3
"""
VR Streaming Server — Quick Launcher

Default configuration for the DaVinci VR project.
Edit the CONFIG section below to change settings permanently.

Usage:
    python run_server.py
"""

import asyncio
import signal
import logging
import os

from vr_streamer import VRStreamer, VRStreamerConfig, StreamProtocol
from camera_capture import CameraProfile

_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
config = VRStreamerConfig(
    camera_width=1280, # 720, #1280,
    camera_height=720, # 480, # 720,
    camera_fps=30,
    camera_profile=CameraProfile.LOW_LATENCY,

    web_dir=_WEB_DIR,

    protocol=StreamProtocol.WEBSOCKET,
    host="0.0.0.0",
    port=8000,
    jpeg_quality=85,

    # SSL — required for WebXR on Oculus Quest (wss:// instead of ws://)
    # Generate once on RPi:
    #   openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=raspberrypi.local"
    # Then accept the cert warning once in Oculus browser at https://<RPi-IP>:8000
    ssl_certfile="/home/pi/certs/cert.pem",
    ssl_keyfile="/home/pi/certs/key.pem",

    # Servo control — set servo_enabled=True when servos are wired to GPIO
    servo_enabled=True,
    servo_pan_pin=16,   # GPIO BCM: pan  = yaw
    servo_tilt_pin=20,  # GPIO BCM: tilt = pitch
    servo_roll_pin=21,  # GPIO BCM: roll
    servo_step=0.25,          # Quantization step in degrees
    angle_send_hz=20.0,       # Head angle update rate from Oculus (Hz)
    send_servo_position=True,  # Include servo angles in each frame message

    show_stats=True,
    stats_interval=5.0,
    log_level="INFO",
)
# ─────────────────────────────────────────────────────────────────────────────


async def run():
    streamer = VRStreamer(config)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(streamer.stop()))

    if not await streamer.initialize():
        print("Failed to initialize streamer!")
        return

    try:
        await streamer.start()
    finally:
        await streamer.stop()


if __name__ == "__main__":
    ssl_enabled = bool(config.ssl_certfile and config.ssl_keyfile)
    scheme = "https" if ssl_enabled else "http"
    ws_scheme = "wss" if ssl_enabled else "ws"
    print(f"\nVR Streaming Server")
    print(f"  Camera:     {config.camera_width}x{config.camera_height} @ {config.camera_fps}fps")
    print(f"  WebSocket:  {ws_scheme}://0.0.0.0:{config.port}/ws")
    print(f"  Web viewer: {scheme}://0.0.0.0:{config.port}")
    if ssl_enabled:
        print(f"  SSL:        enabled ({config.ssl_certfile})")
    else:
        print(f"  SSL:        disabled  (set ssl_certfile/ssl_keyfile for Oculus WebXR)")
    print(f"  Press Ctrl+C to stop\n")
    asyncio.run(run())
