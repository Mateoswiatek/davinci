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

from vr_streamer import VRStreamer, VRStreamerConfig, StreamProtocol
from camera_capture import CameraProfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
config = VRStreamerConfig(
    camera_width=1280,
    camera_height=720,
    camera_fps=30,
    camera_profile=CameraProfile.LOW_LATENCY,

    protocol=StreamProtocol.WEBSOCKET,
    host="0.0.0.0",
    port=8000,
    jpeg_quality=85,

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
    print(f"\nVR Streaming Server")
    print(f"  Camera:     {config.camera_width}x{config.camera_height} @ {config.camera_fps}fps")
    print(f"  WebSocket:  ws://0.0.0.0:{config.port}")
    print(f"  Web viewer: http://0.0.0.0:{config.port}")
    print(f"  Press Ctrl+C to stop\n")
    asyncio.run(run())
