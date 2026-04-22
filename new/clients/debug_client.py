#!/usr/bin/env python3
"""
Debug Client for VR Streamer

Python client for receiving and displaying VR streams.
Supports both UDP and WebSocket protocols.

Features:
- Real-time frame display with OpenCV
- Latency measurement
- FPS counter
- Detection overlay
- Frame recording

Author: Claude Code
License: MIT

Usage:
    python debug_client.py --help
    python debug_client.py --protocol websocket --host 192.168.1.100 --port 8000
    python debug_client.py --protocol udp --port 5000
    python debug_client.py --record output.mp4
"""

import argparse
import asyncio
import time
import sys
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from collections import deque
from enum import Enum

import numpy as np

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

logger = logging.getLogger(__name__)


class ClientProtocol(Enum):
    UDP = "udp"
    WEBSOCKET = "websocket"


@dataclass
class ClientConfig:
    """Debug client configuration."""
    protocol: ClientProtocol = ClientProtocol.WEBSOCKET
    host: str = "localhost"
    port: int = 8000
    udp_port: int = 5000

    # Display
    window_name: str = "VR Debug Client"
    fullscreen: bool = False
    scale: float = 1.0

    # Recording
    record: bool = False
    record_path: str = "output.mp4"
    record_fps: int = 30

    # Debug
    show_latency: bool = True
    show_fps: bool = True
    show_detections: bool = True
    verbose: bool = False


class DebugClient:
    """
    Debug client for viewing VR streams.

    Provides real-time visualization with performance metrics.
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self._running = False

        # Stats
        self._frame_count = 0
        self._fps_times = deque(maxlen=30)
        self._latencies = deque(maxlen=100)
        self._last_frame_time = 0

        # Recording
        self._video_writer: Optional[cv2.VideoWriter] = None

        # Current frame
        self.last_frame: Optional[np.ndarray] = None
        self.last_metadata: Optional[Dict] = None

    async def start(self):
        """Start the debug client."""
        self._running = True

        # Create display window
        if CV2_AVAILABLE:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
            if self.config.fullscreen:
                cv2.setWindowProperty(
                    self.config.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )

        logger.info(f"Debug client starting ({self.config.protocol.value})...")

        # Start receiving based on protocol
        if self.config.protocol == ClientProtocol.WEBSOCKET:
            await self._websocket_loop()
        else:
            await self._udp_loop()

    async def _websocket_loop(self):
        """WebSocket receive loop."""
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp not available. Install with: pip install aiohttp")
            return

        url = f"ws://{self.config.host}:{self.config.port}/ws"
        logger.info(f"Connecting to {url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url) as ws:
                    logger.info("Connected!")

                    async for msg in ws:
                        if not self._running:
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            import json
                            data = json.loads(msg.data)
                            await self._handle_message(data)

                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await self._handle_binary(msg.data)

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break

        except aiohttp.ClientError as e:
            logger.error(f"Connection failed: {e}")
        except asyncio.CancelledError:
            pass

    async def _udp_loop(self):
        """UDP receive loop."""
        # Import UDP receiver
        sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
        from protocols.udp_streamer import UDPReceiver

        receiver = UDPReceiver(
            host="0.0.0.0",
            port=self.config.udp_port
        )

        try:
            receiver.start()
            logger.info(f"Listening for UDP on port {self.config.udp_port}")

            while self._running:
                result = receiver.receive_frame_sync(timeout=0.5)

                if result:
                    frame, info = result
                    await self._process_frame(frame, info)

                # Handle keyboard
                if CV2_AVAILABLE:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self._running = False
                        break

        finally:
            receiver.stop()

    async def _handle_message(self, data: Dict):
        """Handle WebSocket JSON message."""
        msg_type = data.get('type')

        if msg_type == 'frame':
            # Decode base64 image
            import base64
            image_data = base64.b64decode(data['image'])
            frame = self._decode_image(image_data)

            info = {
                'frame_id': data.get('frame_id', 0),
                'timestamp': data.get('timestamp', 0),
                'width': data.get('width', 0),
                'height': data.get('height', 0)
            }

            metadata = data.get('metadata')
            await self._process_frame(frame, info, metadata)

    async def _handle_binary(self, data: bytes):
        """Handle binary WebSocket message."""
        frame = self._decode_image(data)
        await self._process_frame(frame, {})

    def _decode_image(self, data: bytes) -> np.ndarray:
        """Decode image from bytes."""
        if CV2_AVAILABLE:
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(data))
            return np.array(img)

    async def _process_frame(
        self,
        frame: np.ndarray,
        info: Dict,
        metadata: Optional[Dict] = None
    ):
        """Process and display a received frame."""
        now = time.time()
        self._frame_count += 1
        self._fps_times.append(now)

        # Calculate latency
        timestamp = info.get('timestamp', 0)
        if timestamp > 0:
            latency_ms = (now - timestamp) * 1000
            self._latencies.append(latency_ms)

        # Store frame
        self.last_frame = frame
        self.last_metadata = metadata

        # Draw overlays
        display_frame = frame.copy()

        if self.config.show_detections and metadata:
            display_frame = self._draw_detections(display_frame, metadata)

        if self.config.show_fps or self.config.show_latency:
            display_frame = self._draw_stats(display_frame, info)

        # Scale if needed
        if self.config.scale != 1.0:
            h, w = display_frame.shape[:2]
            new_size = (int(w * self.config.scale), int(h * self.config.scale))
            display_frame = cv2.resize(display_frame, new_size)

        # Display
        if CV2_AVAILABLE:
            cv2.imshow(self.config.window_name, display_frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._running = False
            elif key == ord('s'):
                # Save screenshot
                filename = f"screenshot_{int(now)}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Saved screenshot: {filename}")
            elif key == ord('r'):
                # Toggle recording
                self._toggle_recording(frame.shape)

        # Recording
        if self._video_writer:
            self._video_writer.write(frame)

        self._last_frame_time = now

    def _draw_detections(self, frame: np.ndarray, metadata: Dict) -> np.ndarray:
        """Draw detection boxes on frame."""
        if not CV2_AVAILABLE:
            return frame

        detections = metadata.get('detections', [])

        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bbox)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{det.get('class_name', 'object')}: {det.get('confidence', 0):.2f}"
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )

        return frame

    def _draw_stats(self, frame: np.ndarray, info: Dict) -> np.ndarray:
        """Draw statistics overlay."""
        if not CV2_AVAILABLE:
            return frame

        y = 30
        color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # FPS
        if self.config.show_fps and len(self._fps_times) >= 2:
            time_span = self._fps_times[-1] - self._fps_times[0]
            if time_span > 0:
                fps = len(self._fps_times) / time_span
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), font, 0.7, color, 2)
                y += 25

        # Latency
        if self.config.show_latency and self._latencies:
            avg_latency = sum(self._latencies) / len(self._latencies)
            cv2.putText(frame, f"Latency: {avg_latency:.1f}ms", (10, y), font, 0.7, color, 2)
            y += 25

        # Frame info
        frame_id = info.get('frame_id', 0)
        cv2.putText(frame, f"Frame: {frame_id}", (10, y), font, 0.7, color, 2)
        y += 25

        # Recording indicator
        if self._video_writer:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 70, 35), font, 0.5, (0, 0, 255), 2)

        # Detection count
        if self.last_metadata:
            det_count = len(self.last_metadata.get('detections', []))
            cv2.putText(frame, f"Objects: {det_count}", (10, y), font, 0.7, color, 2)

        return frame

    def _toggle_recording(self, shape: Tuple[int, int, int]):
        """Toggle video recording."""
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
            logger.info("Recording stopped")
        else:
            h, w = shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._video_writer = cv2.VideoWriter(
                self.config.record_path,
                fourcc,
                self.config.record_fps,
                (w, h)
            )
            logger.info(f"Recording to {self.config.record_path}")

    async def stop(self):
        """Stop the debug client."""
        self._running = False

        if self._video_writer:
            self._video_writer.release()

        if CV2_AVAILABLE:
            cv2.destroyAllWindows()

        logger.info("Debug client stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        fps = 0
        if len(self._fps_times) >= 2:
            time_span = self._fps_times[-1] - self._fps_times[0]
            if time_span > 0:
                fps = len(self._fps_times) / time_span

        avg_latency = 0
        if self._latencies:
            avg_latency = sum(self._latencies) / len(self._latencies)

        return {
            'frames_received': self._frame_count,
            'fps': fps,
            'avg_latency_ms': avg_latency,
            'min_latency_ms': min(self._latencies) if self._latencies else 0,
            'max_latency_ms': max(self._latencies) if self._latencies else 0
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Debug Client for VR Streamer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Connection
    parser.add_argument(
        "--protocol",
        choices=["websocket", "udp"],
        default="websocket",
        help="Connection protocol"
    )
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket port")
    parser.add_argument("--udp-port", type=int, default=5000, help="UDP port")

    # Display
    parser.add_argument("--fullscreen", action="store_true", help="Fullscreen mode")
    parser.add_argument("--scale", type=float, default=1.0, help="Display scale factor")
    parser.add_argument("--no-latency", action="store_true", help="Hide latency display")
    parser.add_argument("--no-fps", action="store_true", help="Hide FPS display")
    parser.add_argument("--no-detections", action="store_true", help="Hide detections")

    # Recording
    parser.add_argument("--record", metavar="FILE", help="Record to file")

    # Debug
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Create config
    config = ClientConfig(
        protocol=ClientProtocol(args.protocol),
        host=args.host,
        port=args.port,
        udp_port=args.udp_port,
        fullscreen=args.fullscreen,
        scale=args.scale,
        show_latency=not args.no_latency,
        show_fps=not args.no_fps,
        show_detections=not args.no_detections,
        record=bool(args.record),
        record_path=args.record or "output.mp4",
        verbose=args.verbose
    )

    # Create and start client
    client = DebugClient(config)

    try:
        await client.start()
    except KeyboardInterrupt:
        pass
    finally:
        await client.stop()

        # Print final stats
        stats = client.get_stats()
        print("\n=== Session Statistics ===")
        print(f"Frames received: {stats['frames_received']}")
        print(f"Average FPS: {stats['fps']:.1f}")
        print(f"Average latency: {stats['avg_latency_ms']:.1f}ms")
        print(f"Min/Max latency: {stats['min_latency_ms']:.1f}ms / {stats['max_latency_ms']:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())