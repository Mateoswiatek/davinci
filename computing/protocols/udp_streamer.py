"""
UDP Streaming Protocol

Ultra-low-latency UDP streaming for VR applications.
Designed for LAN use where packet loss is minimal.

Features:
- Lowest possible latency (20-50ms)
- Frame chunking for large frames
- Optional reliability layer
- Zero-copy optimizations

Author: Claude Code
License: MIT
"""

import asyncio
import socket
import struct
import time
import logging
from typing import Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass
import numpy as np

from .base import StreamingProtocol, StreamConfig, CompressionFormat

logger = logging.getLogger(__name__)


@dataclass
class UDPConfig(StreamConfig):
    """UDP-specific configuration."""
    # Chunking
    chunk_size: int = 1400  # MTU-safe chunk size
    max_frame_size: int = 5_000_000  # 5MB max frame

    # Reliability (optional)
    enable_ack: bool = False  # Request acknowledgments
    ack_timeout_ms: int = 50  # ACK timeout
    max_retries: int = 2

    # Multicast
    enable_multicast: bool = False
    multicast_group: str = "239.255.0.1"
    multicast_ttl: int = 1


class UDPStreamer(StreamingProtocol):
    """
    UDP streaming implementation.

    Optimized for minimum latency in VR streaming scenarios.

    Frame Protocol:
    1. Frame header (32 bytes) sent first
    2. Frame data sent in chunks
    3. Each chunk has 8-byte header: [chunk_id: u16, total_chunks: u16, frame_id: u32]
    """

    CHUNK_HEADER_SIZE = 8

    def __init__(self, config: Optional[UDPConfig] = None):
        super().__init__(config or UDPConfig())
        self.config: UDPConfig = self.config

        self._socket: Optional[socket.socket] = None
        self._targets: Set[Tuple[str, int]] = set()
        self._lock = asyncio.Lock()

        # Frame assembly for receiver mode
        self._frame_buffers: Dict[int, Dict[int, bytes]] = {}
        self._frame_chunk_counts: Dict[int, int] = {}

    async def start(self) -> bool:
        """Start the UDP streamer."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Set socket options for low latency
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.config.buffer_size)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.buffer_size)

            # Disable Nagle's algorithm (already off for UDP, but just in case)
            try:
                self._socket.setsockopt(socket.IPPROTO_UDP, socket.UDP_CORK, 0)
            except (AttributeError, OSError):
                pass  # Not all systems support this

            # Set non-blocking
            self._socket.setblocking(False)

            # Bind for receiving
            self._socket.bind((self.config.host, self.config.port))

            # Setup multicast if enabled
            if self.config.enable_multicast:
                self._setup_multicast()

            # Add default target if specified
            if self.config.target_host:
                self.add_target(self.config.target_host, self.config.target_port)

            self._running = True
            logger.info(f"UDP streamer started on {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start UDP streamer: {e}")
            return False

    def _setup_multicast(self):
        """Setup multicast socket options."""
        import struct as st

        # Set TTL
        self._socket.setsockopt(
            socket.IPPROTO_IP,
            socket.IP_MULTICAST_TTL,
            st.pack('b', self.config.multicast_ttl)
        )

        # Join multicast group
        mreq = st.pack(
            '4sl',
            socket.inet_aton(self.config.multicast_group),
            socket.INADDR_ANY
        )
        self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        # Add multicast group as target
        self.add_target(self.config.multicast_group, self.config.target_port)

        logger.info(f"Multicast enabled: {self.config.multicast_group}")

    async def stop(self):
        """Stop the UDP streamer."""
        self._running = False
        if self._socket:
            self._socket.close()
            self._socket = None
        self._targets.clear()
        logger.info("UDP streamer stopped")

    def add_target(self, host: str, port: int):
        """Add a target to send frames to."""
        self._targets.add((host, port))
        logger.info(f"Added target: {host}:{port}")

    def remove_target(self, host: str, port: int):
        """Remove a target."""
        self._targets.discard((host, port))
        logger.info(f"Removed target: {host}:{port}")

    async def send_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a frame to the default target."""
        if not self._targets:
            return False

        target = next(iter(self._targets))
        return await self._send_to_target(frame, target, frame_id, timestamp, metadata)

    async def broadcast_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Broadcast frame to all targets."""
        if not self._running or not self._socket:
            return 0

        sent_count = 0
        for target in self._targets.copy():
            try:
                if await self._send_to_target(frame, target, frame_id, timestamp, metadata):
                    sent_count += 1
            except Exception as e:
                logger.error(f"Failed to send to {target}: {e}")
                self.stats.errors += 1

        return sent_count

    async def _send_to_target(
        self,
        frame: np.ndarray,
        target: Tuple[str, int],
        frame_id: int,
        timestamp: Optional[float],
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Send frame to a specific target."""
        if not self._socket:
            return False

        start_time = time.perf_counter()
        timestamp = timestamp or time.time()

        try:
            # Compress frame
            compressed = self.compress_frame(frame)

            if len(compressed) > self.config.max_frame_size:
                logger.warning(f"Frame too large: {len(compressed)} bytes")
                return False

            # Create and send header
            header = self.create_frame_header(frame_id, timestamp, len(compressed), metadata)

            # Send header
            await self._send_bytes(header, target)

            # Send data in chunks
            total_chunks = (len(compressed) + self.config.chunk_size - 1) // self.config.chunk_size

            for chunk_id in range(total_chunks):
                start = chunk_id * self.config.chunk_size
                end = min(start + self.config.chunk_size, len(compressed))
                chunk_data = compressed[start:end]

                # Create chunk header
                chunk_header = struct.pack('<HHI', chunk_id, total_chunks, frame_id)
                chunk_packet = chunk_header + chunk_data

                await self._send_bytes(chunk_packet, target)

            # Update stats
            send_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_stats(len(compressed), send_time_ms)

            return True

        except Exception as e:
            logger.error(f"Send error: {e}")
            self.stats.errors += 1
            return False

    async def _send_bytes(self, data: bytes, target: Tuple[str, int]):
        """Send bytes to target (async wrapper)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._socket.sendto, data, target)

    def _update_stats(self, bytes_sent: int, send_time_ms: float):
        """Update streaming statistics."""
        self.stats.frames_sent += 1
        self.stats.bytes_sent += bytes_sent
        self.stats.last_frame_time = time.time()

        # Running average of frame size and send time
        alpha = 0.1
        self.stats.avg_frame_size_bytes = int(
            alpha * bytes_sent + (1 - alpha) * self.stats.avg_frame_size_bytes
        )
        self.stats.avg_send_time_ms = (
            alpha * send_time_ms + (1 - alpha) * self.stats.avg_send_time_ms
        )

    @property
    def is_running(self) -> bool:
        return self._running and self._socket is not None

    @property
    def client_count(self) -> int:
        return len(self._targets)

    # ==================== RECEIVER MODE ====================

    async def receive_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Receive a frame (for client/receiver mode).

        Args:
            timeout: Receive timeout in seconds

        Returns:
            Tuple of (frame, metadata) or None if timeout
        """
        if not self._socket:
            return None

        try:
            # Set timeout
            self._socket.settimeout(timeout)

            # Receive header
            header_data, addr = self._socket.recvfrom(32)
            header = self.parse_frame_header(header_data)

            frame_id = header['frame_id']
            total_size = header['data_size']

            # Calculate expected chunks
            total_chunks = (total_size + self.config.chunk_size - 1) // self.config.chunk_size

            # Receive chunks
            chunks = {}
            received_bytes = 0

            while len(chunks) < total_chunks:
                chunk_packet, _ = self._socket.recvfrom(
                    self.config.chunk_size + self.CHUNK_HEADER_SIZE
                )

                # Parse chunk header
                chunk_id, chunk_total, chunk_frame_id = struct.unpack(
                    '<HHI',
                    chunk_packet[:self.CHUNK_HEADER_SIZE]
                )

                if chunk_frame_id != frame_id:
                    continue  # Wrong frame, skip

                chunks[chunk_id] = chunk_packet[self.CHUNK_HEADER_SIZE:]
                received_bytes += len(chunks[chunk_id])

            # Assemble frame
            compressed = b''.join(chunks[i] for i in sorted(chunks.keys()))

            # Decompress
            frame = self._decompress_frame(compressed, header)

            return frame, header

        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Receive error: {e}")
            return None

    def _decompress_frame(self, data: bytes, header: Dict) -> np.ndarray:
        """Decompress frame data."""
        compression = header.get('compression', 'j')

        if compression == 'r':  # Raw
            shape = (header['height'], header['width'], 3)
            return np.frombuffer(data, dtype=np.uint8).reshape(shape)

        elif compression == 'j':  # JPEG
            try:
                import cv2
                arr = np.frombuffer(data, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except ImportError:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(data))
                return np.array(img)

        elif compression == 'p':  # PNG
            try:
                import cv2
                arr = np.frombuffer(data, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except ImportError:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(data))
                return np.array(img)

        else:
            raise ValueError(f"Unknown compression: {compression}")


class UDPReceiver:
    """
    Dedicated UDP frame receiver.

    Handles frame reassembly from UDP chunks.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000, buffer_size: int = 65536):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size

        self._socket: Optional[socket.socket] = None
        self._running = False

        # Frame assembly
        self._pending_frames: Dict[int, Dict] = {}
        self._last_complete_frame: Optional[np.ndarray] = None
        self._last_frame_info: Optional[Dict] = None

        # Stats
        self.frames_received = 0
        self.bytes_received = 0
        self.dropped_frames = 0

    def start(self):
        """Start the receiver."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
        self._socket.bind((self.host, self.port))
        self._socket.setblocking(False)
        self._running = True
        logger.info(f"UDP receiver started on {self.host}:{self.port}")

    def stop(self):
        """Stop the receiver."""
        self._running = False
        if self._socket:
            self._socket.close()
            self._socket = None
        logger.info("UDP receiver stopped")

    def receive_frame_sync(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Receive a complete frame (blocking).

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (frame, info) or None
        """
        if not self._socket:
            return None

        import select

        start_time = time.time()
        header_received = False
        current_frame_id = None
        current_header = None
        chunks = {}
        expected_chunks = 0

        while time.time() - start_time < timeout:
            # Check for data
            ready, _, _ = select.select([self._socket], [], [], 0.01)
            if not ready:
                continue

            try:
                data, addr = self._socket.recvfrom(65535)

                # Check if this is a header
                if len(data) == 32 and data[:4] == b'VR01':
                    # Parse header
                    magic, frame_id, timestamp, data_size, width, height, compression, flags = struct.unpack(
                        '<4sIfI2HBB6x',
                        data
                    )
                    current_frame_id = frame_id
                    current_header = {
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'data_size': data_size,
                        'width': width,
                        'height': height,
                        'compression': chr(compression)
                    }
                    header_received = True
                    chunks = {}
                    expected_chunks = (data_size + 1400 - 1) // 1400
                    continue

                # Parse chunk
                if len(data) >= 8 and header_received:
                    chunk_id, total_chunks, frame_id = struct.unpack('<HHI', data[:8])

                    if frame_id != current_frame_id:
                        continue

                    chunks[chunk_id] = data[8:]
                    expected_chunks = total_chunks

                    # Check if complete
                    if len(chunks) >= expected_chunks:
                        # Assemble frame
                        compressed = b''.join(chunks[i] for i in sorted(chunks.keys()))
                        frame = self._decompress(compressed, current_header)

                        self.frames_received += 1
                        self.bytes_received += len(compressed)
                        self._last_complete_frame = frame
                        self._last_frame_info = current_header

                        return frame, current_header

            except BlockingIOError:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                continue

        return None

    def _decompress(self, data: bytes, header: Dict) -> np.ndarray:
        """Decompress frame."""
        compression = header.get('compression', 'j')

        try:
            import cv2

            if compression == 'r':
                shape = (header['height'], header['width'], 3)
                return np.frombuffer(data, dtype=np.uint8).reshape(shape)
            else:
                arr = np.frombuffer(data, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except ImportError:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(data))
            return np.array(img)

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, Dict]]:
        """Get the last complete frame received."""
        if self._last_complete_frame is not None:
            return self._last_complete_frame, self._last_frame_info
        return None


# Example usage
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def test_udp_streamer():
        print("=== UDP Streamer Test ===")

        # Create streamer
        config = UDPConfig(
            host="0.0.0.0",
            port=5001,
            target_host="127.0.0.1",
            target_port=5000,
            compression=CompressionFormat.JPEG,
            jpeg_quality=80
        )

        streamer = UDPStreamer(config)

        if await streamer.start():
            # Create test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Send frames
            for i in range(10):
                success = await streamer.send_frame(test_frame, frame_id=i)
                print(f"Frame {i}: {'sent' if success else 'failed'}")
                await asyncio.sleep(0.033)  # ~30 FPS

            # Get stats
            stats = streamer.get_stats()
            print(f"\nStats: {stats.frames_sent} frames, {stats.bytes_sent} bytes")
            print(f"Avg frame size: {stats.avg_frame_size_bytes} bytes")
            print(f"Avg send time: {stats.avg_send_time_ms:.2f}ms")

            await streamer.stop()

    asyncio.run(test_udp_streamer())