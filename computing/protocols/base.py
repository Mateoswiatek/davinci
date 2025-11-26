"""
Base Streaming Protocol Interface

Abstract base class that all streaming protocols must implement.
Allows easy switching between UDP, WebSocket, WebRTC, etc.

Author: Claude Code
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import numpy as np
import time


class CompressionFormat(Enum):
    """Supported frame compression formats."""
    RAW = "raw"          # No compression (highest bandwidth, lowest latency)
    JPEG = "jpeg"        # JPEG compression (good balance)
    PNG = "png"          # PNG compression (lossless, high bandwidth)
    H264 = "h264"        # H.264 video codec (not recommended for Pi 5)
    WEBP = "webp"        # WebP (good compression, slower)


@dataclass
class StreamConfig:
    """Streaming configuration parameters."""
    # Network
    host: str = "0.0.0.0"
    port: int = 8000
    target_host: str = ""  # For UDP: target IP
    target_port: int = 5000

    # Video
    width: int = 1280
    height: int = 720
    fps: int = 30
    compression: CompressionFormat = CompressionFormat.JPEG
    jpeg_quality: int = 85  # 1-100

    # Buffering
    buffer_size: int = 65536  # Socket buffer size
    max_clients: int = 5

    # Features
    include_timestamp: bool = True
    include_frame_id: bool = True
    enable_stats: bool = True


@dataclass
class StreamStats:
    """Streaming statistics."""
    frames_sent: int = 0
    bytes_sent: int = 0
    clients_connected: int = 0
    avg_frame_size_bytes: int = 0
    avg_send_time_ms: float = 0.0
    current_fps: float = 0.0
    bandwidth_mbps: float = 0.0
    dropped_frames: int = 0
    errors: int = 0

    # Timing
    last_frame_time: float = 0.0
    start_time: float = field(default_factory=time.time)

    def update_fps(self):
        """Update current FPS calculation."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.current_fps = self.frames_sent / elapsed
            self.bandwidth_mbps = (self.bytes_sent * 8) / (elapsed * 1_000_000)


class StreamingProtocol(ABC):
    """
    Abstract base class for streaming protocols.

    All streaming implementations (UDP, WebSocket, WebRTC, etc.)
    must implement this interface.
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.stats = StreamStats()
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = {
            'on_client_connect': [],
            'on_client_disconnect': [],
            'on_error': [],
            'on_frame_sent': []
        }

    @abstractmethod
    async def start(self) -> bool:
        """
        Start the streaming server/sender.

        Returns:
            True if started successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def stop(self):
        """Stop the streaming server/sender."""
        pass

    @abstractmethod
    async def send_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a video frame.

        Args:
            frame: Video frame (numpy array, BGR or RGB)
            frame_id: Frame identifier
            timestamp: Frame timestamp (uses current time if None)
            metadata: Additional metadata (e.g., detections)

        Returns:
            True if sent successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def broadcast_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Broadcast frame to all connected clients.

        Args:
            frame: Video frame
            frame_id: Frame identifier
            timestamp: Frame timestamp
            metadata: Additional metadata

        Returns:
            Number of clients that received the frame.
        """
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the streamer is running."""
        pass

    @property
    @abstractmethod
    def client_count(self) -> int:
        """Get number of connected clients."""
        pass

    def get_stats(self) -> StreamStats:
        """Get streaming statistics."""
        self.stats.update_fps()
        return self.stats

    def reset_stats(self):
        """Reset streaming statistics."""
        self.stats = StreamStats()

    def add_callback(self, event: str, callback: Callable):
        """Add event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: Callable):
        """Remove event callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """Trigger all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass  # Don't let callback errors affect streaming

    def compress_frame(self, frame: np.ndarray) -> bytes:
        """
        Compress frame according to configuration.

        Args:
            frame: Input frame (numpy array)

        Returns:
            Compressed frame bytes
        """
        if self.config.compression == CompressionFormat.RAW:
            return frame.tobytes()

        elif self.config.compression == CompressionFormat.JPEG:
            try:
                import cv2
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                return buffer.tobytes()
            except ImportError:
                # Fallback to PIL
                from PIL import Image
                import io
                img = Image.fromarray(frame)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=self.config.jpeg_quality)
                return buffer.getvalue()

        elif self.config.compression == CompressionFormat.PNG:
            try:
                import cv2
                _, buffer = cv2.imencode('.png', frame)
                return buffer.tobytes()
            except ImportError:
                from PIL import Image
                import io
                img = Image.fromarray(frame)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                return buffer.getvalue()

        elif self.config.compression == CompressionFormat.WEBP:
            try:
                import cv2
                encode_params = [cv2.IMWRITE_WEBP_QUALITY, self.config.jpeg_quality]
                _, buffer = cv2.imencode('.webp', frame, encode_params)
                return buffer.tobytes()
            except ImportError:
                from PIL import Image
                import io
                img = Image.fromarray(frame)
                buffer = io.BytesIO()
                img.save(buffer, format='WEBP', quality=self.config.jpeg_quality)
                return buffer.getvalue()

        else:
            # Default to raw
            return frame.tobytes()

    def create_frame_header(
        self,
        frame_id: int,
        timestamp: float,
        data_size: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Create a frame header for the protocol.

        Header format (32 bytes):
        - Magic (4 bytes): "VR01"
        - Frame ID (4 bytes): uint32
        - Timestamp (8 bytes): float64 (seconds since epoch)
        - Data size (4 bytes): uint32
        - Width (2 bytes): uint16
        - Height (2 bytes): uint16
        - Compression (1 byte): uint8
        - Flags (1 byte): uint8
        - Reserved (6 bytes)

        Args:
            frame_id: Frame identifier
            timestamp: Frame timestamp
            data_size: Size of compressed frame data
            metadata: Optional metadata

        Returns:
            Header bytes
        """
        import struct

        flags = 0
        if metadata:
            flags |= 0x01  # Has metadata

        header = struct.pack(
            '<4sIfI2HBB6x',  # Little-endian
            b'VR01',                                    # Magic
            frame_id,                                   # Frame ID
            timestamp,                                  # Timestamp (float)
            data_size,                                  # Data size
            self.config.width,                          # Width
            self.config.height,                         # Height
            self.config.compression.value.encode()[0],  # Compression type
            flags                                       # Flags
        )

        return header

    def parse_frame_header(self, header: bytes) -> Dict[str, Any]:
        """
        Parse a frame header.

        Args:
            header: Header bytes (32 bytes)

        Returns:
            Parsed header dictionary
        """
        import struct

        if len(header) < 32:
            raise ValueError(f"Header too short: {len(header)} bytes")

        magic, frame_id, timestamp, data_size, width, height, compression, flags = struct.unpack(
            '<4sIfI2HBB6x',
            header[:32]
        )

        if magic != b'VR01':
            raise ValueError(f"Invalid magic: {magic}")

        return {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'data_size': data_size,
            'width': width,
            'height': height,
            'compression': chr(compression),
            'has_metadata': bool(flags & 0x01)
        }


class MultiProtocolStreamer:
    """
    Manager for multiple streaming protocols.

    Allows simultaneous streaming over different protocols
    (e.g., UDP for local VR headset + WebSocket for debug viewer).
    """

    def __init__(self):
        self.protocols: Dict[str, StreamingProtocol] = {}

    def add_protocol(self, name: str, protocol: StreamingProtocol):
        """Add a streaming protocol."""
        self.protocols[name] = protocol

    def remove_protocol(self, name: str):
        """Remove a streaming protocol."""
        if name in self.protocols:
            del self.protocols[name]

    async def start_all(self):
        """Start all protocols."""
        for name, protocol in self.protocols.items():
            await protocol.start()

    async def stop_all(self):
        """Stop all protocols."""
        for protocol in self.protocols.values():
            await protocol.stop()

    async def broadcast_to_all(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Broadcast frame to all protocols.

        Returns:
            Dict mapping protocol name to number of clients reached.
        """
        results = {}
        for name, protocol in self.protocols.items():
            try:
                count = await protocol.broadcast_frame(frame, frame_id, timestamp, metadata)
                results[name] = count
            except Exception:
                results[name] = 0
        return results

    def get_all_stats(self) -> Dict[str, StreamStats]:
        """Get stats for all protocols."""
        return {name: p.get_stats() for name, p in self.protocols.items()}

    def get_total_clients(self) -> int:
        """Get total number of clients across all protocols."""
        return sum(p.client_count for p in self.protocols.values())