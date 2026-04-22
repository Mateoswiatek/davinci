"""
Streaming Protocols Package

Provides multiple streaming protocol implementations:
- UDP: Lowest latency, LAN only
- WebSocket + MJPEG: Browser compatible
- WebRTC: NAT traversal, works over Internet

All protocols implement the StreamingProtocol base class.
"""

from .base import StreamingProtocol, StreamConfig, StreamStats, MultiProtocolStreamer
from .udp_streamer import UDPStreamer
from .websocket_streamer import WebSocketStreamer

__all__ = [
    'StreamingProtocol',
    'StreamConfig',
    'StreamStats',
    'MultiProtocolStreamer',
    'UDPStreamer',
    'WebSocketStreamer',
]