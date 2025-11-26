"""
WebSocket Streaming Protocol

Browser-compatible streaming using WebSocket + MJPEG/Base64.
Works with any modern web browser including Oculus Quest browser.

Features:
- Browser compatibility
- Multiple client support
- JSON metadata support
- Easy debugging

Author: Claude Code
License: MIT
"""

import asyncio
import json
import base64
import time
import logging
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass
import numpy as np

from .base import StreamingProtocol, StreamConfig, CompressionFormat

logger = logging.getLogger(__name__)

# Optional imports
try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available - install with: pip install aiohttp")

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


@dataclass
class WebSocketConfig(StreamConfig):
    """WebSocket-specific configuration."""
    # Message format
    use_binary: bool = False  # True for binary, False for base64 JSON
    include_detections: bool = True

    # HTTP server
    enable_http: bool = True
    static_path: Optional[str] = None  # Path to serve static files

    # Ping/Pong
    ping_interval: float = 20.0
    ping_timeout: float = 10.0


class WebSocketStreamer(StreamingProtocol):
    """
    WebSocket streaming implementation.

    Sends frames as either:
    - Binary WebSocket messages (lower overhead)
    - Base64-encoded JSON (easier debugging, metadata support)
    """

    def __init__(self, config: Optional[WebSocketConfig] = None):
        super().__init__(config or WebSocketConfig())
        self.config: WebSocketConfig = self.config

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._clients: Set[web.WebSocketResponse] = set()
        self._lock = asyncio.Lock()

    async def start(self) -> bool:
        """Start the WebSocket server."""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available")
            return False

        try:
            self._app = web.Application()

            # Add routes
            self._app.router.add_get('/ws', self._websocket_handler)
            self._app.router.add_get('/stream', self._mjpeg_handler)
            self._app.router.add_get('/health', self._health_handler)

            if self.config.enable_http:
                self._app.router.add_get('/', self._index_handler)

            if self.config.static_path:
                self._app.router.add_static('/static', self.config.static_path)

            # Start server
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()

            self._site = web.TCPSite(
                self._runner,
                self.config.host,
                self.config.port
            )
            await self._site.start()

            self._running = True
            logger.info(f"WebSocket server started on http://{self.config.host}:{self.config.port}")
            logger.info(f"  WebSocket: ws://{self.config.host}:{self.config.port}/ws")
            logger.info(f"  MJPEG: http://{self.config.host}:{self.config.port}/stream")

            return True

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False

    async def stop(self):
        """Stop the WebSocket server."""
        self._running = False

        # Close all client connections
        async with self._lock:
            for ws in self._clients.copy():
                try:
                    await ws.close()
                except:
                    pass
            self._clients.clear()

        # Stop server
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

        logger.info("WebSocket server stopped")

    async def _websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse(
            heartbeat=self.config.ping_interval,
            receive_timeout=self.config.ping_timeout
        )
        await ws.prepare(request)

        client_ip = request.remote
        logger.info(f"WebSocket client connected: {client_ip}")

        async with self._lock:
            self._clients.add(ws)
            self.stats.clients_connected = len(self._clients)

        self._trigger_callbacks('on_client_connect', client_ip)

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle incoming messages (e.g., control commands)
                    try:
                        data = json.loads(msg.data)
                        await self._handle_client_message(ws, data)
                    except json.JSONDecodeError:
                        pass
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            async with self._lock:
                self._clients.discard(ws)
                self.stats.clients_connected = len(self._clients)

            self._trigger_callbacks('on_client_disconnect', client_ip)
            logger.info(f"WebSocket client disconnected: {client_ip}")

        return ws

    async def _handle_client_message(self, ws: web.WebSocketResponse, data: Dict):
        """Handle incoming client messages."""
        msg_type = data.get('type')

        if msg_type == 'ping':
            await ws.send_json({'type': 'pong', 'timestamp': time.time()})

        elif msg_type == 'get_stats':
            stats = self.get_stats()
            await ws.send_json({
                'type': 'stats',
                'data': {
                    'frames_sent': stats.frames_sent,
                    'bytes_sent': stats.bytes_sent,
                    'fps': stats.current_fps,
                    'bandwidth_mbps': stats.bandwidth_mbps,
                    'clients': stats.clients_connected
                }
            })

    async def _mjpeg_handler(self, request: web.Request) -> web.StreamResponse:
        """Handle MJPEG stream requests."""
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
        await response.prepare(request)

        client_ip = request.remote
        logger.info(f"MJPEG client connected: {client_ip}")

        # Add to clients (as a special marker)
        mjpeg_client = {'response': response, 'type': 'mjpeg'}
        async with self._lock:
            self._clients.add(response)

        try:
            while self._running:
                await asyncio.sleep(1)  # Keep connection alive
        except Exception as e:
            logger.error(f"MJPEG handler error: {e}")
        finally:
            async with self._lock:
                self._clients.discard(response)

        return response

    async def _index_handler(self, request: web.Request) -> web.Response:
        """Serve a simple HTML page for testing."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>VR Stream Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        #video { max-width: 100%; border: 2px solid #333; }
        #stats { margin-top: 10px; font-family: monospace; }
        .controls { margin: 10px 0; }
        button { padding: 10px 20px; margin-right: 10px; }
    </style>
</head>
<body>
    <h1>VR Stream Viewer</h1>

    <div class="controls">
        <button onclick="startWebSocket()">WebSocket</button>
        <button onclick="startMJPEG()">MJPEG</button>
        <button onclick="stop()">Stop</button>
    </div>

    <div>
        <img id="video" src="" alt="Video stream">
    </div>

    <div id="stats">
        FPS: <span id="fps">0</span> |
        Frames: <span id="frames">0</span> |
        Latency: <span id="latency">0</span>ms
    </div>

    <script>
        let ws = null;
        let frameCount = 0;
        let lastFrameTime = 0;
        let fpsCounter = 0;

        setInterval(() => {
            document.getElementById('fps').textContent = fpsCounter;
            fpsCounter = 0;
        }, 1000);

        function startWebSocket() {
            stop();
            const host = window.location.host;
            ws = new WebSocket(`ws://${host}/ws`);

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'frame') {
                    document.getElementById('video').src = 'data:image/jpeg;base64,' + data.image;
                    frameCount++;
                    fpsCounter++;
                    document.getElementById('frames').textContent = frameCount;

                    if (data.timestamp) {
                        const latency = Date.now() - (data.timestamp * 1000);
                        document.getElementById('latency').textContent = latency.toFixed(0);
                    }
                }
            };

            ws.onopen = () => console.log('WebSocket connected');
            ws.onclose = () => console.log('WebSocket disconnected');
            ws.onerror = (e) => console.error('WebSocket error:', e);
        }

        function startMJPEG() {
            stop();
            document.getElementById('video').src = '/stream';
        }

        function stop() {
            if (ws) {
                ws.close();
                ws = null;
            }
            document.getElementById('video').src = '';
        }
    </script>
</body>
</html>
        """
        return web.Response(text=html, content_type='text/html')

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        stats = self.get_stats()
        return web.json_response({
            'status': 'ok',
            'running': self._running,
            'clients': stats.clients_connected,
            'frames_sent': stats.frames_sent,
            'fps': round(stats.current_fps, 1)
        })

    async def send_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send frame to a single client (first connected)."""
        async with self._lock:
            if not self._clients:
                return False
            client = next(iter(self._clients))

        return await self._send_to_client(client, frame, frame_id, timestamp, metadata)

    async def broadcast_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Broadcast frame to all connected clients."""
        if not self._running:
            return 0

        start_time = time.perf_counter()
        timestamp = timestamp or time.time()

        # Compress once for all clients
        compressed = self.compress_frame(frame)

        sent_count = 0
        dead_clients = set()

        async with self._lock:
            clients = self._clients.copy()

        for client in clients:
            try:
                if isinstance(client, web.WebSocketResponse):
                    if client.closed:
                        dead_clients.add(client)
                        continue

                    if self.config.use_binary:
                        await client.send_bytes(compressed)
                    else:
                        # Send as base64 JSON
                        message = {
                            'type': 'frame',
                            'frame_id': frame_id,
                            'timestamp': timestamp,
                            'image': base64.b64encode(compressed).decode('utf-8'),
                            'width': frame.shape[1],
                            'height': frame.shape[0]
                        }

                        if metadata and self.config.include_detections:
                            message['metadata'] = metadata

                        await client.send_json(message)

                    sent_count += 1

                elif isinstance(client, web.StreamResponse):
                    # MJPEG client
                    try:
                        await client.write(
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' +
                            compressed +
                            b'\r\n'
                        )
                        sent_count += 1
                    except Exception:
                        dead_clients.add(client)

            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                dead_clients.add(client)

        # Remove dead clients
        if dead_clients:
            async with self._lock:
                for client in dead_clients:
                    self._clients.discard(client)
                self.stats.clients_connected = len(self._clients)

        # Update stats
        send_time_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(len(compressed), send_time_ms)

        return sent_count

    async def _send_to_client(
        self,
        client: web.WebSocketResponse,
        frame: np.ndarray,
        frame_id: int,
        timestamp: Optional[float],
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Send frame to a specific client."""
        if client.closed:
            return False

        timestamp = timestamp or time.time()
        compressed = self.compress_frame(frame)

        try:
            if self.config.use_binary:
                await client.send_bytes(compressed)
            else:
                message = {
                    'type': 'frame',
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'image': base64.b64encode(compressed).decode('utf-8'),
                    'width': frame.shape[1],
                    'height': frame.shape[0]
                }
                if metadata:
                    message['metadata'] = metadata
                await client.send_json(message)

            return True

        except Exception as e:
            logger.error(f"Send error: {e}")
            return False

    def _update_stats(self, bytes_sent: int, send_time_ms: float):
        """Update streaming statistics."""
        self.stats.frames_sent += 1
        self.stats.bytes_sent += bytes_sent
        self.stats.last_frame_time = time.time()

        alpha = 0.1
        self.stats.avg_frame_size_bytes = int(
            alpha * bytes_sent + (1 - alpha) * self.stats.avg_frame_size_bytes
        )
        self.stats.avg_send_time_ms = (
            alpha * send_time_ms + (1 - alpha) * self.stats.avg_send_time_ms
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def send_message(self, message: Dict[str, Any]):
        """Send a custom message to all clients."""
        async with self._lock:
            for client in self._clients:
                if isinstance(client, web.WebSocketResponse) and not client.closed:
                    try:
                        await client.send_json(message)
                    except:
                        pass


# Simple WebSocket client for testing
class WebSocketClient:
    """Simple WebSocket client for receiving frames."""

    def __init__(self, url: str):
        self.url = url
        self._ws = None
        self._running = False
        self.last_frame: Optional[np.ndarray] = None
        self.last_metadata: Optional[Dict] = None
        self.frames_received = 0

    async def connect(self):
        """Connect to the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE and not AIOHTTP_AVAILABLE:
            raise RuntimeError("No WebSocket library available")

        if AIOHTTP_AVAILABLE:
            import aiohttp
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(self.url)
        else:
            import websockets
            self._ws = await websockets.connect(self.url)

        self._running = True
        logger.info(f"Connected to {self.url}")

    async def disconnect(self):
        """Disconnect from the server."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if hasattr(self, '_session'):
            await self._session.close()

    async def receive_frame(self) -> Optional[np.ndarray]:
        """Receive a single frame."""
        if not self._ws:
            return None

        try:
            if AIOHTTP_AVAILABLE:
                msg = await self._ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Binary frame
                    return self._decode_binary(msg.data)
                else:
                    return None
            else:
                msg = await self._ws.recv()
                data = json.loads(msg)

            if data.get('type') == 'frame':
                # Decode base64 image
                image_data = base64.b64decode(data['image'])
                frame = self._decode_image(image_data)

                self.last_frame = frame
                self.last_metadata = data.get('metadata')
                self.frames_received += 1

                return frame

        except Exception as e:
            logger.error(f"Receive error: {e}")

        return None

    def _decode_image(self, data: bytes) -> np.ndarray:
        """Decode image from bytes."""
        try:
            import cv2
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except ImportError:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(data))
            return np.array(img)

    def _decode_binary(self, data: bytes) -> np.ndarray:
        """Decode binary frame."""
        return self._decode_image(data)


# Example usage
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def test_websocket_streamer():
        print("=== WebSocket Streamer Test ===")

        config = WebSocketConfig(
            host="0.0.0.0",
            port=8000,
            compression=CompressionFormat.JPEG,
            jpeg_quality=80
        )

        streamer = WebSocketStreamer(config)

        if await streamer.start():
            print(f"Server running at http://localhost:{config.port}")
            print("Open in browser to test")

            # Create test frames
            for i in range(100):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                # Add text to frame
                try:
                    import cv2
                    cv2.putText(
                        frame,
                        f"Frame {i}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                except:
                    pass

                sent = await streamer.broadcast_frame(frame, frame_id=i)
                print(f"Frame {i}: sent to {sent} clients")

                await asyncio.sleep(0.033)  # ~30 FPS

            stats = streamer.get_stats()
            print(f"\nStats: {stats.frames_sent} frames, {stats.clients_connected} clients")

            await streamer.stop()

    asyncio.run(test_websocket_streamer())