# VR Streaming System for Raspberry Pi 5

```bash
/home/pi/venv/davinci/bin/python3 /home/pi/Desktop/project/davinci/computing/run_server.py --with-yolo
```

High-performance video streaming system designed for VR applications. Captures video from Picamera2, optionally processes with YOLO object detection, and streams to VR headsets (Oculus Quest) or debug clients.

## Features

- **Ultra-low latency streaming** (<50ms target)
- **Multiple protocols**: UDP (lowest latency), WebSocket (browser compatible)
- **YOLO object detection**: Async processing that doesn't block streaming
- **Stereo camera support**: For side-by-side VR views
- **Web viewer**: Works in Oculus Quest browser
- **Debug client**: Python viewer with stats and recording

## Quick Start

### 1. Install Dependencies

```bash
# On Raspberry Pi 5
pip install picamera2 numpy opencv-python aiohttp

# For YOLO (optional)
pip install ultralytics

# For development (without Pi camera)
pip install pillow
```

### 2. Run the Streamer

```bash
# Basic WebSocket streaming
python vr_streamer.py --protocol websocket --port 8000

# With YOLO detection
python vr_streamer.py --protocol websocket --yolo --yolo-skip 3

# UDP streaming (lowest latency)
python vr_streamer.py --protocol udp --target 192.168.1.100 --port 5000

# Both protocols simultaneously
python vr_streamer.py --protocol both --port 8000 --target 192.168.1.100
```

### 3. View the Stream

**Web Browser (Oculus Quest):**
```
http://<raspberry-pi-ip>:8000
```

**Python Debug Client:**
```bash
python clients/debug_client.py --host <raspberry-pi-ip> --port 8000
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Raspberry Pi 5                              │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Camera     │    │    YOLO      │    │   Streamer   │     │
│  │   Capture    │───>│  (async)     │───>│  (UDP/WS)    │────>│ VR/Debug
│  │  (Picamera2) │    │  (optional)  │    │              │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
computing/
├── camera_capture.py      # Picamera2 capture module
├── yolo_processor.py      # YOLO detection (sync & async)
├── vr_streamer.py         # Main pipeline (entry point)
│
├── protocols/             # Streaming protocols
│   ├── __init__.py
│   ├── base.py            # Abstract base class
│   ├── udp_streamer.py    # UDP streaming
│   └── websocket_streamer.py  # WebSocket streaming
│
├── clients/               # Receiver clients
│   └── debug_client.py    # Python debug viewer
│
├── web/                   # Web interface
│   └── index.html         # VR stream viewer
│
├── docs/                  # Documentation
│   └── ANALYSIS.md        # Detailed analysis & comparison
│
└── README.md              # This file
```

## Configuration

### Camera Options

| Option | Default | Description |
|--------|---------|-------------|
| `--width` | 1280 | Camera width |
| `--height` | 720 | Camera height |
| `--fps` | 30 | Frames per second |
| `--stereo` | false | Enable stereo camera |
| `--camera-profile` | low_latency | Latency profile |

### Streaming Options

| Option | Default | Description |
|--------|---------|-------------|
| `--protocol` | websocket | Protocol: udp, websocket, both |
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8000 | WebSocket port |
| `--target` | | UDP target IP |
| `--target-port` | 5000 | UDP target port |
| `--quality` | 85 | JPEG quality (1-100) |

### YOLO Options

| Option | Default | Description |
|--------|---------|-------------|
| `--yolo` | false | Enable YOLO detection |
| `--yolo-model` | yolov8n.pt | Model path |
| `--yolo-backend` | pytorch | Backend: pytorch, onnx, ncnn |
| `--yolo-skip` | 3 | Process every N-th frame |
| `--yolo-conf` | 0.5 | Confidence threshold |
| `--yolo-sync` | false | Use sync YOLO (not recommended) |

### Performance Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cpu-camera` | | CPU affinity for camera |
| `--cpu-yolo` | | CPU affinity for YOLO |

## Performance

### Expected Latency

| Configuration | Latency | FPS |
|---------------|---------|-----|
| UDP + no YOLO | 25-40ms | 30 |
| UDP + YOLO async | 35-50ms | 30 |
| WebSocket + no YOLO | 50-80ms | 30 |
| WebSocket + YOLO async | 60-100ms | 30 |

### Optimization Tips

1. **Use Ethernet** instead of WiFi for lowest latency
2. **Enable CPU pinning** to reduce context switches
3. **Lower resolution** for higher FPS: 640x480 @ 60fps
4. **Increase YOLO skip** for faster streaming
5. **Use UDP** for minimum latency (LAN only)

## API Reference

### CameraCapture

```python
from camera_capture import CameraCapture, CameraConfig, CameraProfile

config = CameraConfig(
    width=1280,
    height=720,
    fps=30,
    profile=CameraProfile.LOW_LATENCY
)

camera = CameraCapture(config)
camera.initialize()
camera.start()

frame = camera.capture_frame()
print(f"Frame {frame.frame_id}: {frame.width}x{frame.height}")

camera.close()
```

### YOLOProcessor

```python
from yolo_processor import YOLOProcessor, YOLOConfig, AsyncYOLOProcessor

# Synchronous
config = YOLOConfig(
    model_path="yolov8n.pt",
    skip_n_frames=3
)
processor = YOLOProcessor(config)
processor.initialize()
result = processor.process(frame)

# Asynchronous (recommended)
async_processor = AsyncYOLOProcessor(config)
async_processor.start()
async_processor.submit(frame, frame_id=1)
result = async_processor.get_result()
```

### Streaming

```python
from protocols import UDPStreamer, WebSocketStreamer
from protocols.udp_streamer import UDPConfig
from protocols.websocket_streamer import WebSocketConfig

# UDP
udp_config = UDPConfig(target_host="192.168.1.100", target_port=5000)
udp = UDPStreamer(udp_config)
await udp.start()
await udp.send_frame(frame)

# WebSocket
ws_config = WebSocketConfig(host="0.0.0.0", port=8000)
ws = WebSocketStreamer(ws_config)
await ws.start()
await ws.broadcast_frame(frame)
```

## Troubleshooting

### Camera not found

```bash
# Check camera
libcamera-hello --list-cameras

# If using legacy camera stack
sudo raspi-config
# -> Interface Options -> Legacy Camera -> Enable
```

### High latency

1. Check network: `ping <device>`
2. Use wired connection
3. Reduce resolution
4. Increase JPEG compression

### YOLO slow

1. Use `yolov8n` (nano) instead of larger models
2. Increase `--yolo-skip` to 5-10
3. Consider Coral USB accelerator

### WebSocket disconnects

Check firewall:
```bash
sudo ufw allow 8000/tcp
```

## See Also

- [Detailed Analysis](docs/ANALYSIS.md) - Protocol comparison and architecture
- [Picamera2 Documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [Ultralytics YOLO](https://docs.ultralytics.com/)