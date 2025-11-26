# VR Streaming System - Quick Start Guide

Kompletny system ultra-low-latency streamingu dla VR na Raspberry Pi 5.

## Zawartość Dokumentacji

1. **picamera2_vr_analysis.md** - Szczegółowa analiza Picamera2 i porównanie metod streamingu
2. **streaming_comparison.md** - Testy wydajnościowe i troubleshooting
3. Ten README - Szybki start

## Pliki Implementacyjne

```
backend/
├── vr_udp_streamer.py      # Server - Raw UDP streaming (20-30ms latency)
├── vr_udp_receiver.py      # Client - Odbiornik UDP
└── vr_yolo_streamer.py     # Server - UDP + YOLO detection (40-50ms)
```

---

## Szybki Start (5 minut)

### 1. Instalacja Zależności

**Na Raspberry Pi 5:**
```bash
# System packages
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy

# Python packages
pip3 install numpy opencv-python pillow

# Opcjonalnie - YOLO
pip3 install ultralytics torch torchvision
```

**Na PC/VR Client:**
```bash
pip install numpy opencv-python pillow
```

---

### 2. Test Sieci

```bash
# Na Pi
hostname -I  # Sprawdź IP

# Na PC
ping <pi-ip>  # Sprawdź connectivity

# Test bandwidth (opcjonalnie)
# Na Pi:
iperf3 -s

# Na PC:
iperf3 -c <pi-ip> -t 10
# Powinno pokazać >600 Mbps dla Gigabit Ethernet
```

---

### 3. Uruchom Streaming

#### Opcja A: Basic Raw UDP (Najszybszy - 20ms)

**Server (Pi):**
```bash
cd /mnt/adata-disk/projects/agh/davinci/davinci/backend
python3 vr_udp_streamer.py --host <pc-ip> --width 2560 --height 800 --fps 30
```

**Client (PC):**
```bash
python3 vr_udp_receiver.py --port 5000
```

**Spodziewane:**
- Okno OpenCV z live feed
- FPS: ~30
- Latencja: ~20-30ms

---

#### Opcja B: Z YOLO Detection (40ms)

**Server (Pi):**
```bash
python3 vr_yolo_streamer.py \
    --host <pc-ip> \
    --width 1280 \
    --height 400 \
    --fps 30 \
    --model yolov8n.pt
```

**Client (PC):**
```bash
python3 vr_udp_receiver.py --port 5000
```

**Spodziewane:**
- Live feed z bounding boxes
- FPS: ~30
- Latencja: ~40-50ms

---

## Parametry Konfiguracyjne

### vr_udp_streamer.py

```bash
python3 vr_udp_streamer.py [OPTIONS]

Options:
  --host TEXT        IP VR headset/PC (default: 192.168.1.100)
  --port INTEGER     UDP port (default: 5000)
  --width INTEGER    Frame width (default: 2560)
  --height INTEGER   Frame height (default: 800)
  --fps INTEGER      Target FPS (default: 30)
```

**Przykłady:**

```bash
# Full resolution, 30 FPS
python3 vr_udp_streamer.py --host 192.168.1.100 --width 2560 --height 800 --fps 30

# Half resolution dla mniejszego bandwidth
python3 vr_udp_streamer.py --host 192.168.1.100 --width 1280 --height 400 --fps 30

# Higher FPS (wymaga więcej bandwidth)
python3 vr_udp_streamer.py --host 192.168.1.100 --width 1280 --height 400 --fps 60
```

---

### vr_yolo_streamer.py

```bash
python3 vr_yolo_streamer.py [OPTIONS]

Options:
  --host TEXT        IP VR headset/PC
  --port INTEGER     UDP port
  --width INTEGER    Frame width
  --height INTEGER   Frame height
  --fps INTEGER      Target FPS
  --no-yolo          Disable YOLO detection
  --model TEXT       YOLO model (yolov8n.pt, yolov8s.pt, etc.)
  --conf FLOAT       Confidence threshold (default: 0.5)
```

**Przykłady:**

```bash
# YOLO Nano (najszybszy)
python3 vr_yolo_streamer.py --model yolov8n.pt --conf 0.5

# YOLO Small (lepszy accuracy, wolniejszy)
python3 vr_yolo_streamer.py --model yolov8s.pt --conf 0.6

# Bez YOLO (fallback do raw streaming)
python3 vr_yolo_streamer.py --no-yolo
```

---

## Bandwidth Requirements

| Resolution | FPS | Format | Bandwidth | Network |
|------------|-----|--------|-----------|---------|
| 2560x800 | 30 | RGB888 | 590 Mbps | Gigabit Ethernet |
| 2560x800 | 60 | RGB888 | 1180 Mbps | ⚠️ Exceeds Gigabit |
| 1280x400 | 30 | RGB888 | 148 Mbps | WiFi 5/6 |
| 1280x400 | 60 | RGB888 | 295 Mbps | Gigabit Ethernet |

**Rekomendacje:**
- **Gigabit Ethernet:** 2560x800 @ 30fps ✅
- **WiFi 6:** 1280x400 @ 30fps ✅
- **WiFi 5:** 1280x400 @ 20fps (obniż FPS)

---

## Expected Performance

### Raw UDP Streaming

```
Resolution: 2560x800
FPS: 30
Format: RGB888

Performance Metrics:
├─ Capture:     3-5ms
├─ Serialize:   2-3ms
├─ UDP Send:    10-15ms
├─ Network:     3-8ms (LAN)
└─ Total:       20-30ms ✅ VR Ready

CPU Usage: 15-20%
Bandwidth: 590 Mbps
```

### UDP + YOLO

```
Resolution: 1280x400
FPS: 30
Model: YOLOv8n

Performance Metrics:
├─ Capture:     3-5ms
├─ YOLO:        18-25ms
├─ UDP Send:    8-12ms
├─ Network:     3-8ms
└─ Total:       35-50ms ✅ Acceptable for VR

CPU Usage: 60-70%
Bandwidth: 148 Mbps
```

---

## Troubleshooting

### Problem: "No frames received"

**Sprawdź:**
1. Firewall - czy port 5000 UDP otwarty?
   ```bash
   # Na Pi
   sudo ufw allow 5000/udp
   ```

2. IP address - czy używasz poprawnego IP?
   ```bash
   # Na Pi
   hostname -I
   ```

3. Network connectivity
   ```bash
   ping <destination-ip>
   ```

---

### Problem: "High latency (>100ms)"

**Możliwe przyczyny:**

1. **WiFi zamiast Ethernet**
   - Fix: Podłącz kabel Gigabit

2. **Network congestion**
   - Fix: Zamknij inne aplikacje sieciowe
   - Fix: Użyj dedykowanej sieci

3. **CPU overload**
   ```bash
   top  # Sprawdź CPU usage
   ```
   - Fix: Obniż rozdzielczość
   - Fix: Wyłącz YOLO

---

### Problem: "Frame drops"

**Diagnostyka:**
```bash
# Sprawdź packet loss
python3 vr_udp_receiver.py --port 5000
# W logach: "Incomplete frames: X"
```

**Fix:**
1. Zwiększ UDP buffer:
   ```python
   # W vr_udp_receiver.py, zmień:
   self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**21)  # 2MB
   ```

2. Obniż bandwidth:
   ```bash
   # Niższa rozdzielczość lub FPS
   --width 1280 --height 400 --fps 20
   ```

---

### Problem: "High CPU (>90%)"

**Możliwe przyczyny:**

1. **YOLO zbyt ciężki**
   ```bash
   # Użyj Nano zamiast Small/Medium
   --model yolov8n.pt
   ```

2. **Zbyt wysoka rozdzielczość dla YOLO**
   ```bash
   # Maksymalnie 1280x400 dla YOLO na Pi 5
   --width 1280 --height 400
   ```

3. **Thermal throttling**
   ```bash
   vcgencmd measure_temp
   # Powinno być <70°C
   ```
   - Fix: Dodaj radiator i wentylator

---

## Advanced Usage

### Custom Processing Pipeline

```python
#!/usr/bin/env python3
"""
Custom pipeline - dodaj własne przetwarzanie
"""
from vr_udp_streamer import UDPFrameStreamer, StreamConfig
from picamera2 import Picamera2
import cv2

config = StreamConfig(
    width=2560,
    height=800,
    fps=30,
    vr_host="192.168.1.100"
)

streamer = UDPFrameStreamer(config)
picam2 = Picamera2()

# Configure camera
cam_config = picam2.create_video_configuration(
    main={"size": (2560, 800), "format": "RGB888"}
)
picam2.configure(cam_config)
picam2.start()

while True:
    # Capture
    frame = picam2.capture_array("main")

    # ============ CUSTOM PROCESSING HERE ============
    # Przykład: edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    frame_processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # ================================================

    # Stream
    streamer.send_frame(frame_processed)
```

---

### Side-by-Side Stereo Processing

```python
"""
Przetwarzaj osobno lewe i prawe oko
"""
import numpy as np

# Capture stereo frame (2560x800)
stereo_frame = picam2.capture_array("main")

# Split
left = stereo_frame[:, :1280, :]   # (800, 1280, 3)
right = stereo_frame[:, 1280:, :]  # (800, 1280, 3)

# Process tylko lewe oko (oszczędza CPU)
left_processed = yolo_detector.detect(left)

# Combine back
stereo_processed = np.hstack([left_processed, right])

# Stream
streamer.send_frame(stereo_processed)
```

---

## Integration z Istniejącym Kodem

### Integracja z backend/main.py

```python
# backend/main.py
from vr_udp_streamer import UDPFrameStreamer, StreamConfig

# W VRAnglesSender lub podobnej klasie:
class VRAnglesSender:
    def __init__(self, ...):
        # ... existing code ...

        # Add UDP streamer
        self.video_streamer = UDPFrameStreamer(
            StreamConfig(
                width=2560,
                height=800,
                fps=30,
                vr_host=vr_client_ip,
                vr_port=5001  # Different port dla video
            )
        )

    async def send_frame(self, frame: np.ndarray):
        """Non-blocking frame send"""
        await asyncio.to_thread(self.video_streamer.send_frame, frame)
```

---

## Performance Tuning Checklist

- [ ] **Network:** Gigabit Ethernet (nie WiFi)
- [ ] **Cables:** CAT6 lub lepsze
- [ ] **Switch:** Managed switch z QoS (opcjonalne)
- [ ] **Pi cooling:** Radiator + aktywny wentylator
- [ ] **Power:** Quality 5V 5A power supply
- [ ] **Firewall:** Port 5000 UDP otwarty
- [ ] **Buffer size:** Zwiększone bufory UDP
- [ ] **Resolution:** Dostosowane do bandwidth
- [ ] **FPS:** 30 dla balansu latency/smoothness

---

## Monitoring i Debugging

### Włącz verbose logging

```python
# Na początku pliku
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Real-time metrics

```bash
# Terminal 1 - Server
python3 vr_udp_streamer.py --host <ip>

# Terminal 2 - Monitor bandwidth
watch -n 1 'ifconfig eth0 | grep "RX packets"'

# Terminal 3 - Monitor CPU
htop
```

### Packet capture (advanced)

```bash
# Capture UDP packets
sudo tcpdump -i eth0 udp port 5000 -w vr_stream.pcap

# Analyze with Wireshark
wireshark vr_stream.pcap
```

---

## FAQ

**Q: Jaka jest minimalna latencja możliwa do osiągnięcia?**
A: ~20ms z raw UDP streaming w LAN. To składa się z:
- Capture: 3ms
- Network: 5ms
- Processing: 2ms
- Display: 10ms

**Q: Czy mogę używać WiFi?**
A: Tak, ale tylko dla niższych rozdzielczości (1280x400). WiFi dodaje jitter i zwiększa latencję do ~50-100ms.

**Q: Dlaczego YOLO dodaje tyle latencji?**
A: YOLOv8n na Pi 5 (CPU-only) zajmuje ~20-25ms dla 2560x800. Obniż do 1280x400 → ~14ms.

**Q: Czy mogę użyć GPU na Pi 5?**
A: Pi 5 nie ma CUDA, tylko CPU. PyTorch na Pi 5 = CPU only.

**Q: Jak zredukować bandwidth?**
A:
1. Niższa rozdzielczość (2560→1280)
2. Niższy FPS (30→20)
3. YUV420 zamiast RGB888 (-33% bandwidth)

**Q: Co zrobić jeśli latencja >50ms?**
A:
1. Sprawdź network (użyj Ethernet!)
2. Obniż rozdzielczość
3. Wyłącz YOLO
4. Sprawdź CPU throttling

---

## Dalsze Kroki

1. **Przeczytaj szczegółową analizę:** `docs/picamera2_vr_analysis.md`
2. **Benchmark swoją sieć:** `docs/streaming_comparison.md`
3. **Dostosuj parametry** do swojego hardware
4. **Test różnych rozdzielczości** i znajdź sweet spot
5. **Monitor performance** i iteruj

---

## Support

**Dokumentacja:**
- `docs/picamera2_vr_analysis.md` - Kompletna analiza techniczna
- `docs/streaming_comparison.md` - Testy i benchmarki

**Linki:**
- [Picamera2 Official Docs](https://picamera2.com/)
- [Raspberry Pi Forums](https://forums.raspberrypi.com/)
- [YOLO Ultralytics](https://docs.ultralytics.com/)

---

**Good luck with your VR project! 🚀**

*Last updated: 2025-11-26*