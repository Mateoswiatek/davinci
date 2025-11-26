# YOLO Quick Reference - Raspberry Pi 5

One-page cheat sheet for YOLO on Pi 5.

---

## Decision Tree

```
Budget?
├─ $0 (CPU-only)
│  └─ Model: YOLO11n NCNN
│     └─ FPS: 10-12
│        └─ Strategy: Skip every 5th frame
│           └─ VR: ✅ OK (with skipping)
│
├─ $60 (Coral USB)
│  └─ Model: YOLOv8n EdgeTPU
│     └─ FPS: 50-66
│        └─ Strategy: Skip every 2nd frame
│           └─ VR: ✅ Excellent
│
└─ $70 (Hailo-8L)
   └─ Model: YOLOv8n HEF
      └─ FPS: 431
         └─ Strategy: Process every frame!
            └─ VR: ✅ Perfect
```

---

## Quick Commands

### Test what you have
```bash
python backend/quick_yolo_test.py
```

### Benchmark
```bash
# CPU
python backend/yolo_benchmark.py --pytorch yolov8n.pt --runs 100

# Coral
python backend/yolo_benchmark.py --edgetpu model_edgetpu.tflite --runs 100

# Compare all
python backend/yolo_benchmark.py \
  --pytorch yolov8n.pt \
  --onnx yolov8n.onnx \
  --output results.json
```

### VR Streaming

**CPU-only:**
```bash
python backend/vr_yolo_streamer_v2.py \
  --model yolov8n.pt \
  --skip 5 \
  --host 192.168.1.100
```

**Coral USB:**
```bash
python backend/vr_yolo_streamer_v2.py \
  --model yolov8n_edgetpu.tflite \
  --format edgetpu \
  --skip 2 \
  --host 192.168.1.100
```

**Hailo-8L:**
```bash
python backend/vr_yolo_streamer_v2.py \
  --model yolov8n.hef \
  --format hef \
  --skip 1 \
  --host 192.168.1.100
```

---

## Performance Table

| Setup | Hardware | FPS | Latency | Cost | VR OK? | Notes |
|-------|----------|-----|---------|------|--------|-------|
| **CPU Basic** | Pi 5 | 5-6 | 150ms | $60 | ⚠️ | PyTorch, skip=5 |
| **CPU NCNN** | Pi 5 | 10-12 | 85ms | $60 | ✅ | Fastest CPU, skip=5 |
| **Coral** | Pi 5 + USB | 50-66 | 17ms | $120 | ✅ | Best budget |
| **Hailo** | Pi 5 + M.2 | 431 | 2.5ms | $130 | ✅ | Best overall |

---

## Installation One-Liners

### CPU (PyTorch)
```bash
pip install ultralytics opencv-python numpy
```

### CPU (NCNN) - Fastest
```bash
pip install ncnn
# Convert on PC, transfer .param + .bin files
```

### Coral EdgeTPU
```bash
# Add Coral repo
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Install
sudo apt update
sudo apt install python3-pycoral libedgetpu1-std

# Test
python3 -c "from pycoral.utils import edgetpu; print(edgetpu.list_edge_tpus())"
```

### Hailo-8L
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install hailo-all
pip install hailo-platform

# Test
hailortcli fw-control identify
```

---

## Model Export Quick Reference

### PyTorch → ONNX
```python
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='onnx', simplify=True)
```

### PyTorch → NCNN (on PC!)
```bash
# Step 1: Export to ONNX
yolo export model=yolov8n.pt format=onnx

# Step 2: ONNX → NCNN (download ncnn tools first)
./onnx2ncnn yolov8n.onnx yolov8n.param yolov8n.bin
./ncnnoptimize yolov8n.param yolov8n.bin yolov8n-opt.param yolov8n-opt.bin 1
```

### PyTorch → EdgeTPU (on PC!)
```python
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='edgetpu', imgsz=640)
# Creates: yolov8n_saved_model/*_edgetpu.tflite
```

### PyTorch → Hailo (on PC with DFC)
```bash
hailomz compile yolov8n --ckpt=yolov8n.pt --hw-arch hailo8l
# Or download pre-compiled:
hailomz download yolov8n --hw-arch hailo8l
```

---

## Optimization Checklist

### Essential (Do First)
- [ ] Set CPU governor to `performance`
- [ ] Add heatsink or fan
- [ ] Test with `quick_yolo_test.py`
- [ ] Choose appropriate frame skip interval

### Recommended
- [ ] Pin YOLO worker to cores 2-3
- [ ] Disable unnecessary services
- [ ] Monitor temperature during load
- [ ] Profile to find bottlenecks

### Advanced
- [ ] Enable jumbo frames (MTU 9000)
- [ ] Use TurboJPEG for encoding
- [ ] Disable swap
- [ ] Lock memory for real-time

### Performance Governor
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Check Throttling
```bash
vcgencmd get_throttled
# 0x0 = OK, anything else = throttled!
```

---

## Troubleshooting

### Low FPS
1. Check: `vcgencmd measure_temp` (add cooling if >70°C)
2. Check: `vcgencmd get_throttled` (0x0 = OK)
3. Set performance governor
4. Increase frame skip interval

### High Latency
1. Use async processor (multiprocessing)
2. Increase skip interval
3. Reduce input resolution (320x320 or 416x416)
4. Disable annotation (`--no-annotation`)

### OOM (Out of Memory)
1. Reduce queue size: `--queue-size 1`
2. Use smaller model (YOLO11n)
3. Lower input resolution
4. Close background processes

### Coral Not Detected
```bash
# Check USB
lsusb | grep "Global Unichip"

# Reinstall runtime
sudo apt remove libedgetpu1-std
sudo apt install libedgetpu1-std

# Add to plugdev group
sudo usermod -aG plugdev $USER
```

### Hailo Not Detected
```bash
# Check PCIe
lspci | grep Hailo

# If not found, check HAT connection
# Update firmware
sudo hailortcli fw-update
```

---

## Performance Targets

| Metric | Target | Why |
|--------|--------|-----|
| Total latency | <50ms | VR comfort threshold |
| Streaming FPS | 30 | Smooth video |
| Detection FPS | 6-10 | Usable object tracking |
| Temperature | <70°C | Avoid throttling |
| CPU usage | <85% | Leave headroom |

---

## Model Recommendations

| Use Case | Model | Format | Hardware |
|----------|-------|--------|----------|
| **Testing** | YOLOv8n | PyTorch | CPU |
| **CPU Production** | YOLO11n | NCNN | CPU |
| **Budget Accelerated** | YOLOv8n | EdgeTPU | Coral USB |
| **Best Performance** | YOLOv8n/s | HEF | Hailo-8L |
| **High Accuracy** | YOLOv8m | HEF | Hailo-8L |

---

## Common Pitfalls

❌ **DON'T:**
- Run without cooling
- Use INT8 quantization on CPU (slower than FP32!)
- Process every frame on CPU-only
- Ignore thermal throttling
- Use threading instead of multiprocessing
- Forget to set performance governor

✅ **DO:**
- Add at least passive cooling
- Use NCNN for CPU-only deployments
- Frame skipping is your friend
- Monitor temperature
- Use multiprocessing for async
- Profile before optimizing

---

## Resources

- Full guide: `docs/YOLO_OPTIMIZATION_GUIDE.md`
- Setup: `docs/YOLO_SETUP_GUIDE.md`
- Tuning: `docs/YOLO_PERFORMANCE_TUNING.md`
- Code: `backend/yolo_processor_optimized.py`
- Benchmark: `backend/yolo_benchmark.py`
- Quick test: `backend/quick_yolo_test.py`

---

## One-Line Setup

```bash
# CPU-only (instant)
pip install ultralytics && python backend/vr_yolo_streamer_v2.py --model yolov8n.pt --skip 5

# Coral (after hardware setup)
sudo apt install python3-pycoral && python backend/vr_yolo_streamer_v2.py --model model_edgetpu.tflite --format edgetpu --skip 2

# Hailo (after hardware setup)
sudo apt install hailo-all && python backend/vr_yolo_streamer_v2.py --model yolov8n.hef --format hef --skip 1
```

---

**TL;DR:**
- Testing? Use PyTorch
- Production CPU? Use NCNN + skip 5
- Have $60? Buy Coral USB
- Have $70? Buy Hailo-8L (best ROI)
- Always add cooling!