# YOLO Setup Guide - Raspberry Pi 5

Kompletny przewodnik instalacji i konfiguracji YOLO na Raspberry Pi 5 dla różnych formatów i akceleratorów.

---

## Quick Start - CPU Only (Najszybszy Setup)

```bash
# 1. Install dependencies
pip install ultralytics opencv-python numpy

# 2. Download model
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Auto-downloads

# 3. Test
python backend/yolo_benchmark.py --pytorch yolov8n.pt --runs 50

# 4. Run VR streaming
python backend/vr_yolo_streamer_v2.py \
  --model yolov8n.pt \
  --format pt \
  --skip 5 \
  --host 192.168.1.100
```

**Expected Performance:**
- FPS: 5-6
- Latency: 150-200ms per inference
- With `--skip 5`: Effective 30ms amortized latency (OK for VR)

---

## Format 1: NCNN (FASTEST CPU-ONLY)

**Performance:** 10-12 FPS @ 640x640 (2-3x faster than PyTorch)

### Installation

```bash
# Install NCNN Python bindings
pip install ncnn

# Or build from source for better performance:
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=OFF \
      -DNCNN_BUILD_EXAMPLES=ON \
      -DNCNN_PYTHON=ON \
      ..
make -j4
cd python
pip install .
```

### Model Conversion

**IMPORTANT:** NCNN conversion must be done on x86_64, NOT on Pi!

```bash
# On your PC (x86_64 Linux/WSL):

# 1. Export YOLO to ONNX
pip install ultralytics
python3 << EOF
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', simplify=True, opset=12)
EOF

# 2. Download NCNN tools
wget https://github.com/Tencent/ncnn/releases/download/20240102/ncnn-20240102-ubuntu-2204.zip
unzip ncnn-20240102-ubuntu-2204.zip
cd ncnn-20240102-ubuntu-2204

# 3. Convert ONNX → NCNN
./onnx2ncnn yolov8n.onnx yolov8n.param yolov8n.bin

# 4. Optimize for ARM
./ncnnoptimize yolov8n.param yolov8n.bin yolov8n-opt.param yolov8n-opt.bin 1

# 5. Transfer to Pi
scp yolov8n-opt.param yolov8n-opt.bin pi@raspberrypi:~/models/
```

### Usage

```bash
# Benchmark
python backend/yolo_benchmark.py \
  --ncnn-param models/yolov8n-opt.param \
  --ncnn-bin models/yolov8n-opt.bin

# VR Streaming
python backend/vr_yolo_streamer_v2.py \
  --model models/yolov8n-opt.bin \
  --format ncnn \
  --skip 3 \
  --host 192.168.1.100
```

---

## Format 2: ONNX

**Performance:** 6-8 FPS @ 640x640 (slightly better than PyTorch)

### Installation

```bash
pip install onnxruntime
# Or for optimized ARM build:
pip install onnxruntime==1.16.3  # Known good version for ARM
```

### Model Conversion

```bash
# Export YOLO to ONNX
python3 << EOF
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(
    format='onnx',
    simplify=True,
    opset=12,
    dynamic=False,
    imgsz=640
)
EOF

# This creates: yolov8n.onnx
```

### Usage

```bash
# Benchmark
python backend/yolo_benchmark.py --onnx yolov8n.onnx

# VR Streaming
python backend/vr_yolo_streamer_v2.py \
  --model yolov8n.onnx \
  --format onnx \
  --skip 4
```

---

## Format 3: Coral Edge TPU (NAJLEPSZY BUDGET OPTION)

**Performance:** 50-66 FPS @ 640x640 (10x faster than CPU!)

**Required Hardware:** [Coral USB Accelerator](https://coral.ai/products/accelerator) ($60)

### Installation

```bash
# 1. Install Edge TPU runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt update
sudo apt install libedgetpu1-std python3-pycoral

# 2. Install TensorFlow Lite (specific version!)
pip install tensorflow==2.13.1
pip install tflite-runtime==2.14.0

# 3. Verify USB device
lsusb | grep "Global Unichip"
# Should show: Bus 00X Device 00X: ID 1a6e:089a Global Unichip Corp.
```

### Model Conversion

**IMPORTANT:** EdgeTPU compilation ONLY works on x86_64!

```bash
# On your PC (x86_64 Linux - NOT Pi!):

# 1. Install EdgeTPU Compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update
sudo apt install edgetpu-compiler

# 2. Export YOLO to EdgeTPU format
pip install ultralytics tensorflow==2.13.1

python3 << EOF
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(
    format='edgetpu',
    imgsz=640,
    int8=True,
    data='coco128.yaml'  # Calibration dataset
)
EOF

# This creates: yolov8n_saved_model/yolov8n_full_integer_quant_edgetpu.tflite

# 3. Transfer to Pi
scp yolov8n_saved_model/*_edgetpu.tflite pi@raspberrypi:~/models/
```

### Usage

```bash
# Test EdgeTPU
python3 << EOF
from pycoral.utils import edgetpu
from pycoral.adapters import common

# List EdgeTPU devices
devices = edgetpu.list_edge_tpus()
print(f"EdgeTPU devices: {devices}")

# Load model
interpreter = edgetpu.make_interpreter('models/yolov8n_full_integer_quant_edgetpu.tflite')
interpreter.allocate_tensors()
print("Model loaded successfully!")
EOF

# Benchmark
python backend/yolo_benchmark.py \
  --edgetpu models/yolov8n_full_integer_quant_edgetpu.tflite

# VR Streaming (can process every 2nd frame!)
python backend/vr_yolo_streamer_v2.py \
  --model models/yolov8n_full_integer_quant_edgetpu.tflite \
  --format edgetpu \
  --skip 2 \
  --host 192.168.1.100
```

### Troubleshooting EdgeTPU

#### Problem: "Failed to load delegate from libedgetpu.so.1"

```bash
# Solution 1: Reinstall runtime
sudo apt remove libedgetpu1-std
sudo apt install libedgetpu1-std

# Solution 2: Check USB permissions
sudo usermod -aG plugdev $USER
# Log out and back in

# Solution 3: Update udev rules
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a6e", GROUP="plugdev"' | \
  sudo tee /etc/udev/rules.d/99-edgetpu-accelerator.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

#### Problem: "Model not compatible with EdgeTPU"

```bash
# Model filename MUST contain "_edgetpu.tflite"
mv your_model.tflite your_model_edgetpu.tflite
```

---

## Format 4: Hailo-8L AI Kit (NAJSZYBSZY - PRODUCTION)

**Performance:** 431 FPS @ 640x640 (100x faster than CPU!)

**Required Hardware:** [Raspberry Pi AI Kit](https://www.raspberrypi.com/products/ai-kit/) ($70)

### Installation

```bash
# 1. Update system (requires latest kernel)
sudo apt update && sudo apt full-upgrade -y
sudo reboot

# 2. Install Hailo software
sudo apt install hailo-all

# 3. Verify installation
hailortcli fw-control identify
# Should show: Hailo-8L device info

# 4. Install Hailo Python
pip install hailo-platform hailort

# 5. Check PCIe
lspci | grep Hailo
# Should show: Hailo-8 AI Processor
```

### Model Conversion

**Requires Hailo Dataflow Compiler (DFC) - x86_64 only!**

```bash
# On your PC (x86_64 Linux):

# 1. Install Hailo DFC (requires registration)
# Download from: https://hailo.ai/developer-zone/

# 2. Install hailomz (model zoo)
pip install hailomz

# 3. Compile YOLOv8
hailomz compile yolov8n \
  --ckpt=yolov8n.pt \
  --calib-path=coco_calib_128.tfrecord \
  --hw-arch hailo8l \
  --performance

# This creates: yolov8n.hef

# 4. Transfer to Pi
scp yolov8n.hef pi@raspberrypi:~/models/
```

### Pre-compiled Models

Hailo provides pre-compiled models in their model zoo:

```bash
# Download pre-compiled YOLOv8n for Hailo-8L
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8l/yolov8n.hef

# Or use hailomz
hailomz download yolov8n --hw-arch hailo8l
```

### Usage

```bash
# Test Hailo
python3 << EOF
from hailo_platform import HEF, Device, VDevice, InferVStreams, ConfigureParams

hef = HEF("models/yolov8n.hef")
devices = Device.scan()
print(f"Hailo devices: {devices}")
EOF

# VR Streaming (can process EVERY frame at 30 FPS!)
python backend/vr_yolo_streamer_v2.py \
  --model models/yolov8n.hef \
  --format hef \
  --skip 1 \
  --host 192.168.1.100
```

### Troubleshooting Hailo

#### Problem: "No Hailo devices found"

```bash
# Check PCIe
lspci | grep Hailo

# If not shown, check HAT connection
# Ensure M.2 HAT is properly connected to PCIe slot

# Update firmware
sudo hailortcli fw-update
```

#### Problem: "PCIe Gen2 instead of Gen3"

```bash
# Enable PCIe Gen3 (experimental)
echo "dtparam=pciex1_gen=3" | sudo tee -a /boot/firmware/config.txt
sudo reboot

# Verify
lspci -vv | grep -A 10 Hailo | grep LnkSta
# Should show: Speed 8GT/s (ok), Width x1 (ok)
```

---

## Format Comparison Summary

| Format | FPS | Latency | Cost | Setup Difficulty | Best For |
|--------|-----|---------|------|------------------|----------|
| PyTorch | 5-6 | 150-200ms | $0 | Easy | Testing |
| ONNX | 6-8 | 120-150ms | $0 | Easy | Slightly better than PT |
| NCNN | 10-12 | 80-100ms | $0 | Medium | **Best CPU-only** |
| EdgeTPU | 50-66 | 15-20ms | $60 | Medium | **Best budget** |
| Hailo-8L | 431 | 2-3ms | $70 | Hard | **Best performance** |

---

## Model Size Comparison

| Model | Parameters | Size (PT) | Size (ONNX) | Size (NCNN) | Size (EdgeTPU) | Size (Hailo) |
|-------|-----------|-----------|-------------|-------------|----------------|--------------|
| YOLOv8n | 3.2M | 6.2 MB | 12.1 MB | 12.0 MB | 3.2 MB | 4.8 MB |
| YOLOv8s | 11.2M | 22.5 MB | 44.8 MB | 44.5 MB | 11.4 MB | 16.2 MB |
| YOLO11n | 2.6M | 5.1 MB | 10.2 MB | 10.0 MB | 2.7 MB | 3.9 MB |

---

## Recommended Configurations

### Budget: $0 (CPU-only)

```bash
# NCNN with aggressive frame skipping
python backend/vr_yolo_streamer_v2.py \
  --model yolov8n-opt.bin \
  --format ncnn \
  --skip 5 \
  --strategy adaptive \
  --host 192.168.1.100 \
  --fps 30
```

**Expected:**
- Streaming: 30 FPS (smooth)
- Detection: 6 FPS (every 5th frame)
- Latency: ~15ms total (excellent!)

### Budget: $60 (Coral USB)

```bash
# EdgeTPU with moderate skipping
python backend/vr_yolo_streamer_v2.py \
  --model yolov8n_edgetpu.tflite \
  --format edgetpu \
  --skip 2 \
  --host 192.168.1.100 \
  --fps 30
```

**Expected:**
- Streaming: 30 FPS
- Detection: 15 FPS (every 2nd frame)
- Latency: ~25ms (very good!)

### Budget: $70 (Hailo-8L) - RECOMMENDED

```bash
# Hailo with no skipping!
python backend/vr_yolo_streamer_v2.py \
  --model yolov8n.hef \
  --format hef \
  --skip 1 \
  --host 192.168.1.100 \
  --fps 30
```

**Expected:**
- Streaming: 30 FPS
- Detection: 30 FPS (real-time!)
- Latency: ~15ms (excellent!)

---

## Advanced: CPU Pinning for Real-Time Performance

Pin YOLO worker to specific cores for lower latency variance:

```bash
# Pin worker to cores 2-3 (leave 0-1 for main thread)
sudo python backend/vr_yolo_streamer_v2.py \
  --model yolov8n.pt \
  --cores "2,3" \
  --priority -10 \
  --host 192.168.1.100
```

**Note:** Requires `sudo` for priority setting.

---

## Benchmarking Your Setup

```bash
# Compare all available formats
python backend/yolo_benchmark.py \
  --pytorch yolov8n.pt \
  --onnx yolov8n.onnx \
  --ncnn-param yolov8n-opt.param \
  --ncnn-bin yolov8n-opt.bin \
  --edgetpu yolov8n_edgetpu.tflite \
  --runs 100 \
  --output benchmark_results.json

# View results
cat benchmark_results.json | python -m json.tool
```

---

## Troubleshooting Common Issues

### High CPU Usage

```bash
# Reduce FPS
--fps 24  # Instead of 30

# Increase skip interval
--skip 5  # Instead of 3

# Disable annotation
--no-annotation

# Use smaller model
--model yolo11n.pt  # Instead of yolov8n
```

### High Temperature

```bash
# Monitor temperature
watch -n 1 "vcgencmd measure_temp"

# Add heatsink or fan
# Reduce workload (see above)
```

### OOM (Out of Memory)

```bash
# Reduce queue size
--queue-size 1  # Instead of 2

# Use smaller input size (requires model re-export)
--size 320  # Instead of 640
```

---

## Next Steps

1. **Choose your format** based on budget and requirements
2. **Run benchmark** to verify performance
3. **Test VR streaming** with your setup
4. **Tune skip interval** to balance FPS vs latency
5. **Monitor system** resources and temperature

For production deployment, see: `/docs/VR_STREAMING_README.md`