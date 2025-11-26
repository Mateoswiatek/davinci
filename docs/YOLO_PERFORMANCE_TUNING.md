# YOLO Performance Tuning Guide

Zaawansowane techniki optymalizacji YOLO na Raspberry Pi 5 dla VR.

---

## 1. CPU Optimization

### 1.1 CPU Governor (Max Performance)

```bash
# Check current governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set to performance mode (max 2.4GHz)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
# Should show: 2400000 (2.4GHz)

# Make permanent
sudo apt install cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

**Expected improvement:** 10-15% faster inference

### 1.2 CPU Affinity & Isolation

```bash
# Reserve cores for YOLO worker
# Edit /boot/firmware/cmdline.txt:
sudo nano /boot/firmware/cmdline.txt

# Add: isolcpus=2,3
# This reserves cores 2-3 for YOLO, cores 0-1 for system

# Reboot
sudo reboot

# Now run with CPU pinning:
sudo python backend/vr_yolo_streamer_v2.py \
  --model yolov8n.pt \
  --cores "2,3" \
  --priority -10
```

**Expected improvement:** 20-30% lower latency variance

### 1.3 Disable Services

```bash
# Disable unnecessary services to free CPU
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon
sudo systemctl disable triggerhappy

# Disable GUI (if running headless)
sudo systemctl set-default multi-user.target

# Reboot
sudo reboot
```

**Expected improvement:** 5-10% more CPU available

---

## 2. Memory Optimization

### 2.1 Increase GPU Memory Split

```bash
# Edit config
sudo nano /boot/firmware/config.txt

# Add/modify:
gpu_mem=128  # Default is 76, increase to 128

# Reboot
sudo reboot
```

**Why:** Frees more RAM for YOLO

### 2.2 Disable Swap (Reduce Latency Spikes)

```bash
# Disable swap
sudo dswapoff -a

# Remove swap file
sudo rm /var/swap

# Disable permanently
sudo systemctl disable dphys-swapfile
```

**Expected improvement:** Eliminates latency spikes from swapping

### 2.3 Memory Locking (Real-time)

```python
# In your code (requires root):
import os

# Lock memory (prevent swapping)
try:
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    libc.mlockall(3)  # MCL_CURRENT | MCL_FUTURE
    print("Memory locked")
except:
    pass
```

---

## 3. Thermal Management

### 3.1 Monitor Temperature

```bash
# Install monitoring
sudo apt install lm-sensors

# Watch temperature
watch -n 1 'vcgencmd measure_temp'

# Log temperature during benchmark
while true; do
  echo "$(date +%H:%M:%S) $(vcgencmd measure_temp)"
  sleep 1
done > temp_log.txt
```

### 3.2 Thermal Throttling Detection

```bash
# Check if throttled
vcgencmd get_throttled

# Output meanings:
# 0x0     = No throttling
# 0x50000 = Throttled in the past
# 0x50005 = Currently throttled!
```

**Critical:** If throttled, add cooling immediately!

### 3.3 Cooling Solutions

**Passive (cheap):**
- Aluminum heatsink on CPU: $5
- Expected: 10-15°C reduction

**Active (better):**
- 30mm 5V fan: $10
- Expected: 20-30°C reduction
- Allows sustained high performance

**Extreme:**
- Ice Tower cooler: $20
- Expected: 30-40°C reduction
- Totally overkill but looks cool!

---

## 4. Model Optimization

### 4.1 Input Resolution Trade-off

| Resolution | Inference (ms) | FPS | mAP | Use Case |
|------------|---------------|-----|-----|----------|
| 320x320 | 20-30ms | 33-50 | 25% | Fast, low accuracy |
| 416x416 | 40-50ms | 20-25 | 32% | Balanced |
| 640x640 | 80-100ms | 10-12 | 37% | **Default** |
| 1280x1280 | 300-400ms | 2-3 | 42% | High accuracy, slow |

**Recommendation:** 640x640 for general use, 416x416 if need more speed

### 4.2 Model Size Trade-off

| Model | Params | Size | FPS (CPU) | FPS (Hailo) | mAP | Best For |
|-------|--------|------|-----------|-------------|-----|----------|
| YOLO11n | 2.6M | 5MB | 12-14 | 500+ | 39.5% | **Fastest** |
| YOLOv8n | 3.2M | 6MB | 10-12 | 431 | 37.3% | Balanced |
| YOLOv8s | 11.2M | 22MB | 3-4 | 100 | 44.9% | Accuracy |

**Recommendation:** YOLO11n for CPU, YOLOv8n for accelerators

### 4.3 Class Filtering

```python
# Only detect specific classes (faster!)
yolo_config = YOLOConfig(
    model_path="yolov8n.pt",
    classes=[0, 1, 2]  # Only person, bicycle, car
)

# COCO class IDs:
# 0: person
# 1: bicycle
# 2: car
# 16: dog
# 17: cat
# ... (80 classes total)
```

**Expected improvement:** 10-20% faster with class filtering

---

## 5. Frame Skipping Strategies

### 5.1 Fixed Interval (Simple)

```python
# Process every 3rd frame
processing_config = ProcessingConfig(
    frame_skip_strategy=FrameSkipStrategy.FIXED,
    frame_skip_interval=3  # Every 3rd frame
)
```

**Use when:** Consistent performance, predictable detection rate

### 5.2 Adaptive (Smart)

```python
# Skip based on inference time
processing_config = ProcessingConfig(
    frame_skip_strategy=FrameSkipStrategy.ADAPTIVE,
    # Automatically adjusts based on latency
)
```

**Use when:** Variable load, want to maximize FPS while staying under latency budget

### 5.3 Queue-based (Backpressure)

```python
# Skip only when queue full
processing_config = ProcessingConfig(
    frame_skip_strategy=FrameSkipStrategy.QUEUE_FULL,
    max_queue_size=2
)
```

**Use when:** Want to process as many frames as possible without blocking

### 5.4 Hybrid Approach (Best)

```python
# Combine strategies
class HybridSkipStrategy:
    def __init__(self):
        self.base_interval = 3
        self.max_latency_ms = 40
        self.frame_count = 0

    def should_process(self, last_inference_ms):
        self.frame_count += 1

        # Fixed minimum skip
        if self.frame_count % self.base_interval != 0:
            return False

        # Adaptive: skip more if slow
        if last_inference_ms > self.max_latency_ms:
            skip_ratio = int(last_inference_ms / self.max_latency_ms)
            return self.frame_count % (self.base_interval * skip_ratio) == 0

        return True
```

---

## 6. Batching (Hailo/EdgeTPU Only)

### 6.1 Batch Processing for Higher Throughput

```python
# Hailo supports batching
# Collect 4 frames, process as batch
batch_size = 4
frame_buffer = []

while True:
    frame = camera.capture()
    frame_buffer.append(frame)

    if len(frame_buffer) >= batch_size:
        # Process batch (4x faster on Hailo!)
        results = hailo_model.infer_batch(frame_buffer)
        frame_buffer.clear()

        # Send all results
        for result in results:
            send_to_vr(result)
```

**Performance (Hailo):**
- Batch 1: 431 FPS
- Batch 4: 100 FPS (400 total throughput!)
- Batch 8: 120 FPS (960 total!)

**Trade-off:** Adds latency (need to wait for full batch)

---

## 7. Streaming Optimizations

### 7.1 Turbo JPEG

```bash
# Install turbojpeg (3-5x faster than PIL)
sudo apt install libturbojpeg0-dev
pip install PyTurboJPEG
```

```python
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

# Encode (much faster!)
encoded = jpeg.encode(frame, quality=85)

# Decode (much faster!)
decoded = jpeg.decode(encoded)
```

**Expected improvement:** 10-15ms → 2-3ms encoding

### 7.2 UDP Packet Size Tuning

```python
# Increase MTU for fewer packets
stream_config = StreamConfig(
    max_packet_size=8192,  # Default: 1472
    # Requires MTU 9000 (jumbo frames) on network
)

# Enable on both Pi and PC:
sudo ip link set eth0 mtu 9000
```

**Expected improvement:** 20-30% less CPU for UDP send

### 7.3 Zero-Copy Capture

```python
# Avoid unnecessary copies
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1920, 1080), "format": "RGB888"},
    buffer_count=2  # Minimal buffering
)
picam2.configure(config)

# Direct array access (zero-copy!)
frame = picam2.capture_array("main")  # No copy!
```

---

## 8. Profiling & Debugging

### 8.1 Line Profiler

```bash
pip install line_profiler
```

```python
# Add @profile decorator to functions
@profile
def process_frame(frame):
    # ... your code ...
    pass

# Run with profiler
kernprof -l -v your_script.py
```

### 8.2 Performance Counters

```python
import time
from collections import defaultdict

class PerformanceCounter:
    def __init__(self):
        self.timers = defaultdict(list)

    def time_block(self, name):
        """Context manager for timing"""
        class Timer:
            def __init__(self, counter, name):
                self.counter = counter
                self.name = name

            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, *args):
                elapsed = (time.perf_counter() - self.start) * 1000
                self.counter.timers[self.name].append(elapsed)

        return Timer(self, name)

    def report(self):
        for name, times in self.timers.items():
            avg = sum(times) / len(times)
            print(f"{name}: {avg:.2f}ms")

# Usage
perf = PerformanceCounter()

with perf.time_block("capture"):
    frame = camera.capture()

with perf.time_block("yolo"):
    result = yolo.detect(frame)

with perf.time_block("stream"):
    streamer.send(frame)

perf.report()
```

### 8.3 System Tracing

```bash
# CPU usage per process
top -H -p $(pgrep -f python)

# Network bandwidth
iftop -i eth0

# Disk I/O (should be minimal)
iotop

# Overall system
htop
```

---

## 9. Real-World Benchmarks

### Setup 1: Budget (CPU-only)

```yaml
Hardware: Pi 5 (4GB)
Model: YOLO11n NCNN
Resolution: 640x640
Frame skip: Every 5th

Results:
  Streaming FPS: 30
  Detection FPS: 6
  Latency: 15ms (streaming) + 85ms (detection) = 100ms worst case
  Amortized: ~20ms average

VR Suitable: YES (with caveats)
Cost: $60 (Pi only)
```

### Setup 2: Budget Accelerated

```yaml
Hardware: Pi 5 + Coral USB
Model: YOLOv8n EdgeTPU
Resolution: 640x640
Frame skip: Every 2nd

Results:
  Streaming FPS: 30
  Detection FPS: 15
  Latency: 15ms (streaming) + 17ms (detection) = 32ms worst case

VR Suitable: YES (excellent!)
Cost: $120 ($60 Pi + $60 Coral)
```

### Setup 3: Production (Recommended)

```yaml
Hardware: Pi 5 + Hailo-8L
Model: YOLOv8n Hailo HEF
Resolution: 640x640
Frame skip: None (every frame!)

Results:
  Streaming FPS: 30
  Detection FPS: 30
  Latency: 15ms (streaming) + 2.5ms (detection) = 17.5ms

VR Suitable: YES (perfect!)
Cost: $130 ($60 Pi + $70 Hailo)
```

### Setup 4: Extreme (Overkill)

```yaml
Hardware: Pi 5 (8GB) + Hailo-8L + Active Cooling
Model: YOLOv8s Hailo HEF (larger model)
Resolution: 640x640
Frame skip: None
Optimizations: CPU pinning, turbo JPEG, jumbo frames

Results:
  Streaming FPS: 60 (!)
  Detection FPS: 60
  Latency: 8ms (streaming) + 10ms (detection) = 18ms

VR Suitable: YES (overkill)
Cost: $160 ($80 Pi + $70 Hailo + $10 cooling)
```

---

## 10. Troubleshooting Performance Issues

### Issue: Lower FPS than expected

**Diagnosis:**
```bash
# Check CPU frequency
watch -n 1 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'

# If not 2400000, CPU is throttled
vcgencmd get_throttled
```

**Solutions:**
1. Add cooling
2. Set performance governor
3. Lower resolution or skip more frames

### Issue: High latency variance

**Diagnosis:**
```python
# Monitor P95 and P99 latency
stats = processor.get_stats()
if stats['p99_ms'] > stats['p50_ms'] * 2:
    print("High variance detected!")
```

**Solutions:**
1. Use CPU pinning
2. Disable swap
3. Close background processes

### Issue: Memory leaks

**Diagnosis:**
```bash
# Monitor memory over time
watch -n 1 'free -m'
```

**Solutions:**
1. Clear frame buffers regularly
2. Limit queue sizes
3. Use `del` for large objects

---

## 11. Final Recommendations

**For Development (Budget: $0):**
```bash
python backend/vr_yolo_streamer_v2.py \
  --model yolov8n.pt \
  --format pt \
  --skip 5 \
  --strategy adaptive \
  --no-annotation  # Skip drawing for testing
```

**For Production VR (Budget: $70):**
```bash
# With Hailo-8L
sudo python backend/vr_yolo_streamer_v2.py \
  --model yolov8n.hef \
  --format hef \
  --skip 1 \
  --cores "2,3" \
  --priority -10 \
  --host 192.168.1.100
```

**Key Takeaways:**
1. ✅ CPU governor → performance
2. ✅ Add cooling (fan minimum)
3. ✅ Frame skipping is OK (6-10 FPS detection is usable)
4. ✅ Hailo-8L worth the $70 if doing serious VR
5. ✅ Monitor temperature during development
6. ✅ Profile before optimizing

**DO NOT:**
- ❌ Run without cooling under load
- ❌ Use swap with real-time requirements
- ❌ Process every frame on CPU-only
- ❌ Ignore thermal throttling warnings

---

**Next:** Run `python backend/quick_yolo_test.py` to see what works best on your setup!