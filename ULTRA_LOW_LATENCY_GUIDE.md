# Ultra-Low-Latency VR Streaming Guide
## Osiągnięcie <30ms latencji na Raspberry Pi 5

---

## 📋 Spis Treści

1. [Quick Start](#quick-start)
2. [Architektura Systemu](#architektura-systemu)
3. [Szczegółowa Konfiguracja](#szczegółowa-konfiguracja)
4. [Profiling i Debugging](#profiling-i-debugging)
5. [Optymalizacje](#optymalizacje)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Krok 1: Automatyczna konfiguracja systemu

```bash
# Sklonuj repo (jeśli jeszcze nie)
cd /mnt/adata-disk/projects/agh/davinci/davinci

# Nadaj uprawnienia
chmod +x setup_low_latency_system.sh

# Uruchom setup (WYMAGA SUDO)
sudo ./setup_low_latency_system.sh

# RESTART SYSTEMU!
sudo reboot
```

### Krok 2: Weryfikacja konfiguracji

Po restarcie sprawdź:

```bash
# 1. Isolated CPUs
cat /sys/devices/system/cpu/isolated
# Powinno pokazać: 2-3

# 2. RT limits
ulimit -r
# Powinno pokazać: 99

# 3. BBR congestion control
sysctl net.ipv4.tcp_congestion_control
# Powinno pokazać: bbr

# 4. No swap
free -h
# Swap powinien być 0

# 5. CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Powinno pokazać: performance
```

### Krok 3: Uruchom VR Streaming

```bash
# Podstawowe uruchomienie (bez YOLO)
sudo python3 vr_streaming_optimized.py --ip 192.168.1.100

# Z YOLO detection (dodaje ~15ms latencji)
sudo python3 vr_streaming_optimized.py --ip 192.168.1.100 --yolo

# Z TCP zamiast UDP
sudo python3 vr_streaming_optimized.py --ip 192.168.1.100 --tcp

# Custom rozdzielczość
sudo python3 vr_streaming_optimized.py --ip 192.168.1.100 --width 1920 --height 1080
```

### Krok 4: Monitor latencji

Otwórz drugi terminal:

```bash
# Sprawdź procesy
ps aux | grep vr_streaming

# Monitor CPU per-core
htop
# Naciśnij F2 -> Display options -> Show custom thread names

# Live profiling
sudo py-spy top --pid <PID>

# Network stats
watch -n 1 'cat /sys/class/net/eth0/statistics/{rx,tx}_packets'
```

---

## 🏗️ Architektura Systemu

### Multi-Process Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Raspberry Pi 5 (4 cores)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CPU 0: System + IRQs                                          │
│  ├─ Kernel threads                                             │
│  └─ Network interrupts                                         │
│                                                                 │
│  CPU 1: Network Process (RT priority 85)                       │
│  ├─ H264 encoding                                              │
│  ├─ UDP/TCP sending                                            │
│  └─ Latency: ~5-10ms                                           │
│                                                                 │
│  CPU 2: Camera Process (RT priority 90) [ISOLATED]             │
│  ├─ Picamera2 capture                                          │
│  ├─ Zero-copy to shared memory                                 │
│  └─ Latency: ~2-5ms                                            │
│                                                                 │
│  CPU 3: YOLO Process (RT priority 70) [ISOLATED, OPTIONAL]     │
│  ├─ Object detection                                           │
│  ├─ YOLOv8n @ 320x320                                          │
│  └─ Latency: ~10-15ms                                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Shared Memory (Zero-Copy)                    │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  Ring Buffer 1: Camera → YOLO (3 x 2.8MB)            │    │
│  │  Ring Buffer 2: YOLO → Network (3 x 2.8MB)           │    │
│  └───────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓ UDP (5GHz WiFi or Ethernet)
                            ↓
                    ┌───────────────┐
                    │ Oculus Quest  │
                    │  WebRTC       │
                    └───────────────┘
```

### Latencja End-to-End (bez YOLO)

| Etap | Czas | Optymalizacja |
|------|------|---------------|
| Camera capture | 2-5ms | Picamera2 ultra_low profile, buffer_count=2 |
| Shared memory write | 0.5-1ms | Zero-copy np.copyto() |
| H264 encoding | 3-5ms | Hardware encoder (GPU) |
| Network send | 2-5ms | UDP, TCP_NODELAY, small buffers |
| Network RTT | 1-3ms | 5GHz WiFi, no power save |
| **TOTAL** | **9-19ms** | **Target: <30ms ✓** |

### Z YOLO

| Etap | Czas |
|------|------|
| Wszystko powyżej + YOLO | +10-15ms |
| **TOTAL** | **19-34ms** |

**Rekomendacja:** Dla <30ms, uruchom YOLO async (nie w głównym pipeline) lub wyłącz.

---

## 🔧 Szczegółowa Konfiguracja

### 1. CPU Pinning i Izolacja

**Cel:** Dedykowane rdzenie dla krytycznych procesów, brak preemption od system tasks.

#### cmdline.txt

```bash
sudo nano /boot/firmware/cmdline.txt
```

Dodaj:
```
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3
```

**Wyjaśnienie:**
- `isolcpus=2,3`: Kernel scheduler nie przypisuje automatycznie procesów do CPU 2-3
- `nohz_full=2,3`: Wyłącza timer tick na tych CPU (zmniejsza interrupty)
- `rcu_nocbs=2,3`: RCU callbacks wykonywane na innych CPU

#### Sprawdzenie

```bash
cat /sys/devices/system/cpu/isolated
# Output: 2-3
```

#### Kod Python - CPU Pinning

Zobacz: `cpu_pinning.py`

```python
from cpu_pinning import CPUPinner

pinner = CPUPinner()
pinner.set_cpu_affinity([2])  # Pin do CPU 2

# Lub użyj cgroups
from cpu_pinning import CGroupManager
cgroup = CGroupManager("my_process")
cgroup.create_cgroup(cpu_shares=2048, cpuset="2")
```

---

### 2. Real-Time Scheduling

**Cel:** Gwarancja wykonania dla krytycznych procesów, wyprzedzanie procesów non-RT.

#### Konfiguracja limits.conf

```bash
sudo nano /etc/security/limits.conf
```

Dodaj:
```
pi              -       rtprio          99
pi              -       nice            -20
pi              -       memlock         unlimited
```

**Wyloguj się i zaloguj ponownie!**

#### Sprawdzenie

```bash
ulimit -r
# Output: 99
```

#### Kod Python - RT Scheduling

Zobacz: `realtime_scheduler.py`

```python
from realtime_scheduler import RealtimeScheduler, SCHED_FIFO

scheduler = RealtimeScheduler()

# Ustaw RT priority dla bieżącego procesu
scheduler.set_realtime_priority(priority=90, policy=SCHED_FIFO)

# Dla wątku
scheduler.set_thread_realtime(priority=85)

# Lub użyj RealtimeThread
from realtime_scheduler import RealtimeThread

def camera_loop():
    # Your code
    pass

thread = RealtimeThread(
    target=camera_loop,
    priority=90,
    policy=SCHED_FIFO,
    cpu_affinity=[2]
)
thread.start()
```

#### Priorytety (1-99, wyższy = ważniejszy)

| Proces | Priority | Policy |
|--------|----------|--------|
| Camera capture | 90 | SCHED_FIFO |
| Network send | 85 | SCHED_FIFO |
| Encoding | 80 | SCHED_FIFO |
| YOLO | 70 | SCHED_FIFO |
| Stats/monitoring | 50 | SCHED_FIFO |

**SCHED_FIFO vs SCHED_RR:**
- `SCHED_FIFO`: First-In-First-Out, proces trzyma CPU aż się skończy
- `SCHED_RR`: Round-robin, proces dostaje timeslice

Dla VR streaming: **SCHED_FIFO** (niższa latencja, deterministyczne)

---

### 3. Memory Optimizations

#### Huge Pages

```bash
# Sprawdź dostępność
cat /proc/meminfo | grep Huge

# Alokuj 128 huge pages (128 x 2MB = 256MB)
echo 128 | sudo tee /proc/sys/vm/nr_hugepages

# Permanent w /etc/sysctl.conf
vm.nr_hugepages = 128
```

#### Memory Locking (mlockall)

Zobacz: `memory_optimizations.py`

```python
from memory_optimizations import MemoryManager

mem_mgr = MemoryManager()
mem_mgr.lock_all_memory()  # Zablokuj całą pamięć w RAM
```

**UWAGA:** Wymaga `memlock unlimited` w limits.conf!

#### Shared Memory Zero-Copy

Zobacz: `zero_copy_pipeline.py`

```python
from zero_copy_pipeline import ZeroCopyRingBuffer

# Producer (camera)
ring = ZeroCopyRingBuffer("vr_frames", buffer_count=3,
                         frame_shape=(720, 1280, 3), create=True)

write_buf = ring.get_write_buffer()
write_buf.write_frame(frame, frame_id=i, timestamp_ns=time.time_ns())
ring.commit_write(i)

# Consumer (network)
read_buf = ring.get_read_buffer()
if read_buf:
    frame_view, metadata = read_buf.read_frame()
    # frame_view to memoryview (zero-copy!)
    # Użyj bezpośrednio lub .copy() jeśli potrzebujesz
```

**Korzyści:**
- Brak kopiowania danych między procesami
- ~1ms dla write, ~0.1ms dla read
- Oszczędność CPU i pamięci

---

### 4. Network Tuning

#### Kernel Parameters (sysctl)

```bash
sudo nano /etc/sysctl.d/99-vr-lowlatency.conf
```

```ini
# BBR congestion control (lepsze dla WiFi)
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq

# TCP optimizations
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_nodelay = 1
net.ipv4.tcp_slow_start_after_idle = 0

# Buffers
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 262144
net.core.wmem_default = 262144
```

Zastosuj:
```bash
sudo sysctl -p /etc/sysctl.d/99-vr-lowlatency.conf
```

#### Network Interface

```bash
# Wyłącz offloading (niższa latencja)
sudo ethtool -K eth0 gso off tso off gro off

# Małe ring buffers
sudo ethtool -G eth0 rx 256 tx 256

# Wyłącz interrupt coalescing
sudo ethtool -C eth0 rx-usecs 0 tx-usecs 0
```

#### WiFi Power Save

```bash
# Wyłącz power save
sudo iw dev wlan0 set power_save off

# NetworkManager config
sudo nano /etc/NetworkManager/conf.d/wifi-powersave.conf
```

```ini
[connection]
wifi.powersave = 2
```

```bash
sudo systemctl restart NetworkManager
```

#### Kod Python - Socket Configuration

Zobacz: `network_optimizations.py`

```python
from network_optimizations import LowLatencySocket

# UDP (najniższa latencja)
sock = LowLatencySocket(use_udp=True, port=8554)
sock.bind('0.0.0.0')
sock.send(data, addr=(target_ip, target_port))

# TCP (jeśli UDP nie działa)
sock = LowLatencySocket(use_udp=False, port=8554)
# Automatycznie ustawia TCP_NODELAY, TCP_QUICKACK
```

**UDP vs TCP dla VR:**

| | UDP | TCP |
|---|-----|-----|
| Latencja | ✓ Najniższa | ✗ Wyższa (retransmit) |
| Packet loss | ✗ Brak recovery | ✓ Retransmit |
| Overhead | ✓ Minimalny | ✗ Większy |
| Implementacja | ✗ Złożona | ✓ Prosta |

**Rekomendacja:** UDP + forward error correction (FEC)

---

### 5. Picamera2 Tuning

#### Sensor Modes

```bash
# Lista dostępnych trybów
python3 -c "from picamera2 import Picamera2; print(Picamera2().sensor_modes)"
```

Dla niskiej latencji wybierz:
- Wysoki FPS (≥60)
- Niski crop
- Dopasowana rozdzielczość

#### Configuration

Zobacz: `picamera2_low_latency.py`

```python
from picamera2_low_latency import LowLatencyCamera

camera = LowLatencyCamera(
    camera_num=0,
    profile='ultra_low',  # lub 'low', 'balanced'
    stereo=True  # dla Arducam stereo
)

camera.initialize()

# Zero-copy capture
frame, timestamp = camera.capture_with_timestamp()

# Hardware H264 encoding
camera.start_recording_h264("output.h264", quality=Quality.VERY_LOW)
```

#### Profiles

| Profile | Buffer Count | Queue Size | Latencja | Jakość |
|---------|-------------|-----------|----------|--------|
| ultra_low | 2 | 1 | <10ms | ✗ Niska |
| low | 3 | 2 | <20ms | ✓ Dobra |
| balanced | 4 | 2 | <30ms | ✓✓ Wysoka |

**ultra_low:**
- AE/AWB wyłączone (fixed exposure)
- Noise reduction OFF
- Minimum bufors

**Rekomendacja:** Zacznij od `ultra_low`, zwiększ do `low` jeśli obraz niestabilny.

---

### 6. Zero-Copy Techniques

#### Zasady

1. **Unikaj kopii:** Używaj `np.copyto()` zamiast `np.copy()`
2. **Shared memory:** Transferuj między procesami bez kopiowania
3. **memoryview:** Zero-copy slicing
4. **DMA buffers:** Direct memory access z kamery (v4l2)

#### Benchmark

Zobacz: `zero_copy_pipeline.py`

```bash
python3 zero_copy_pipeline.py
```

Typowe wyniki (2560x800x3 frame):

| Metoda | Czas/frame |
|--------|-----------|
| memoryview (view only) | 0.001ms |
| np.copyto() | 1.2ms |
| np.copy() | 2.5ms |
| Shared memory write | 1.5ms |

**Różnica:** 2x szybciej!

#### Kod

```python
import numpy as np

# ✗ ZŁE - tworzy kopię
dest = np.copy(source)

# ✓ DOBRE - in-place, zero allocation
np.copyto(dest_buffer, source)

# ✓ NAJLEPSZE - zero-copy view
view = memoryview(source)
# Użyj view bezpośrednio lub konwertuj:
array = np.frombuffer(view, dtype=np.uint8).reshape((height, width, 3))
```

---

### 7. Profiling & Measurement

#### End-to-End Latency

Zobacz: `latency_profiler.py`

```python
from latency_profiler import LatencyProfiler

profiler = LatencyProfiler()

# Dla każdego frame'a
profiler.start_frame(frame_id)
profiler.mark('camera_capture')
# ... camera code ...
profiler.mark('camera_ready')
# ... encoding ...
profiler.mark('encode_done')
# ... network ...
profiler.mark('network_sent')
profiler.end_frame()

# Statystyki
profiler.print_statistics(last_n=100)
profiler.export_json("latency.json")
```

Output:
```
TOTAL LATENCY:
  Mean:    24.50 ms
  Median:  23.80 ms
  Min:     18.20 ms
  Max:     35.40 ms
  P95:     28.70 ms
  P99:     32.10 ms

PER-STAGE LATENCIES:
  camera_capture -> camera_ready:
    Mean:    4.20 ms
  ...
```

#### CPU Profiling (perf)

```bash
# Record
sudo perf record -F 999 -g -p <PID> -- sleep 10

# Report
sudo perf report

# Flame graph
sudo perf script | /opt/FlameGraph/stackcollapse-perf.pl > perf.folded
/opt/FlameGraph/flamegraph.pl perf.folded > flamegraph.svg
```

#### Python Profiling (py-spy)

```bash
# Install
pip install py-spy

# Live monitoring
sudo py-spy top --pid <PID>

# Record flame graph
sudo py-spy record --pid <PID> --duration 10 --output profile.svg

# Open profile.svg in browser
```

#### Network Latency

```bash
# Ping
ping -c 100 <oculus_ip>

# RTT statistics
ping -c 100 <oculus_ip> | tail -1

# Packet capture
sudo tcpdump -i wlan0 -w capture.pcap
# Analyze with Wireshark
```

---

## 🎯 Optymalizacje - Checklist

### System Level

- [ ] Isolated CPUs (2-3) w cmdline.txt
- [ ] GPU memory = 256 MB
- [ ] RT priority limits (ulimit -r = 99)
- [ ] Swap disabled
- [ ] CPU governor = performance
- [ ] BBR congestion control
- [ ] Transparent huge pages = madvise
- [ ] Network interface optimized (no offloading, small buffers)
- [ ] WiFi power save OFF

### Application Level

- [ ] CPU pinning dla każdego procesu
- [ ] RT scheduling (SCHED_FIFO)
- [ ] Memory locking (mlockall)
- [ ] Zero-copy shared memory
- [ ] UDP zamiast TCP
- [ ] Socket TCP_NODELAY (jeśli TCP)
- [ ] Small socket buffers (256KB)
- [ ] Picamera2 ultra_low profile
- [ ] Hardware H264 encoding
- [ ] Minimal buffering (2-3 frames max)

### Code Level

- [ ] np.copyto() zamiast np.copy()
- [ ] memoryview dla slicing
- [ ] Brak Python GIL contention (multiprocessing)
- [ ] Brak list/dict conversions w hot path
- [ ] Pre-allocated buffers (memory pool)
- [ ] Inline functions (dla C extensions)

---

## 🐛 Troubleshooting

### Problem: RT scheduling nie działa

**Symptom:**
```
OSError: sched_setscheduler failed: Operation not permitted
```

**Rozwiązanie:**
```bash
# Sprawdź limits
ulimit -r
# Jeśli 0, edytuj:
sudo nano /etc/security/limits.conf
# Dodaj:
pi  -  rtprio  99

# Wyloguj się i zaloguj ponownie
```

### Problem: Isolated CPUs nie działają

**Symptom:**
```bash
cat /sys/devices/system/cpu/isolated
# Puste
```

**Rozwiązanie:**
```bash
# Sprawdź cmdline.txt
cat /boot/firmware/cmdline.txt | grep isolcpus

# Jeśli brak, dodaj i restart
sudo nano /boot/firmware/cmdline.txt
# Dodaj: isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3
sudo reboot
```

### Problem: Wysoka latencja mimo optymalizacji

**Debug:**

1. **Zmierz każdy stage:**
```python
profiler.print_statistics()
# Sprawdź który stage jest wolny
```

2. **Sprawdź CPU usage:**
```bash
htop
# F5 dla tree view
# Sprawdź czy procesy są na właściwych CPU
```

3. **Network RTT:**
```bash
ping -c 100 <oculus_ip> | tail -1
# Jeśli >5ms, problem z siecią
```

4. **Camera FPS:**
```python
camera.measure_latency(num_frames=100)
# Sprawdź actual FPS
```

**Typowe przyczyny:**

| Problem | Rozwiązanie |
|---------|-------------|
| Camera >10ms | Zmień profile na ultra_low, zmniejsz rozdzielczość |
| YOLO >20ms | Użyj YOLOv8n, resize do 320x320, skip frames |
| Network >10ms | Sprawdź WiFi (użyj 5GHz), disable power save |
| Encoding >10ms | Użyj hardware H264, zmniejsz bitrate/jakość |

### Problem: Frame drops

**Symptom:**
```
[Camera] Frame 100
[Network] No new frames
```

**Przyczyny:**
1. Ring buffer overflow (consumer wolniejszy niż producer)
2. GC pauses
3. Swap (jeśli nie disabled)

**Rozwiązanie:**
```bash
# Sprawdź swap
free -h
# Jeśli >0, disable:
sudo swapoff -a

# Sprawdź ring buffer size
# Zwiększ num_buffers z 3 do 4

# Wyłącz GC w hot path
import gc
gc.disable()
# ... critical code ...
gc.enable()
```

### Problem: Picamera2 error

**Symptom:**
```
picamera2.picamera2.Picamera2Error: Camera is not open
```

**Rozwiązanie:**
```bash
# Sprawdź czy kamera wykryta
vcgencmd get_camera
# Powinno: supported=1 detected=1

# Jeśli nie, włącz w raspi-config
sudo raspi-config
# Interface Options -> Camera -> Enable

# Restart
sudo reboot

# Test
libcamera-hello --list-cameras
```

---

## 📊 Benchmarking

### Kompletny benchmark

```bash
# System
cd /mnt/adata-disk/projects/agh/davinci/davinci

# 1. Memory copy methods
python3 zero_copy_pipeline.py

# 2. Camera latency (wszystkie profile)
python3 picamera2_low_latency.py

# 3. Network latency
python3 network_optimizations.py

# 4. End-to-end (dry run bez kamery)
sudo python3 vr_streaming_optimized.py --ip 127.0.0.1
```

### Expected Results

**Memory copy (2560x800x3):**
- memoryview: <0.01ms
- np.copyto: ~1.2ms
- shared memory write: ~1.5ms

**Camera (720p):**
- ultra_low: 3-8ms avg
- low: 5-12ms avg
- balanced: 8-18ms avg

**Network (local):**
- UDP send: 0.5-2ms
- TCP send: 1-3ms

**End-to-end (no YOLO):**
- Avg: 20-28ms
- P95: <30ms
- P99: <35ms

---

## 🎓 Zaawansowane Topics

### Async YOLO

Dla <30ms z YOLO, uruchom detection async:

```python
# Skip frames - detect co 3 frame
if frame_id % 3 == 0:
    yolo_queue.put(frame)

# YOLO w background thread
def yolo_worker():
    while True:
        frame = yolo_queue.get()
        results = model(frame)
        # Update global state, don't block main pipeline
```

### Hardware Acceleration

```bash
# Check dla GPU/NPU na Pi 5
# (obecnie brak CUDA, rozważ Coral TPU USB)

# Coral TPU
pip3 install pycoral
# Model musi być skompilowany dla EdgeTPU
```

### WebRTC Streaming

Zamiast raw UDP, użyj WebRTC dla lepszej kompatybilności:

```bash
pip3 install aiortc

# Server
python3 webrtc_server.py

# Oculus: http://<pi_ip>:8080
```

---

## 📚 Pliki i Moduły

| Plik | Opis |
|------|------|
| `setup_low_latency_system.sh` | Automatyczna konfiguracja systemu |
| `vr_streaming_optimized.py` | Główny program VR streaming |
| `cpu_pinning.py` | CPU pinning i cgroups |
| `realtime_scheduler.py` | RT scheduling (SCHED_FIFO/RR) |
| `memory_optimizations.py` | Huge pages, mlockall, shared memory |
| `network_optimizations.py` | Socket tuning, sysctl, BBR |
| `picamera2_low_latency.py` | Camera configuration profiles |
| `zero_copy_pipeline.py` | Shared memory ring buffers |
| `latency_profiler.py` | End-to-end latency measurement |

---

## 🏆 Summary - Osiągnięcie <30ms

**Kluczowe optymalizacje:**

1. ✅ **CPU Isolation** - dedykowane rdzenie, zero contention
2. ✅ **RT Scheduling** - deterministyczne wykonanie, no preemption
3. ✅ **Zero-Copy** - shared memory, brak kopiowania
4. ✅ **UDP Networking** - no retransmits, minimal overhead
5. ✅ **Hardware Encoding** - GPU H264, nie CPU
6. ✅ **Ultra-Low Camera Profile** - 2 buffers, fixed exposure
7. ✅ **WiFi Optimization** - 5GHz, no power save, BBR

**Breakdown dla <30ms (bez YOLO):**

```
Camera:       3-5ms
Encoding:     3-5ms
Network:      2-5ms
WiFi RTT:     1-3ms
Overhead:     1-2ms
──────────────────
TOTAL:       10-20ms ✓
```

**Z YOLO (async, skip frames):**

```
+10-15ms (background, nie blokuje)
──────────────────────────────────
TOTAL:       20-35ms (avg <30ms) ✓
```

---

## 📞 Support

Jeśli masz problemy:

1. Sprawdź logs w terminalu
2. Uruchom benchmark scripts
3. Porównaj z expected results
4. Sprawdź troubleshooting section

---

**Powodzenia z ultra-low-latency VR streaming! 🚀**