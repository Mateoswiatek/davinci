# VR Streaming Ultra-Low-Latency - Cheat Sheet

## Szybki Start

```bash
# 1. Setup systemu (tylko raz)
sudo ./setup_low_latency_system.sh
sudo reboot

# 2. Weryfikacja
cat /sys/devices/system/cpu/isolated  # Powinno: 2-3
ulimit -r                              # Powinno: 99
free -h                                # Swap: 0

# 3. Quick benchmark
python3 quick_benchmark.py

# 4. Uruchom streaming
sudo python3 vr_streaming_optimized.py --ip 192.168.1.100
```

---

## Komendy Systemowe

### CPU

```bash
# Sprawdź isolated CPUs
cat /sys/devices/system/cpu/isolated

# Sprawdź CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Ustaw performance governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" | sudo tee $cpu
done

# Wyłącz CPU frequency scaling
sudo cpupower frequency-set -g performance

# Sprawdź IRQ affinity
grep . /proc/irq/*/smp_affinity_list

# Ustaw IRQ affinity (przykład: IRQ 45 na CPU 1)
echo 2 | sudo tee /proc/irq/45/smp_affinity  # 2 = bitmask dla CPU 1
```

### Memory

```bash
# Sprawdź huge pages
cat /proc/meminfo | grep Huge

# Alokuj huge pages
echo 128 | sudo tee /proc/sys/vm/nr_hugepages

# Sprawdź swap
free -h

# Wyłącz swap
sudo swapoff -a

# Permanent disable (edytuj /etc/fstab)
sudo nano /etc/fstab
# Zakomentuj linię ze swap

# Sprawdź memory locks
ulimit -l  # unlimited = OK
```

### Network

```bash
# Sprawdź BBR
sysctl net.ipv4.tcp_congestion_control

# Włącz BBR
echo "net.ipv4.tcp_congestion_control = bbr" | sudo tee -a /etc/sysctl.conf
echo "net.core.default_qdisc = fq" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Wyłącz offloading (eth0)
sudo ethtool -K eth0 gso off tso off gro off

# Wyłącz interrupt coalescing
sudo ethtool -C eth0 rx-usecs 0 tx-usecs 0

# Małe ring buffers
sudo ethtool -G eth0 rx 256 tx 256

# WiFi power save OFF
sudo iw dev wlan0 set power_save off

# Sprawdź network interface stats
cat /sys/class/net/eth0/statistics/rx_dropped
cat /sys/class/net/eth0/statistics/tx_dropped

# Ping test
ping -c 100 192.168.1.100 | tail -1
```

### Services

```bash
# Wyłącz IRQ balancing
sudo systemctl stop irqbalance
sudo systemctl disable irqbalance

# Wyłącz Bluetooth
sudo systemctl stop bluetooth
sudo systemctl disable bluetooth

# Lista aktywnych services
systemctl list-units --type=service --state=running
```

---

## Profiling i Debugging

### CPU Profiling

```bash
# perf record
sudo perf record -F 999 -g -p <PID> -- sleep 10

# perf report
sudo perf report

# perf top (live)
sudo perf top -p <PID>

# py-spy (Python)
pip install py-spy
sudo py-spy top --pid <PID>
sudo py-spy record --pid <PID> --duration 10 --output profile.svg
```

### Latency Measurement

```bash
# W kodzie Python:
from latency_profiler import LatencyProfiler

profiler = LatencyProfiler()
profiler.start_frame()
profiler.mark('camera_capture')
# ... code ...
profiler.mark('network_sent')
profiler.end_frame()
profiler.print_statistics()
```

### System Monitoring

```bash
# htop (per-core CPU)
htop
# F2 -> Display options -> Show custom thread names
# F5 -> Tree view

# Monitor specific process
htop -p <PID>

# Watch CPU usage
watch -n 1 'mpstat -P ALL 1 1'

# Watch memory
watch -n 1 'free -h'

# Watch network
watch -n 1 'cat /sys/class/net/eth0/statistics/{rx,tx}_packets'

# iotop (I/O monitoring)
sudo iotop -p <PID>
```

### Network Debugging

```bash
# tcpdump
sudo tcpdump -i wlan0 -w capture.pcap
# Analyze with Wireshark

# netstat (connections)
netstat -tulpn

# ss (socket statistics)
ss -s

# iperf (bandwidth test)
# Server:
iperf3 -s
# Client:
iperf3 -c <server_ip>

# Check MTU
ip link show eth0 | grep mtu
```

---

## Camera (Picamera2)

```bash
# Test camera
libcamera-hello --list-cameras
libcamera-hello -t 5000

# Capture test image
libcamera-still -o test.jpg

# List sensor modes
python3 -c "from picamera2 import Picamera2; cam = Picamera2(); print(cam.sensor_modes)"

# Python benchmark
python3 picamera2_low_latency.py
```

---

## Python Code Snippets

### CPU Pinning

```python
from cpu_pinning import CPUPinner

pinner = CPUPinner()
pinner.set_cpu_affinity([2])  # Pin to CPU 2
```

### RT Scheduling

```python
from realtime_scheduler import RealtimeScheduler, SCHED_FIFO

scheduler = RealtimeScheduler()
scheduler.set_realtime_priority(priority=90, policy=SCHED_FIFO)
```

### Memory Locking

```python
from memory_optimizations import MemoryManager

mem_mgr = MemoryManager()
mem_mgr.lock_all_memory()  # mlockall
```

### Shared Memory

```python
from zero_copy_pipeline import ZeroCopyRingBuffer

# Create
ring = ZeroCopyRingBuffer("vr_frames", buffer_count=3,
                         frame_shape=(720, 1280, 3), create=True)

# Write (producer)
buf = ring.get_write_buffer()
buf.write_frame(frame, frame_id=i, timestamp_ns=time.time_ns())
ring.commit_write(i)

# Read (consumer)
buf = ring.get_read_buffer()
if buf:
    frame_view, metadata = buf.read_frame()
    # Use frame_view (zero-copy memoryview)
```

### Low-Latency Socket

```python
from network_optimizations import LowLatencySocket

sock = LowLatencySocket(use_udp=True, port=8554)
sock.send(data, addr=(target_ip, target_port))
```

### Camera Capture

```python
from picamera2_low_latency import LowLatencyCamera

camera = LowLatencyCamera(profile='ultra_low', stereo=False)
camera.initialize()

frame, timestamp = camera.capture_with_timestamp()
```

---

## Quick Fixes

### Problem: "Operation not permitted" dla RT scheduling

```bash
# Sprawdź limits
ulimit -r

# Jeśli 0, edytuj:
sudo nano /etc/security/limits.conf
# Dodaj:
pi  -  rtprio  99

# Wyloguj i zaloguj ponownie
```

### Problem: Isolated CPUs nie działają

```bash
# Sprawdź cmdline.txt
cat /boot/firmware/cmdline.txt | grep isolcpus

# Jeśli brak, dodaj:
sudo nano /boot/firmware/cmdline.txt
# Dodaj na końcu linii:
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3

# Restart
sudo reboot
```

### Problem: Wysoka latencja

```bash
# 1. Sprawdź każdy stage
python3 -c "
from latency_profiler import LatencyProfiler
# Measure each stage separately
"

# 2. Sprawdź network RTT
ping -c 100 <oculus_ip> | tail -1

# 3. Sprawdź CPU usage
htop

# 4. Sprawdź dropped frames
cat /sys/class/net/wlan0/statistics/tx_dropped
```

### Problem: Camera error

```bash
# Sprawdź czy wykryta
vcgencmd get_camera

# Jeśli nie, enable w raspi-config
sudo raspi-config
# Interface Options -> Camera -> Enable

# Restart
sudo reboot

# Test
libcamera-hello --list-cameras
```

---

## Parametry Optymalizacyjne

### Picamera2 Profiles

| Profile | Buffer | Latency | Jakość |
|---------|--------|---------|--------|
| ultra_low | 2 | <10ms | Niska |
| low | 3 | <20ms | Średnia |
| balanced | 4 | <30ms | Wysoka |

```python
camera = LowLatencyCamera(profile='ultra_low')  # Zmień tutaj
```

### Ring Buffer Size

```python
# Mniej buforów = niższa latencja, więcej dropped frames
# Więcej buforów = wyższa latencja, mniej dropped frames

ring = ZeroCopyRingBuffer(..., buffer_count=3)  # 3-4 recommended
```

### Socket Buffers

```python
# Mniejsze bufory = niższa latencja
send_buf_size = 256 * 1024  # 256KB dla ultra-low
# vs
send_buf_size = 4 * 1024 * 1024  # 4MB dla high throughput
```

---

## Targets dla <30ms

| Komponent | Target | Command |
|-----------|--------|---------|
| Camera | <5ms | `python3 picamera2_low_latency.py` |
| Memory copy | <2ms | `python3 zero_copy_pipeline.py` |
| H264 encoding | <5ms | Hardware encoder |
| Network send | <5ms | UDP, small buffers |
| Network RTT | <3ms | `ping -c 100 <ip>` |
| **TOTAL** | **<20ms** | `python3 quick_benchmark.py` |

---

## Files Reference

| File | Purpose |
|------|---------|
| `setup_low_latency_system.sh` | Automatic system setup |
| `vr_streaming_optimized.py` | Main VR streaming program |
| `quick_benchmark.py` | Fast benchmark suite |
| `cpu_pinning.py` | CPU pinning utilities |
| `realtime_scheduler.py` | RT scheduling |
| `memory_optimizations.py` | Memory tuning |
| `network_optimizations.py` | Network tuning |
| `picamera2_low_latency.py` | Camera profiles |
| `zero_copy_pipeline.py` | Shared memory |
| `latency_profiler.py` | Latency measurement |

---

## Useful Links

- Raspberry Pi documentation: https://www.raspberrypi.com/documentation/
- Picamera2 manual: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
- Linux RT: https://wiki.linuxfoundation.org/realtime/start
- BBR congestion control: https://github.com/google/bbr

---

**Pro Tips:**

1. Zawsze uruchom `quick_benchmark.py` przed produkcją
2. Monitor latencji w czasie rzeczywistym: `LatencyProfiler.print_statistics()`
3. Dla <30ms: wyłącz YOLO lub uruchom async
4. Użyj 5GHz WiFi zamiast 2.4GHz
5. Wyłącz wszystkie niepotrzebne services
6. Pin network IRQs do CPU 0 lub 1 (nie isolated)

---

**Emergency Debug Commands:**

```bash
# System is slow?
htop                    # Check CPU
free -h                 # Check memory
iotop                   # Check I/O
netstat -s | grep -i lost  # Check packet loss

# Quick reset
sudo swapoff -a
sudo systemctl restart NetworkManager
sudo cpupower frequency-set -g performance

# Nuclear option
sudo reboot
```