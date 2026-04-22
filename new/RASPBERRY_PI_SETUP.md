# Raspberry Pi Setup - VR Streaming with YOLO

## Quick Start (5 minut)

### 1. Wymagane pliki YOLO
Pliki modelu znajdują się w katalogu `../yolo/`:
- `yolov4-tiny.cfg` - konfiguracja sieci
- `yolov4-tiny.weights` - wagi modelu (24 MB)
- `coco.names` - nazwy klas

Jeśli pliki nie istnieją, pobierz je:
```bash
cd /path/to/davinci/yolo
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

### 2. Uruchomienie serwera z YOLO
```bash
cd /path/to/davinci/computing

# Podstawowe YOLO (OpenCV DNN - zalecane!)
python run_server.py --with-yolo

# YOLO z detekcją tylko osób (szybsze)
python run_server.py --with-yolo --yolo-persons-only

# YOLO z UDP dla VR headsetu
python run_server.py --with-yolo --with-udp --target 192.168.1.100
```

---

## Konfiguracja Raspberry Pi dla optymalnej wydajności

### Krok 1: Aktualizacja systemu
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-opencv python3-picamera2
```

### Krok 2: Instalacja zależności Python
```bash
pip3 install numpy opencv-python-headless aiohttp websockets pillow
```

### Krok 3: Optymalizacja CPU Governor
```bash
# Ustaw tryb performance dla wszystkich rdzeni
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Dodaj do /etc/rc.local dla trwałości:
echo 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor' | sudo tee -a /etc/rc.local
```

### Krok 4: Izolacja CPU (opcjonalne, dla ultra-low-latency)
Edytuj `/boot/firmware/cmdline.txt`:
```
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3
```

Następnie uruchom serwer z pinowaniem CPU:
```bash
python run_server.py --with-yolo --cpu-camera 2 --cpu-yolo 3
```

### Krok 5: Pamięć GPU (opcjonalne)
Edytuj `/boot/firmware/config.txt`:
```
gpu_mem=128
```

---

## Opcje uruchomienia YOLO

### Backend OpenCV DNN (ZALECANY)
```bash
# Domyślny - lekki, nie wymaga PyTorch
python run_server.py --with-yolo

# Z wyższą dokładnością (wolniejszy)
python run_server.py --with-yolo --yolo-input-size 416

# Tylko osoby
python run_server.py --with-yolo --yolo-persons-only
```

### Backend PyTorch (wymaga więcej RAM)
```bash
# Jeśli masz zainstalowany ultralytics
python run_server.py --with-yolo-pytorch

# lub
python run_server.py --with-yolo --yolo-backend pytorch
```

### Parametry wydajności
```bash
# Szybszy (mniej dokładny)
python run_server.py --with-yolo --yolo-skip 15 --yolo-input-size 256 --yolo-confidence 0.4

# Dokładniejszy (wolniejszy)
python run_server.py --with-yolo --yolo-skip 5 --yolo-input-size 416 --yolo-confidence 0.25
```

---

## Porównanie backendów YOLO

| Backend | Zależności | RAM | Latencja | Uwagi |
|---------|-----------|-----|----------|-------|
| **opencv_dnn** | OpenCV (wbudowany) | ~100MB | ~80-100ms | **ZALECANY dla Pi** |
| pytorch | ultralytics, torch | ~400MB | ~150-200ms | Ciężki, ale dokładny |
| onnx | onnxruntime | ~150MB | ~80-100ms | Alternatywa dla opencv |
| ncnn | ncnn (kompilacja) | ~80MB | ~60-80ms | Najszybszy CPU |

---

## Skrypt startowy (systemd)

Stwórz plik `/etc/systemd/system/vr-streamer.service`:
```ini
[Unit]
Description=VR Streamer with YOLO
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/davinci/computing
ExecStart=/usr/bin/python3 run_server.py --with-yolo --yolo-persons-only
Restart=always
RestartSec=3

# Optymalizacje
Nice=-10
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=50

[Install]
WantedBy=multi-user.target
```

Aktywacja:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vr-streamer
sudo systemctl start vr-streamer
```

Status:
```bash
sudo systemctl status vr-streamer
journalctl -u vr-streamer -f
```

---

## Rozwiązywanie problemów

### Problem: "OpenCV DNN config file not found"
```bash
# Pobierz pliki YOLO
cd ../yolo
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

### Problem: Niska wydajność YOLO
```bash
# Zwiększ skip frames
python run_server.py --with-yolo --yolo-skip 15

# Zmniejsz rozmiar wejścia
python run_server.py --with-yolo --yolo-input-size 256

# Wykrywaj tylko osoby
python run_server.py --with-yolo --yolo-persons-only
```

### Problem: Brak kamery
```bash
# Sprawdź czy kamera jest podłączona
libcamera-hello --list-cameras

# Włącz kamerę w raspi-config
sudo raspi-config
# Interface Options -> Camera -> Enable
```

### Problem: Wysoka temperatura CPU
```bash
# Sprawdź temperaturę
vcgencmd measure_temp

# Dodaj aktywne chłodzenie lub zmniejsz obciążenie
python run_server.py --with-yolo --yolo-skip 20 --fps 24
```

---

## Zaawansowane: Konfiguracja sieci dla VR

### Ethernet (zalecane dla najniższej latencji)
```bash
# Statyczny IP
sudo nmcli con mod "Wired connection 1" ipv4.addresses 192.168.1.10/24
sudo nmcli con mod "Wired connection 1" ipv4.method manual
sudo nmcli con mod "Wired connection 1" ipv4.gateway 192.168.1.1
```

### WiFi 5GHz (alternatywa)
```bash
# Połącz z siecią 5GHz
sudo nmcli device wifi connect "YOUR_5GHZ_SSID" password "YOUR_PASSWORD"

# Sprawdź jakość połączenia
iwconfig wlan0
```

### Optymalizacja UDP
```bash
# Zwiększ bufory UDP
echo 'net.core.rmem_max=26214400' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_default=26214400' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max=26214400' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_default=26214400' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

## Podsumowanie wydajności

### Oczekiwane wyniki na Raspberry Pi 5 (z YOLO OpenCV DNN):

| Parametr | Bez YOLO | Z YOLO (skip=10) |
|----------|----------|------------------|
| FPS streamingu | 30 | 30 |
| Latencja (UDP) | 20-30ms | 20-35ms |
| Detekcje/s | - | 3 FPS |
| CPU | ~30% | ~60% |
| RAM | ~150MB | ~250MB |

### Zalecana konfiguracja dla VR:
```bash
python run_server.py \
  --with-yolo \
  --yolo-persons-only \
  --yolo-skip 10 \
  --yolo-input-size 320 \
  --yolo-confidence 0.3 \
  --with-udp \
  --target 192.168.1.X \
  --quality 80 \
  --fps 30
```