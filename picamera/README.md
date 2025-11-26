# Picamera2 - Przewodnik dla kamery Arducam pivariety MONO

Kamera: **Arducam pivariety [2560x800 10-bit MONO]**

## Dostępne tryby kamery

```
Tryb      | Rozdzielczość | Format     | FPS
----------|---------------|------------|-----
R8        | 1280x400      | 8-bit mono | 30
R8        | 2560x720      | 8-bit mono | 30
R8        | 2560x800      | 8-bit mono | 30
R10_CSI2P | 640x200       | 10-bit mono| 30
R10_CSI2P | 1280x400      | 10-bit mono| 30
R10_CSI2P | 2560x720      | 10-bit mono| 30
R10_CSI2P | 2560x800      | 10-bit mono| 30
```

## Szybki start

### Instalacja

```bash
sudo apt install -y python3-picamera2
```

### Najprostsze zdjęcie

```python
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start_and_capture_file("photo.jpg", delay=2)
picam2.close()
```

### Najprostsze wideo

```python
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start_and_record_video("video.h264", duration=5)
picam2.close()
```

## Podstawowy workflow

```python
from picamera2 import Picamera2
import time

# 1. Inicjalizacja
picam2 = Picamera2()

# 2. Konfiguracja
config = picam2.create_still_configuration()  # lub create_video_configuration()
picam2.configure(config)

# 3. Start
picam2.start()
time.sleep(2)  # stabilizacja

# 4. Operacje (zdjęcie/wideo/przetwarzanie)
picam2.capture_file("photo.jpg")

# 5. Stop
picam2.stop()
picam2.close()
```

### Z context manager (zalecane)

```python
from picamera2 import Picamera2
import time

with Picamera2() as picam2:
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    picam2.capture_file("photo.jpg")
# automatyczne zamknięcie
```

## Konfiguracje

### Still (zdjęcia)

```python
config = picam2.create_still_configuration(
    main={
        "size": (2560, 800),    # pełna rozdzielczość
        "format": "BGR888"
    },
    raw={},  # opcjonalnie surowe dane
)
```

### Video

```python
config = picam2.create_video_configuration(
    main={
        "size": (1280, 400),
        "format": "XBGR8888"
    },
    controls={
        "FrameDurationLimits": (33333, 33333)  # 30 FPS
    }
)
```

### Preview (podgląd)

```python
config = picam2.create_preview_configuration(
    main={
        "size": (640, 200),
        "format": "YUV420"
    }
)
```

## Robienie zdjęć

### Do pliku

```python
picam2.capture_file("photo.jpg")          # JPEG
picam2.capture_file("photo.png")          # PNG (bezstratny)
picam2.capture_file("photo.bmp")          # BMP
```

### Jako numpy array

```python
import numpy as np

array = picam2.capture_array("main")
print(f"Shape: {array.shape}, dtype: {array.dtype}")
```

### Jako PIL Image

```python
from PIL import Image

image = picam2.capture_image("main")
image.save("photo.png")
```

### Seria zdjęć

```python
for i in range(10):
    picam2.capture_file(f"photo_{i:03d}.jpg")
    time.sleep(0.5)
```

### Z pełną kontrolą (captured_request)

```python
with picam2.captured_request() as request:
    # Obraz jako array
    array = request.make_array("main")

    # Metadane
    metadata = request.get_metadata()
    print(f"Ekspozycja: {metadata['ExposureTime']}us")

    # Raw DNG
    request.save_dng("photo.dng")
```

## Nagrywanie wideo

### H264

```python
from picamera2.encoders import H264Encoder

encoder = H264Encoder(bitrate=10_000_000)  # 10 Mbps
picam2.start_recording(encoder, "video.h264")
time.sleep(10)
picam2.stop_recording()
```

### MP4 (wymaga FFmpeg)

```python
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

encoder = H264Encoder()
output = FfmpegOutput("video.mp4")
picam2.start_recording(encoder, output)
time.sleep(10)
picam2.stop_recording()
```

### Z kontrolą jakości

```python
from picamera2.encoders import H264Encoder, Quality

picam2.start_recording(
    H264Encoder(),
    "video.h264",
    quality=Quality.HIGH  # LOW, MEDIUM, HIGH, VERY_HIGH
)
```

## Kontrola parametrów kamery

### Wyświetl dostępne kontrolki

```python
controls = picam2.camera_controls
for name, (min_val, max_val, default) in sorted(controls.items()):
    print(f"{name}: ({min_val}, {max_val}, default={default})")
```

### Ustawianie parametrów

```python
# Ręczna ekspozycja
picam2.set_controls({
    "AeEnable": False,       # wyłącz auto-ekspozycję
    "ExposureTime": 10000,   # 10ms (w mikrosekundach)
    "AnalogueGain": 2.0,     # wzmocnienie x2
})

# Jasność i kontrast
picam2.set_controls({
    "Brightness": 0.2,       # -1.0 do 1.0
    "Contrast": 1.5,         # 0.0 do 32.0
    "Sharpness": 2.0,        # 0.0 do 16.0
})

# Framerate
frame_duration = int(1_000_000 / 30)  # 30 FPS
picam2.set_controls({
    "FrameDurationLimits": (frame_duration, frame_duration)
})
```

### Pobierz aktualne metadane

```python
metadata = picam2.capture_metadata()
print(f"Ekspozycja: {metadata['ExposureTime']}us")
print(f"Gain: {metadata['AnalogueGain']}")
print(f"Frame duration: {metadata['FrameDuration']}us")
```

## Specyficzne dla kamery MONO

### Przechwycenie RAW 10-bit

```python
config = picam2.create_still_configuration(
    main={"format": "BGR888"},
    raw={"format": "R10_CSI2P"},
    sensor={"bit_depth": 10}
)
picam2.configure(config)
picam2.start()

with picam2.captured_request() as request:
    raw_array = request.make_array("raw")
    # raw_array: uint16, wartości 0-1023

    # Normalizacja do 8-bit
    mono_8bit = (raw_array / 4).astype(np.uint8)
```

### Sprawdź tryby sensora

```python
for mode in picam2.sensor_modes:
    print(f"Format: {mode['format']}, Size: {mode['size']}, "
          f"FPS: {mode.get('fps')}, Bit depth: {mode.get('bit_depth')}")
```

### Optymalizacja dla mono

```python
# Wyłącz niepotrzebne przetwarzanie kolorów
picam2.set_controls({
    "AwbEnable": False,   # brak balansu bieli
    "Saturation": 0.0,    # brak saturacji
})
```

## Transformacje obrazu

```python
from libcamera import Transform

# Flip poziomy
config = picam2.create_still_configuration(
    transform=Transform(hflip=True)
)

# Flip pionowy
config = picam2.create_still_configuration(
    transform=Transform(vflip=True)
)

# Obrót 180°
config = picam2.create_still_configuration(
    transform=Transform(hflip=True, vflip=True)
)
```

## Ciągłe przechwytywanie (streaming)

```python
picam2.configure(picam2.create_preview_configuration())
picam2.start()

for i in range(100):
    array = picam2.capture_array("main")
    # przetwarzanie...
    process_frame(array)

picam2.stop()
```

## Przydatne wzorce

### Bracketing ekspozycji (HDR)

```python
metadata = picam2.capture_metadata()
base_exp = metadata['ExposureTime']

picam2.set_controls({"AeEnable": False})

for ev in [-2, -1, 0, 1, 2]:
    exposure = int(base_exp * (2 ** ev))
    picam2.set_controls({"ExposureTime": exposure})
    time.sleep(0.3)
    picam2.capture_file(f"bracket_ev{ev:+d}.jpg")
```

### Timelapse

```python
interval = 60  # sekundy między zdjęciami

for i in range(100):
    picam2.capture_file(f"timelapse_{i:04d}.jpg")
    time.sleep(interval)
```

### Przełączanie trybów (preview -> still)

```python
# Start w trybie preview
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)
picam2.start()

# ... podgląd działa ...

# Zrób zdjęcie w pełnej rozdzielczości
still_config = picam2.create_still_configuration()
picam2.switch_mode_and_capture_file(still_config, "photo.jpg")

# Automatycznie wraca do preview
```

## Struktura przykładów

```
picamera/
├── 01_capture_photo.py    # Robienie zdjęć
├── 02_record_video.py     # Nagrywanie wideo
├── 03_camera_settings.py  # Ustawienia kamery
├── 04_mono_camera.py      # Specyficzne dla mono
├── 05_numpy_processing.py # Przetwarzanie numpy/OpenCV
└── README.md              # Ten plik
```

## Uruchamianie przykładów

```bash
# Wszystkie funkcje
python 01_capture_photo.py

# Konkretny przykład
python 03_camera_settings.py info
python 03_camera_settings.py exposure
python 04_mono_camera.py modes
python 05_numpy_processing.py cv_basic
```

## Rozwiązywanie problemów

### Kamera nie wykryta

```bash
rpicam-hello --list-cameras
# Powinno pokazać kamerę
```

### Brak uprawnień

```bash
sudo usermod -a -G video $USER
# Wyloguj się i zaloguj ponownie
```

### Błędy pamięci

Zmniejsz liczbę buforów lub rozdzielczość:
```python
config = picam2.create_video_configuration(buffer_count=3)
```

### Wolne działanie

```python
# Użyj mniejszej rozdzielczości dla preview
config = picam2.create_preview_configuration(
    main={"size": (640, 200)}
)
```

## Przydatne linki

- [Dokumentacja Picamera2](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [GitHub Picamera2](https://github.com/raspberrypi/picamera2)
- [Arducam Wiki](https://docs.arducam.com/)