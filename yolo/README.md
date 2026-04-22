# YOLO — Object Detection (odłączone, do późniejszej integracji)

YOLO zostało tymczasowo wyizolowane bo wysypuje się na RPi i wymaga dużo zasobów.
Ten folder zawiera wszystkie implementacje i dokumentację — żeby łatwo wrócić gdy będzie potrzeba.

## Pliki

| Plik | Opis |
|------|------|
| `yolo_processor.py` | Główny moduł YOLO z `new/` — `YOLOProcessor`, `AsyncYOLOProcessor`, obsługa wielu backendów |
| `yolo_processor_optimized.py` | Zoptymalizowana wersja — CPU pinning, frame skipping, profilowanie |
| `OpenCVSimpleYOLO.py` | Prosta implementacja przez OpenCV DNN (bez PyTorch) |
| `vr_yolo_streamer.py` | Stary monolit — YOLO + UDP streaming razem |
| `vr_yolo_streamer_v2.py` | v2 starszego monolitu |
| `yolo_benchmark.py` | Testy wydajności różnych backendów na RPi |
| `raspberrypi-new.py` | Wersja `raspberrypi.py` z wbudowanym YOLO |
| `yolov4-tiny.cfg` | Konfiguracja sieci YOLOv4-tiny |
| `yolov4-tiny.weights` | Wagi modelu YOLOv4-tiny (24 MB) — backend: OpenCV DNN |
| `coco.names` | Nazwy 80 klas COCO |

## Jak pobrać wagi (jeśli brakuje)

```bash
cd yolo/
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

## Jak podłączyć do `new/`

`yolo_processor.py` był oryginalnie w `new/` i jest importowany przez `vr_streamer.py`:

```python
from yolo_processor import YOLOProcessor, AsyncYOLOProcessor, YOLOConfig, YOLOBackend
```

Żeby przywrócić YOLO:
1. Skopiuj `yolo_processor.py` do `new/`
2. W `new/run_server.py` użyj flagi `--with-yolo`
3. Upewnij się, że wagi są w `yolo/` (ścieżki są skonfigurowane w `run_server.py`)

## Backendy YOLO (od najlżejszego)

| Backend | Wymagania | Szybkość na Pi 5 | Uwagi |
|---------|-----------|-------------------|-------|
| `opencv_dnn` | `opencv-python` | ~100-200ms/frame | **Zalecany** — nie wymaga PyTorch |
| `onnx` | `onnxruntime` | ~80-150ms/frame | Dobry kompromis |
| `pytorch` | `ultralytics` (~500MB) | ~50-100ms/frame | Dużo RAM |
| `ncnn` | `ncnn` | ~30-60ms/frame | Najszybszy CPU, trudna instalacja |

## Krytyczna uwaga: RPi 5 NIE MA hardware H.264 encodera

RPi 5 nie ma sprzętowego encodera H.264 (był w RPi 4). Software encoding kosztuje ~80-100% CPU.
To sprawia, że YOLO + streaming jednocześnie jest bardzo wymagające.

**Rekomendacja przy powrocie do YOLO:**
- Używaj `async` YOLO (nie blokuje streaming pipeline)
- Frame skipping: `--yolo-skip 10` (co 10. klatka przez YOLO)
- Backend: `opencv_dnn` z YOLOv4-tiny (najmniejsze wymagania)
- Tylko detekcja osób: `--yolo-persons-only` (klasa 0)
