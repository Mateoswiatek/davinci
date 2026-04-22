# DaVinci V2 — Zdalna głowa VR

Raspberry Pi strumieniuje obraz kamery do okularów Oculus Quest. W drugą stronę przesyłane są kąty obrotu głowy operatora (pitch/yaw/roll), które sterują trzema serwami obracającymi kamerę — tak żeby pozycja kamery odpowiadała pozycji głowy operatora.

```
Operator (Oculus) ──── kąty głowy ────► RPi ──► serwa (pan/tilt/roll)
Operator (Oculus) ◄─── obraz JPEG ──── RPi ◄── kamera (Picamera2)
```

## Struktura

```
new/          # aktywna wersja — tutaj developerujemy
old/          # stara wersja (działa, ręcznie pisana)
yolo/         # YOLO odłączone — do integracji w przyszłości
docs/         # analizy, porównania, notatki techniczne
```

## Szybki start

### RPi — uruchomienie serwera

```bash
cd new/
pip install -r requirements.txt
python run_server.py
# WebSocket na porcie 8000, 1280x720, 30fps
```

### Oculus Quest — przeglądarka

```
http://<IP-RPi>:8000
```
Otwórz w przeglądarce na Oculus Quest. Plik HTML serwuje się automatycznie.

### PC — klient debug (OpenCV)

```bash
cd new/
python clients/debug_client.py --host <IP-RPi> --port 8000
```

## Dokumentacja

- [`new/README.md`](new/README.md) — szczegóły nowej wersji, opcje uruchomienia
- [`new/RASPBERRY_PI_SETUP.md`](new/RASPBERRY_PI_SETUP.md) — setup RPi krok po kroku
- [`old/README.md`](old/README.md) — opis starej wersji i jej protokołu
- [`yolo/README.md`](yolo/README.md) — jak podłączyć YOLO z powrotem
- [`docs/`](docs/) — analizy techniczne (latencja, protokoły, picamera2)

## Sprzęt

- Raspberry Pi 5
- Kamera: Picamera2 (Arducam stereo 2560x800 lub Camera Module 3)
- 3x serwa na GPIO: pan=14, tilt=15, roll=18
- Okulary: Oculus / Meta Quest (przeglądarka WebXR)

## Stan projektu

| Funkcja | Status |
|---------|--------|
| Streaming obrazu WebSocket | działa |
| Viewer przeglądarkowy (Oculus) | działa |
| Klient debug Python | działa |
| Sterowanie serwami | podłączone, nie w pełni przetestowane |
| UDP streaming | zaimplementowane, nie testowane |
| YOLO detection | wyizolowane w `yolo/` |
