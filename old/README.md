# Stara wersja — DaVinci V2 (ręcznie pisana)

Pierwsza działająca wersja projektu. Napisana ręcznie, bez dodatkowych abstrakcji.
Streaming obrazu potwierdzony jako działający. Sterowanie serwami było podłączone ale nie w pełni przetestowane.

## Pliki

| Plik | Opis |
|------|------|
| `raspberrypi.py` | Serwer na RPi — kamera → JPEG → base64 → WebSocket → klienci |
| `client_t.py` | Klient testowy na PC — odbiera obraz, wyświetla przez OpenCV, wysyła kąty głowy |
| `vr.html` | Przeglądarkowy viewer VR — działa na Oculus Quest przez ngrok lub sieć LAN |
| `VRAnglesSender.py` | Standalone — wysyła kąty głowy do RPi bez obrazu (do testowania serw klawiaturą) |
| `vr_udp_streamer.py` | Stary monolit — UDP streaming z RPi (<30ms latencja, ale brak podziału na moduły) |
| `vr_udp_receiver.py` | Odbiorca UDP — para do vr_udp_streamer.py |

## Jak uruchomić (stara wersja)

### RPi — serwer
```bash
python3 raspberrypi.py
# Domyślnie: ws://0.0.0.0:8765, 1600x540, JPEG quality 60, 30fps
```

### PC — klient testowy (OpenCV)
```bash
# Edytuj PI_SERVER_HOST w pliku, potem:
python3 client_t.py
```
Sterowanie klawiaturą:
- `↑↓` — pitch
- `←→` — yaw
- `< >` — roll
- `r` — reset do 0°
- `m` — tryb auto (sinusoidalne ruchy)
- `q` / `ESC` — wyjście

### Oculus Quest / przeglądarka — `vr.html`

Dwa sposoby połączenia:
1. **Przez ngrok** (działa przez internet):
   - Na RPi: uruchom `ngrok http http://localhost:8765`
   - Skopiuj fragment URL z sekcji Forwarding (np. `4b63-46-205-197-171.ngrok-free.app`)
   - Wklej do `vr.html` jako adres WebSocket
2. **Przez LAN** (mniejsza latencja):
   - Wpisz IP RPi bezpośrednio

## Protokół komunikacji

Klient → RPi (JSON):
```json
{ "type": "head_angles", "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "timestamp": 1234567890.0 }
```

RPi → Klient (JSON):
```json
{
  "type": "camera_frame",
  "timestamp": 1234567890.0,
  "frame_id": 42,
  "image": "<base64 JPEG>",
  "server_timing": { "capture_ms": 5.2, "compression_ms": 12.1, "total_processing_ms": 18.3 }
}
```

## Znane problemy / ograniczenia

- `still_configuration` na RPi — wolniejsze niż `video_configuration` (dodatkowe ~20ms na capture)
- base64 encoding dodaje ~33% rozmiaru danych względem binarnego transferu
- Serwa: mapowanie kątów VR→servo nie było w pełni przetestowane, zakres 0–360° może być błędny
- Brak reconnect logic w kliencie
- Obraz obrócony o 180° (rotate w capture_frame — zależy od montażu kamery)
