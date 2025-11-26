#!/usr/bin/env python3
"""
Przykład 4: Specyficzne dla kamery monochromatycznej

Kamera: Arducam pivariety [2560x800 10-bit MONO]

Dostępne tryby:
- R8: 1280x400, 2560x720, 2560x800 @ 30fps
- R10_CSI2P: 640x200, 1280x400, 2560x720, 2560x800 @ 30fps
"""

from picamera2 import Picamera2
import numpy as np
from PIL import Image
import time


def show_mono_camera_modes():
    """Wyświetlenie dostępnych trybów dla kamery mono."""
    with Picamera2() as picam2:
        print("=== TRYBY KAMERY MONO ===\n")

        print("Właściwości kamery:")
        props = picam2.camera_properties
        print(f"  Model: {props.get('Model', 'Unknown')}")

        # Sprawdź czy mono
        cfa = props.get('ColorFilterArrangement')
        if cfa is None or cfa == 0:
            print("  Typ: MONO (monochromatyczna)")
        else:
            print(f"  Typ: Bayer (CFA: {cfa})")

        print("\nDostępne tryby sensora:")
        for i, mode in enumerate(picam2.sensor_modes):
            fmt = mode.get('format', 'Unknown')
            size = mode.get('size', (0, 0))
            fps = mode.get('fps', 0)
            bit_depth = mode.get('bit_depth', 'N/A')

            print(f"\n  Tryb {i}:")
            print(f"    Format: {fmt}")
            print(f"    Rozmiar: {size[0]}x{size[1]}")
            print(f"    FPS: {fps}")
            print(f"    Bit depth: {bit_depth}")

            # Dla mono - wyświetl dodatkowe info
            if fmt.startswith('R'):
                if fmt == 'R8':
                    print("    -> 8-bit mono (1 bajt/pixel)")
                elif fmt == 'R10' or fmt == 'R10_CSI2P':
                    print("    -> 10-bit mono (2 bajty/pixel, wartości 0-1023)")
                elif fmt == 'R12':
                    print("    -> 12-bit mono (2 bajty/pixel, wartości 0-4095)")


def capture_mono_raw():
    """Przechwycenie surowych danych mono."""
    with Picamera2() as picam2:
        # Konfiguracja z raw stream
        config = picam2.create_still_configuration(
            main={"format": "BGR888"},
            raw={}  # automatyczny wybór formatu raw
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        # Przechwycenie z raw
        with picam2.captured_request() as request:
            # Główny strumień (przetworzone)
            main_array = request.make_array("main")
            print(f"Main array: {main_array.shape}, dtype: {main_array.dtype}")

            # Raw strumień (surowe mono)
            try:
                raw_array = request.make_array("raw")
                print(f"Raw array: {raw_array.shape}, dtype: {raw_array.dtype}")
                print(f"Raw min: {raw_array.min()}, max: {raw_array.max()}")

                # Zapisz raw jako PNG (po normalizacji)
                if raw_array.dtype == np.uint16:
                    # 10-bit lub 12-bit - normalizuj do 8-bit
                    max_val = raw_array.max()
                    if max_val > 0:
                        normalized = (raw_array.astype(np.float32) / max_val * 255).astype(np.uint8)
                    else:
                        normalized = np.zeros_like(raw_array, dtype=np.uint8)

                    img = Image.fromarray(normalized, mode='L')
                    img.save("mono_raw_normalized.png")
                    print("Zapisano: mono_raw_normalized.png")

                    # Zapisz też surowe 16-bit
                    img16 = Image.fromarray(raw_array, mode='I;16')
                    img16.save("mono_raw_16bit.png")
                    print("Zapisano: mono_raw_16bit.png")
                else:
                    # 8-bit
                    img = Image.fromarray(raw_array, mode='L')
                    img.save("mono_raw_8bit.png")
                    print("Zapisano: mono_raw_8bit.png")

            except Exception as e:
                print(f"Nie można pobrać raw: {e}")


def capture_mono_10bit():
    """Specyficzne przechwycenie 10-bit mono."""
    with Picamera2() as picam2:
        # Sprawdź dostępność R10
        r10_mode = None
        for mode in picam2.sensor_modes:
            if mode.get('format') in ['R10', 'R10_CSI2P']:
                r10_mode = mode
                break

        if r10_mode is None:
            print("Brak trybu R10 dla tej kamery")
            return

        print(f"Używam trybu: {r10_mode}")

        config = picam2.create_still_configuration(
            main={"format": "BGR888"},
            raw={"format": r10_mode.get('format'), "size": r10_mode.get('size')},
            sensor={"bit_depth": 10}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        with picam2.captured_request() as request:
            raw_array = request.make_array("raw")
            print(f"10-bit raw: {raw_array.shape}, dtype: {raw_array.dtype}")
            print(f"Zakres wartości: {raw_array.min()} - {raw_array.max()}")

            # 10-bit = wartości 0-1023
            # Normalizacja do 8-bit
            mono_8bit = (raw_array / 4).astype(np.uint8)  # 1024 / 4 = 256

            img = Image.fromarray(mono_8bit, mode='L')
            img.save("mono_10bit_to_8bit.png")
            print("Zapisano: mono_10bit_to_8bit.png")

            # Zachowaj pełną precyzję (16-bit PNG)
            # Skaluj do 16-bit dla zapisu
            mono_16bit = (raw_array << 6).astype(np.uint16)  # shift left 6 bits
            img16 = Image.fromarray(mono_16bit, mode='I;16')
            img16.save("mono_10bit_full.png")
            print("Zapisano: mono_10bit_full.png (pełna precyzja)")


def capture_all_resolutions():
    """Przechwycenie we wszystkich dostępnych rozdzielczościach."""
    with Picamera2() as picam2:
        modes = picam2.sensor_modes

        for i, mode in enumerate(modes):
            size = mode.get('size', (640, 480))
            fmt = mode.get('format', 'Unknown')

            print(f"\nTryb {i}: {fmt} {size[0]}x{size[1]}")

            try:
                config = picam2.create_still_configuration(
                    main={"size": size, "format": "BGR888"}
                )
                picam2.configure(config)
                picam2.start()
                time.sleep(1)

                filename = f"mono_{size[0]}x{size[1]}.jpg"
                picam2.capture_file(filename)
                print(f"  Zapisano: {filename}")

                picam2.stop()
            except Exception as e:
                print(f"  BŁĄD: {e}")


def continuous_mono_capture():
    """Ciągłe przechwytywanie mono do przetwarzania."""
    with Picamera2() as picam2:
        config = picam2.create_preview_configuration(
            main={"format": "YUV420", "size": (1280, 400)}
        )
        picam2.configure(config)
        picam2.start()

        print("Ciągłe przechwytywanie (10 klatek)...")

        for i in range(10):
            # Pobierz obraz
            array = picam2.capture_array("main")

            # Dla YUV420 - Y channel to luminancja (mono)
            # Format YUV420: Y plane jest na początku
            h, w = array.shape[:2]
            y_plane = array[:h, :w, 0] if len(array.shape) == 3 else array

            # Statystyki
            mean_val = np.mean(y_plane)
            std_val = np.std(y_plane)

            print(f"Klatka {i}: mean={mean_val:.1f}, std={std_val:.1f}")


def histogram_analysis():
    """Analiza histogramu obrazu mono."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        array = picam2.capture_array("main")

        # Konwersja do grayscale jeśli RGB
        if len(array.shape) == 3:
            gray = np.mean(array, axis=2).astype(np.uint8)
        else:
            gray = array

        # Histogram
        hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))

        # Statystyki
        print("=== ANALIZA HISTOGRAMU ===")
        print(f"Min: {gray.min()}")
        print(f"Max: {gray.max()}")
        print(f"Mean: {gray.mean():.1f}")
        print(f"Std: {gray.std():.1f}")
        print(f"Median: {np.median(gray):.1f}")

        # Percentyle
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"Percentyl {p}%: {np.percentile(gray, p):.1f}")

        # Zapisz histogram jako tekst
        with open("histogram.txt", "w") as f:
            f.write("bin,count\n")
            for b, c in zip(bins[:-1], hist):
                f.write(f"{int(b)},{c}\n")
        print("\nZapisano: histogram.txt")


def save_mono_dng():
    """Zapisanie obrazu mono jako DNG (Digital Negative)."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration(raw={})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        with picam2.captured_request() as request:
            try:
                request.save_dng("mono_image.dng")
                print("Zapisano: mono_image.dng")
                print("DNG można otworzyć w RawTherapee, darktable, Lightroom")
            except Exception as e:
                print(f"Nie można zapisać DNG: {e}")


def optimize_for_mono():
    """
    Optymalizacja ustawień dla kamery mono.
    Wyłączamy niepotrzebne przetwarzanie kolorów.
    """
    with Picamera2() as picam2:
        config = picam2.create_preview_configuration(
            main={"format": "YUV420", "size": (1280, 400)}
        )

        # Wyłącz AWB i color processing (nie potrzebne dla mono)
        config["controls"] = {
            "AwbEnable": False,  # Brak balansu bieli
        }

        picam2.configure(config)
        picam2.start()

        # Dodatkowe ustawienia optymalizacyjne
        picam2.set_controls({
            "Saturation": 0.0,  # Brak saturacji (i tak mono)
        })

        time.sleep(1)

        # Test wydajności
        import time as t
        start = t.time()
        frames = 100

        for _ in range(frames):
            array = picam2.capture_array("main")

        elapsed = t.time() - start
        fps = frames / elapsed

        print(f"Wydajność: {fps:.1f} FPS ({frames} klatek w {elapsed:.2f}s)")


if __name__ == "__main__":
    import sys

    examples = {
        "modes": ("Pokaż tryby kamery", show_mono_camera_modes),
        "raw": ("Przechwycenie raw", capture_mono_raw),
        "10bit": ("Przechwycenie 10-bit", capture_mono_10bit),
        "resolutions": ("Wszystkie rozdzielczości", capture_all_resolutions),
        "continuous": ("Ciągłe przechwytywanie", continuous_mono_capture),
        "histogram": ("Analiza histogramu", histogram_analysis),
        "dng": ("Zapisz DNG", save_mono_dng),
        "optimize": ("Optymalizacja dla mono", optimize_for_mono),
    }

    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name in examples:
            name, func = examples[example_name]
            print(f"=== {name} ===\n")
            func()
        else:
            print(f"Nieznany przykład: {example_name}")
            print(f"Dostępne: {', '.join(examples.keys())}")
    else:
        print("Użycie: python 04_mono_camera.py [przykład]")
        print("\nDostępne przykłady:")
        for key, (name, _) in examples.items():
            print(f"  {key}: {name}")

        print("\nUruchom 'modes' aby zobaczyć dostępne tryby kamery")