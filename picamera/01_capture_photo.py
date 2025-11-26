#!/usr/bin/env python3
"""
Przykład 1: Proste robienie zdjęcia z Picamera2

Kamera: Arducam pivariety [2560x800 10-bit MONO]
"""

from picamera2 import Picamera2
import time


def capture_simple():
    """Najprostszy sposób na zrobienie zdjęcia."""
    picam2 = Picamera2()

    # Automatyczna konfiguracja dla zdjęć
    config = picam2.create_still_configuration()
    picam2.configure(config)

    picam2.start()
    time.sleep(2)  # Czas na auto-ekspozycję

    picam2.capture_file("photo_simple.jpg")
    print("Zapisano: photo_simple.jpg")

    picam2.stop()
    picam2.close()


def capture_with_custom_resolution():
    """Zdjęcie z określoną rozdzielczością i poprawną ekspozycją."""
    with Picamera2() as picam2:
        # SPOSÓB 1: Najpierw preview, potem przełącz na full res
        # Kamera ustala ekspozycję na preview, potem przełącza tryb

        # Start w preview (mniejsza rozdzielczość) - AE działa lepiej
        preview_config = picam2.create_preview_configuration(
            main={"size": (1280, 400)}
        )
        picam2.configure(preview_config)
        picam2.start()
        time.sleep(2)  # AE stabilizacja

        # Pobierz ustalone wartości ekspozycji
        metadata = picam2.capture_metadata()
        exposure = metadata.get('ExposureTime', 10000)
        gain = metadata.get('AnalogueGain', 1.0)
        print(f"Auto-ekspozycja: {exposure}us, gain: {gain:.2f}")

        # Konfiguracja full res z RĘCZNĄ ekspozycją
        # raw=None wyłącza raw stream (unika błędu z MONO_PISP_COMP1)
        still_config = picam2.create_still_configuration(
            main={
                "size": (2560, 800),
                "format": "BGR888"
            },
            raw=None,  # WAŻNE: wyłącz raw dla tej kamery
            controls={
                "AeEnable": False,
                "ExposureTime": exposure,
                "AnalogueGain": gain,
            }
        )

        # Przełącz i zrób zdjęcie
        picam2.switch_mode_and_capture_file(still_config, "photo_full_res.jpg")
        print("Zapisano: photo_full_res.jpg (pełna rozdzielczość)")


def capture_as_png():
    """Zdjęcie w formacie PNG (bezstratny)."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()

        time.sleep(2)
        picam2.capture_file("photo.png")
        print("Zapisano: photo.png (bezstratny format)")


if __name__ == "__main__":
    print("=== Przykłady robienia zdjęć z Picamera2 ===\n")

    print("1. Proste zdjęcie...")
    # capture_simple()

    print("\n5. Pełna rozdzielczość...")
    capture_with_custom_resolution()

    print("\n6. Format PNG...")
    capture_as_png()

    print("\n=== Wszystkie przykłady zakończone ===")