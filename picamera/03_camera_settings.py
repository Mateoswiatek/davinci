#!/usr/bin/env python3
"""
Przykład 3: Ustawienia kamery i kontrola parametrów

Kamera: Arducam pivariety [2560x800 10-bit MONO]
"""

from picamera2 import Picamera2
import time


def show_camera_info():
    """Wyświetlenie informacji o kamerze."""
    with Picamera2() as picam2:
        print("=== INFORMACJE O KAMERZE ===\n")

        # Właściwości kamery
        props = picam2.camera_properties
        print("Właściwości kamery:")
        for key, value in props.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 50 + "\n")

        # Dostępne tryby sensora
        print("Dostępne tryby sensora:")
        for i, mode in enumerate(picam2.sensor_modes):
            print(f"\n  Tryb {i}:")
            for key, value in mode.items():
                print(f"    {key}: {value}")

        print("\n" + "=" * 50 + "\n")

        # Dostępne kontrolki
        print("Dostępne kontrolki (nazwa: min, max, default):")
        controls = picam2.camera_controls
        for name, (min_val, max_val, default) in sorted(controls.items()):
            print(f"  {name}: ({min_val}, {max_val}, default={default})")


def manual_exposure_example():
    """Ręczna kontrola ekspozycji."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()

        # Poczekaj na auto-ekspozycję
        time.sleep(2)

        # Pobierz aktualne wartości
        metadata = picam2.capture_metadata()
        current_exposure = metadata.get('ExposureTime', 10000)
        current_gain = metadata.get('AnalogueGain', 1.0)

        print(f"Auto-ekspozycja: {current_exposure}us, Gain: {current_gain}")

        # Zdjęcie z auto-ekspozycją
        picam2.capture_file("exposure_auto.jpg")

        # Ustaw ręczną ekspozycję
        picam2.set_controls({
            "AeEnable": False,  # Wyłącz auto-ekspozycję
            "ExposureTime": current_exposure,
            "AnalogueGain": current_gain
        })
        time.sleep(0.5)
        picam2.capture_file("exposure_manual_same.jpg")
        print("Zapisano: exposure_manual_same.jpg (ręczna = auto)")

        # Dłuższa ekspozycja (jaśniejsze)
        picam2.set_controls({
            "ExposureTime": current_exposure * 2,
        })
        time.sleep(0.5)
        picam2.capture_file("exposure_longer.jpg")
        print("Zapisano: exposure_longer.jpg (2x dłuższa)")

        # Krótsza ekspozycja (ciemniejsze)
        picam2.set_controls({
            "ExposureTime": current_exposure // 2,
        })
        time.sleep(0.5)
        picam2.capture_file("exposure_shorter.jpg")
        print("Zapisano: exposure_shorter.jpg (2x krótsza)")


def gain_control_example():
    """Kontrola wzmocnienia (gain)."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        # Stała ekspozycja
        metadata = picam2.capture_metadata()
        base_exposure = metadata.get('ExposureTime', 10000)

        picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": base_exposure,
        })

        gains = [1.0, 2.0, 4.0, 8.0]
        for gain in gains:
            picam2.set_controls({"AnalogueGain": gain})
            time.sleep(0.3)
            filename = f"gain_{gain:.1f}.jpg"
            picam2.capture_file(filename)
            print(f"Zapisano: {filename}")


def brightness_contrast_example():
    """Kontrola jasności i kontrastu."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        # Normalne
        picam2.capture_file("bc_normal.jpg")
        print("Zapisano: bc_normal.jpg (domyślne)")

        # Jaśniej
        picam2.set_controls({"Brightness": 0.3})
        time.sleep(0.3)
        picam2.capture_file("bc_bright.jpg")
        print("Zapisano: bc_bright.jpg (jasność +0.3)")

        # Ciemniej
        picam2.set_controls({"Brightness": -0.3})
        time.sleep(0.3)
        picam2.capture_file("bc_dark.jpg")
        print("Zapisano: bc_dark.jpg (jasność -0.3)")

        # Reset jasności, wysoki kontrast
        picam2.set_controls({
            "Brightness": 0.0,
            "Contrast": 2.0
        })
        time.sleep(0.3)
        picam2.capture_file("bc_high_contrast.jpg")
        print("Zapisano: bc_high_contrast.jpg (kontrast 2.0)")

        # Niski kontrast
        picam2.set_controls({"Contrast": 0.5})
        time.sleep(0.3)
        picam2.capture_file("bc_low_contrast.jpg")
        print("Zapisano: bc_low_contrast.jpg (kontrast 0.5)")


def sharpness_example():
    """Kontrola ostrości."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        sharpness_values = [0.0, 1.0, 2.0, 4.0, 8.0]
        for sharp in sharpness_values:
            picam2.set_controls({"Sharpness": sharp})
            time.sleep(0.3)
            filename = f"sharpness_{sharp:.1f}.jpg"
            picam2.capture_file(filename)
            print(f"Zapisano: {filename}")


def white_balance_example():
    """
    Kontrola balansu bieli.
    UWAGA: Dla kamery monochromatycznej nie ma sensu, ale działa dla RGB.
    """
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        # Auto white balance
        picam2.set_controls({"AwbEnable": True})
        time.sleep(0.5)
        picam2.capture_file("wb_auto.jpg")
        print("Zapisano: wb_auto.jpg")

        # Ręczny WB - różne temperatury kolorów (ColourGains)
        # ColourGains: (red_gain, blue_gain)
        wb_presets = {
            "tungsten": (0.9, 1.9),      # żarówka
            "daylight": (1.3, 1.5),      # dzienne
            "cloudy": (1.5, 1.3),        # pochmurno
            "fluorescent": (1.1, 1.7),   # świetlówka
        }

        picam2.set_controls({"AwbEnable": False})
        time.sleep(0.2)

        for name, (red, blue) in wb_presets.items():
            picam2.set_controls({"ColourGains": (red, blue)})
            time.sleep(0.3)
            filename = f"wb_{name}.jpg"
            picam2.capture_file(filename)
            print(f"Zapisano: {filename}")


def noise_reduction_example():
    """Ustawienia redukcji szumu."""
    with Picamera2() as picam2:
        import libcamera

        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        nr_modes = {
            "off": libcamera.controls.draft.NoiseReductionModeEnum.Off,
            "fast": libcamera.controls.draft.NoiseReductionModeEnum.Fast,
            "high_quality": libcamera.controls.draft.NoiseReductionModeEnum.HighQuality,
            "minimal": libcamera.controls.draft.NoiseReductionModeEnum.Minimal,
        }

        for name, mode in nr_modes.items():
            picam2.set_controls({"NoiseReductionMode": mode})
            time.sleep(0.3)
            filename = f"nr_{name}.jpg"
            picam2.capture_file(filename)
            print(f"Zapisano: {filename}")


def transform_example():
    """Transformacje obrazu (flip, rotate)."""
    from libcamera import Transform

    with Picamera2() as picam2:
        # Normalny
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(1)
        picam2.capture_file("transform_normal.jpg")
        print("Zapisano: transform_normal.jpg")
        picam2.stop()

        # Flip poziomy
        config = picam2.create_still_configuration(
            transform=Transform(hflip=True)
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.5)
        picam2.capture_file("transform_hflip.jpg")
        print("Zapisano: transform_hflip.jpg")
        picam2.stop()

        # Flip pionowy
        config = picam2.create_still_configuration(
            transform=Transform(vflip=True)
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.5)
        picam2.capture_file("transform_vflip.jpg")
        print("Zapisano: transform_vflip.jpg")
        picam2.stop()

        # Obrót 180°
        config = picam2.create_still_configuration(
            transform=Transform(hflip=True, vflip=True)
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.5)
        picam2.capture_file("transform_180.jpg")
        print("Zapisano: transform_180.jpg")


def framerate_control_example():
    """Kontrola liczby klatek na sekundę."""
    with Picamera2() as picam2:
        framerates = [10, 15, 20, 30]

        for fps in framerates:
            frame_duration = int(1_000_000 / fps)  # mikrosekundy

            config = picam2.create_preview_configuration(
                controls={
                    "FrameDurationLimits": (frame_duration, frame_duration)
                }
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(0.5)

            # Sprawdź rzeczywisty FPS
            metadata = picam2.capture_metadata()
            actual_duration = metadata.get('FrameDuration', frame_duration)
            actual_fps = 1_000_000 / actual_duration

            print(f"Requested: {fps} FPS, Actual: {actual_fps:.1f} FPS")

            picam2.stop()


def exposure_bracketing_example():
    """
    Bracketing ekspozycji - seria zdjęć z różną ekspozycją.
    Przydatne do HDR.
    """
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        # Pobierz bazową ekspozycję
        metadata = picam2.capture_metadata()
        base_exp = metadata.get('ExposureTime', 10000)

        # Wyłącz auto-ekspozycję
        picam2.set_controls({"AeEnable": False})

        # Bracketing: -2EV, -1EV, 0EV, +1EV, +2EV
        ev_stops = [-2, -1, 0, 1, 2]

        for ev in ev_stops:
            # 1 EV = 2x ekspozycja
            exposure = int(base_exp * (2 ** ev))
            exposure = max(100, min(exposure, 200_000))  # clamp

            picam2.set_controls({"ExposureTime": exposure})
            time.sleep(0.3)

            filename = f"bracket_ev{ev:+d}.jpg"
            picam2.capture_file(filename)
            print(f"Zapisano: {filename} (exp: {exposure}us)")


if __name__ == "__main__":
    import sys

    examples = {
        "info": ("Informacje o kamerze", show_camera_info),
        "exposure": ("Kontrola ekspozycji", manual_exposure_example),
        "gain": ("Kontrola gain", gain_control_example),
        "brightness": ("Jasność i kontrast", brightness_contrast_example),
        "sharpness": ("Ostrość", sharpness_example),
        "wb": ("Balans bieli", white_balance_example),
        "noise": ("Redukcja szumu", noise_reduction_example),
        "transform": ("Transformacje", transform_example),
        "fps": ("Kontrola FPS", framerate_control_example),
        "bracket": ("Bracketing ekspozycji", exposure_bracketing_example),
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
        print("Użycie: python 03_camera_settings.py [przykład]")
        print("\nDostępne przykłady:")
        for key, (name, _) in examples.items():
            print(f"  {key}: {name}")

        print("\nUruchomienie wszystkich przykładów:")

        for key, (name, func) in examples.items():
            print(f"\n{'=' * 50}")
            print(f"=== {name} ===")
            print('=' * 50 + "\n")
            try:
                func()
            except Exception as e:
                print(f"BŁĄD: {e}")