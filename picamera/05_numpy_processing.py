#!/usr/bin/env python3
"""
Przykład 5: Przetwarzanie obrazu z NumPy i OpenCV

Kamera: Arducam pivariety [2560x800 10-bit MONO]
"""

from picamera2 import Picamera2
import numpy as np
from PIL import Image
import time

# Opcjonalnie OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("OpenCV nie zainstalowane - niektóre przykłady niedostępne")


def capture_as_numpy():
    """Podstawowe przechwycenie jako numpy array."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        # Przechwycenie jako numpy array
        array = picam2.capture_array("main")

        print(f"Shape: {array.shape}")
        print(f"Dtype: {array.dtype}")
        print(f"Min: {array.min()}, Max: {array.max()}")

        return array


def manual_image_processing():
    """Ręczne przetwarzanie obrazu z numpy."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        array = picam2.capture_array("main")

        # Konwersja do grayscale jeśli RGB
        if len(array.shape) == 3:
            # Metoda luminancji
            gray = (0.299 * array[:, :, 0] +
                   0.587 * array[:, :, 1] +
                   0.114 * array[:, :, 2]).astype(np.uint8)
        else:
            gray = array

        print(f"Grayscale shape: {gray.shape}")

        # Normalizacja kontrastu
        p_low, p_high = np.percentile(gray, [2, 98])
        normalized = np.clip((gray - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)

        # Inwersja
        inverted = 255 - gray

        # Progowanie
        threshold = np.mean(gray)
        binary = (gray > threshold).astype(np.uint8) * 255

        # Zapisz wyniki
        Image.fromarray(gray, mode='L').save("np_gray.png")
        Image.fromarray(normalized, mode='L').save("np_normalized.png")
        Image.fromarray(inverted, mode='L').save("np_inverted.png")
        Image.fromarray(binary, mode='L').save("np_binary.png")

        print("Zapisano: np_gray.png, np_normalized.png, np_inverted.png, np_binary.png")


def edge_detection_sobel():
    """Detekcja krawędzi metodą Sobela (bez OpenCV)."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        array = picam2.capture_array("main")

        # Konwersja do grayscale
        if len(array.shape) == 3:
            gray = np.mean(array, axis=2).astype(np.float32)
        else:
            gray = array.astype(np.float32)

        # Kernele Sobela
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)

        # Konwolucja (prosta implementacja)
        def convolve2d(image, kernel):
            h, w = image.shape
            kh, kw = kernel.shape
            pad_h, pad_w = kh // 2, kw // 2

            # Padding
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

            # Konwolucja
            result = np.zeros_like(image)
            for i in range(h):
                for j in range(w):
                    result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)

            return result

        # Gradienty
        grad_x = convolve2d(gray, sobel_x)
        grad_y = convolve2d(gray, sobel_y)

        # Magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

        Image.fromarray(magnitude, mode='L').save("np_edges_sobel.png")
        print("Zapisano: np_edges_sobel.png")


def histogram_equalization():
    """Wyrównanie histogramu (bez OpenCV)."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        array = picam2.capture_array("main")

        # Konwersja do grayscale
        if len(array.shape) == 3:
            gray = np.mean(array, axis=2).astype(np.uint8)
        else:
            gray = array

        # Histogram
        hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))

        # CDF (Cumulative Distribution Function)
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]

        # Mapowanie
        equalized = cdf_normalized[gray].astype(np.uint8)

        # Zapisz
        Image.fromarray(gray, mode='L').save("np_original.png")
        Image.fromarray(equalized, mode='L').save("np_equalized.png")

        print("Zapisano: np_original.png, np_equalized.png")


def gaussian_blur():
    """Rozmycie Gaussowskie (bez OpenCV)."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        array = picam2.capture_array("main")

        # Konwersja do grayscale
        if len(array.shape) == 3:
            gray = np.mean(array, axis=2).astype(np.float32)
        else:
            gray = array.astype(np.float32)

        # Kernel Gaussowski 5x5
        sigma = 1.0
        size = 5
        x = np.arange(size) - size // 2
        kernel_1d = np.exp(-x**2 / (2 * sigma**2))
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()

        # Konwolucja z scipy lub ręcznie
        try:
            from scipy.ndimage import convolve
            blurred = convolve(gray, kernel_2d)
        except ImportError:
            # Prosta implementacja
            h, w = gray.shape
            kh, kw = kernel_2d.shape
            pad_h, pad_w = kh // 2, kw // 2
            padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            blurred = np.zeros_like(gray)
            for i in range(h):
                for j in range(w):
                    blurred[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel_2d)

        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        Image.fromarray(blurred, mode='L').save("np_blurred.png")
        print("Zapisano: np_blurred.png")


# ============ PRZYKŁADY Z OPENCV ============

def opencv_basic_processing():
    """Podstawowe przetwarzanie z OpenCV."""
    if not HAS_OPENCV:
        print("OpenCV nie zainstalowane!")
        return

    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        array = picam2.capture_array("main")

        # Konwersja BGR -> Grayscale
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

        # Różne operacje
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Zapisz
        cv2.imwrite("cv_gray.png", gray)
        cv2.imwrite("cv_blur.png", blur)
        cv2.imwrite("cv_edges.png", edges)
        cv2.imwrite("cv_binary.png", binary)

        print("Zapisano: cv_gray.png, cv_blur.png, cv_edges.png, cv_binary.png")


def opencv_contour_detection():
    """Detekcja konturów z OpenCV."""
    if not HAS_OPENCV:
        print("OpenCV nie zainstalowane!")
        return

    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        array = picam2.capture_array("main")

        # Grayscale
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

        # Progowanie
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Znajdź kontury
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Znaleziono {len(contours)} konturów")

        # Rysuj kontury
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

        cv2.imwrite("cv_contours.png", output)
        print("Zapisano: cv_contours.png")


def opencv_morphological_operations():
    """Operacje morfologiczne z OpenCV."""
    if not HAS_OPENCV:
        print("OpenCV nie zainstalowane!")
        return

    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        array = picam2.capture_array("main")
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Element strukturalny
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Operacje morfologiczne
        eroded = cv2.erode(binary, kernel, iterations=1)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite("cv_eroded.png", eroded)
        cv2.imwrite("cv_dilated.png", dilated)
        cv2.imwrite("cv_opened.png", opened)
        cv2.imwrite("cv_closed.png", closed)

        print("Zapisano: cv_eroded.png, cv_dilated.png, cv_opened.png, cv_closed.png")


def realtime_processing_loop():
    """Ciągłe przetwarzanie w czasie rzeczywistym."""
    with Picamera2() as picam2:
        config = picam2.create_preview_configuration(
            main={"size": (640, 200)}
        )
        picam2.configure(config)
        picam2.start()

        print("Przetwarzanie w czasie rzeczywistym (20 klatek)...")
        print("Format: [frame] mean brightness | processing time")

        for i in range(20):
            start_time = time.time()

            # Przechwycenie
            array = picam2.capture_array("main")

            # Przetwarzanie
            if len(array.shape) == 3:
                gray = np.mean(array, axis=2)
            else:
                gray = array

            mean_brightness = np.mean(gray)

            # Prosta detekcja - czy obraz jest zbyt ciemny/jasny
            if mean_brightness < 50:
                status = "CIEMNY"
            elif mean_brightness > 200:
                status = "JASNY"
            else:
                status = "OK"

            process_time = (time.time() - start_time) * 1000

            print(f"[{i:3d}] {mean_brightness:6.1f} | {process_time:5.1f}ms | {status}")


def capture_with_metadata():
    """Przechwycenie z pełnymi metadanymi."""
    with Picamera2() as picam2:
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        with picam2.captured_request() as request:
            # Obraz
            array = request.make_array("main")

            # Metadane
            metadata = request.get_metadata()

            print("=== METADANE ZDJĘCIA ===")
            for key, value in sorted(metadata.items()):
                print(f"  {key}: {value}")

            # Zapisz obraz
            Image.fromarray(array).save("np_with_metadata.jpg")
            print("\nZapisano: np_with_metadata.jpg")


if __name__ == "__main__":
    import sys

    examples = {
        "basic": ("Podstawowe numpy", capture_as_numpy),
        "process": ("Przetwarzanie ręczne", manual_image_processing),
        "sobel": ("Detekcja krawędzi Sobel", edge_detection_sobel),
        "histeq": ("Wyrównanie histogramu", histogram_equalization),
        "blur": ("Rozmycie Gaussowskie", gaussian_blur),
        "cv_basic": ("OpenCV podstawowe", opencv_basic_processing),
        "cv_contours": ("OpenCV kontury", opencv_contour_detection),
        "cv_morph": ("OpenCV morfologia", opencv_morphological_operations),
        "realtime": ("Przetwarzanie RT", realtime_processing_loop),
        "metadata": ("Z metadanymi", capture_with_metadata),
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
        print("Użycie: python 05_numpy_processing.py [przykład]")
        print("\nDostępne przykłady:")
        for key, (name, _) in examples.items():
            print(f"  {key}: {name}")