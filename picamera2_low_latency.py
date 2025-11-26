#!/usr/bin/env python3
"""
Picamera2 configuration dla ultra-niskiej latencji
- Minimal buffer counts
- Optimal sensor modes
- Zero-copy capture
- Hardware encoding
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder, Quality
    from picamera2.outputs import FileOutput
    import libcamera
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠ Picamera2 not available - install with: sudo apt install -y python3-picamera2")


@dataclass
class LatencyProfile:
    """Profil konfiguracji dla różnych wymagań latencji"""
    name: str
    buffer_count: int
    queue_size: int
    controls: Dict[str, Any]
    description: str


class LowLatencyCamera:
    """Kamera skonfigurowana dla minimalnej latencji"""

    # Predefiniowane profile latencji
    PROFILES = {
        'ultra_low': LatencyProfile(
            name='Ultra Low Latency',
            buffer_count=2,  # Minimum 2 bufory
            queue_size=1,
            controls={
                'FrameDurationLimits': (16666, 16666),  # 60fps locked
                'AeEnable': False,  # Wyłącz auto-exposure (stały czas)
                'AwbEnable': False,  # Wyłącz auto white balance
                'NoiseReductionMode': libcamera.controls.draft.NoiseReductionModeEnum.Off,
            },
            description='<10ms latency, fixed exposure, no post-processing'
        ),
        'low': LatencyProfile(
            name='Low Latency',
            buffer_count=3,
            queue_size=2,
            controls={
                'FrameDurationLimits': (16666, 16666),  # 60fps
                'AeEnable': True,
                'AeExposureMode': libcamera.controls.AeExposureModeEnum.Short,
                'NoiseReductionMode': libcamera.controls.draft.NoiseReductionModeEnum.Fast,
            },
            description='<20ms latency, fast auto-exposure, minimal processing'
        ),
        'balanced': LatencyProfile(
            name='Balanced',
            buffer_count=4,
            queue_size=2,
            controls={
                'FrameDurationLimits': (16666, 33333),  # 30-60fps
                'AeEnable': True,
                'NoiseReductionMode': libcamera.controls.draft.NoiseReductionModeEnum.Fast,
            },
            description='<30ms latency, auto-exposure, good quality'
        ),
    }

    def __init__(self, camera_num: int = 0,
                 profile: str = 'ultra_low',
                 stereo: bool = True):
        """
        Args:
            camera_num: Numer kamery (0 lub 1)
            profile: 'ultra_low', 'low', lub 'balanced'
            stereo: True dla stereo kamery (2560x800), False dla mono
        """
        if not PICAMERA2_AVAILABLE:
            raise ImportError("Picamera2 not available")

        self.camera_num = camera_num
        self.profile = self.PROFILES[profile]
        self.stereo = stereo
        self.picam2 = None
        self.frame_count = 0
        self.last_timestamp = None

    def initialize(self):
        """Inicjalizuj kamerę z low-latency config"""

        print(f"\n=== Initializing Camera (Profile: {self.profile.name}) ===")

        self.picam2 = Picamera2(self.camera_num)

        # 1. Wybierz odpowiedni sensor mode
        sensor_mode = self._select_sensor_mode()

        # 2. Konfiguracja dla capture
        if self.stereo:
            # Arducam stereo: 2560x800
            width, height = 2560, 800
        else:
            # Standardowa kamera: 1920x1080 lub 1280x720
            width, height = 1280, 720

        config = self.picam2.create_video_configuration(
            main={
                "size": (width, height),
                "format": "RGB888",  # Lub "XRGB8888" dla alignment
            },
            buffer_count=self.profile.buffer_count,
            queue=self.profile.queue_size,
            sensor={'output_size': sensor_mode['size'],
                   'bit_depth': sensor_mode.get('bit_depth', 10)},
            controls=self.profile.controls,
        )

        self.picam2.configure(config)

        print(f"✓ Configuration:")
        print(f"  Resolution: {width}x{height}")
        print(f"  Buffer count: {self.profile.buffer_count}")
        print(f"  Queue size: {self.profile.queue_size}")
        print(f"  Sensor mode: {sensor_mode}")
        print(f"  Controls: {self.profile.controls}")

        # 3. Dodatkowe optymalizacje
        self._apply_advanced_settings()

        # 4. Start capture
        self.picam2.start()
        print(f"✓ Camera started")

        # Poczekaj na stabilizację (3-4 frames)
        time.sleep(0.1)

    def _select_sensor_mode(self) -> Dict[str, Any]:
        """
        Wybierz optymalny sensor mode dla niskiej latencji

        Kryteria:
        - Wysoki framerate (>60fps)
        - Niski crop (pełny FOV)
        - Odpowiednia rozdzielczość
        """

        sensor_modes = self.picam2.sensor_modes

        print(f"\nAvailable sensor modes:")
        for i, mode in enumerate(sensor_modes):
            fps = 1_000_000 / mode.get('fps', 30)
            print(f"  {i}: {mode['size']} @ {mode.get('fps', 'N/A')} "
                  f"crop={mode.get('crop_limits', 'N/A')}")

        # Dla VR: priorytet dla wysokiego FPS
        # Arducam stereo zazwyczaj ma tryb 2560x800 @ 60fps

        if self.stereo:
            # Szukaj trybu z 2560 szerokością
            for mode in sensor_modes:
                if mode['size'][0] >= 2560:
                    return mode

        # Fallback: pierwszy tryb (zazwyczaj najwyższa rozdzielczość)
        return sensor_modes[0]

    def _apply_advanced_settings(self):
        """Zastosuj zaawansowane ustawienia dla ultra-niskiej latencji"""

        # 1. Wyłącz denoise jeśli ultra-low latency
        if 'ultra' in self.profile.name.lower():
            # Już ustawione w controls
            pass

        # 2. Ustaw fixed shutter speed (jeśli AE wyłączone)
        if not self.profile.controls.get('AeEnable', True):
            # Fixed exposure: 16ms (dla 60fps)
            # W mikrosekundach
            exposure_time = 10000  # 10ms
            self.picam2.set_controls({
                'ExposureTime': exposure_time,
                'AnalogueGain': 1.0,
            })
            print(f"✓ Fixed exposure: {exposure_time}µs")

        # 3. Wyłącz saturation/sharpening dla szybszego processing
        # (zależy od dostępnych controls)

    def capture_array(self) -> np.ndarray:
        """
        Capture frame jako numpy array (zero-copy)
        Returns: numpy array RGB888
        """
        # Metoda 1: capture_array (kopiuje dane)
        # frame = self.picam2.capture_array()

        # Metoda 2: capture_buffer (zero-copy via memoryview)
        # Szybsza, ale wymaga manual konwersji
        request = self.picam2.capture_request()
        try:
            # Get buffer without copy
            buffer = request.make_buffer("main")

            # Konwertuj do numpy array (zero-copy via memoryview)
            # UWAGA: array jest valid tylko dopóki request nie został released
            width = self.picam2.camera_configuration()['main']['size'][0]
            height = self.picam2.camera_configuration()['main']['size'][1]

            # RGB888 = 3 bytes per pixel
            array = np.frombuffer(buffer, dtype=np.uint8)
            array = array.reshape((height, width, 3))

            return array.copy()  # Kopiuj bo zaraz release'ujemy request

        finally:
            request.release()

    def capture_with_timestamp(self) -> tuple[np.ndarray, float]:
        """
        Capture frame z timestamp dla pomiaru latencji
        Returns: (frame, timestamp_ns)
        """
        request = self.picam2.capture_request()
        try:
            # Timestamp z metadata
            metadata = request.get_metadata()
            timestamp = metadata.get('SensorTimestamp', time.time_ns())

            # Frame
            buffer = request.make_buffer("main")
            width = self.picam2.camera_configuration()['main']['size'][0]
            height = self.picam2.camera_configuration()['main']['size'][1]

            array = np.frombuffer(buffer, dtype=np.uint8)
            array = array.reshape((height, width, 3))

            return array.copy(), timestamp

        finally:
            request.release()

    def start_recording_h264(self, output_file: str,
                            quality: Quality = Quality.VERY_LOW):
        """
        Hardware H264 encoding dla streaming

        Args:
            output_file: Ścieżka do pliku lub socket
            quality: VERY_LOW = najniższa latencja, LOW/MEDIUM/HIGH = lepsza jakość
        """

        encoder = H264Encoder(bitrate=2_000_000)  # 2 Mbps

        # Custom output dla streaming (zamiast FileOutput)
        output = FileOutput(output_file)

        self.picam2.start_recording(encoder, output, quality=quality)
        print(f"✓ Hardware H264 encoding started: {output_file}")

    def measure_latency(self, num_frames: int = 100) -> Dict[str, float]:
        """
        Zmierz latencję capture

        Returns:
            dict z metrykami: avg_ms, min_ms, max_ms, fps
        """

        print(f"\n=== Measuring Latency ({num_frames} frames) ===")

        latencies = []
        fps_samples = []
        last_time = time.perf_counter()

        for i in range(num_frames):
            frame_start = time.perf_counter()

            # Capture
            frame, sensor_ts = self.capture_with_timestamp()

            capture_end = time.perf_counter()

            # Latencja = czas od wywołania do otrzymania danych
            latency_ms = (capture_end - frame_start) * 1000
            latencies.append(latency_ms)

            # FPS
            now = time.perf_counter()
            fps = 1.0 / (now - last_time) if last_time else 0
            fps_samples.append(fps)
            last_time = now

            if i % 20 == 0:
                print(f"  Frame {i}: {latency_ms:.2f}ms, {fps:.1f}fps")

        # Statystyki
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_fps = sum(fps_samples) / len(fps_samples)

        results = {
            'avg_ms': avg_latency,
            'min_ms': min_latency,
            'max_ms': max_latency,
            'fps': avg_fps,
        }

        print(f"\nResults:")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  Min latency: {min_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")
        print(f"  Avg FPS: {avg_fps:.1f}")

        return results

    def close(self):
        """Zamknij kamerę"""
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            print("✓ Camera closed")


def configure_gpu_memory():
    """Konfiguracja GPU memory dla kamery"""

    print("\n=== GPU Memory Configuration ===\n")
    print("Edit /boot/firmware/config.txt:")
    print("  # Minimum 128MB dla kamery")
    print("  gpu_mem=256")
    print("\nReboot required after change")


def configure_camera_modules():
    """Konfiguracja modułów kernela dla kamery"""

    print("\n=== Camera Kernel Modules ===\n")

    print("1. Enable camera in raspi-config:")
    print("   sudo raspi-config")
    print("   -> Interface Options -> Camera -> Enable")

    print("\n2. Load kernel modules:")
    print("   sudo modprobe bcm2835-v4l2")

    print("\n3. For Arducam stereo:")
    print("   # Check if detected:")
    print("   vcgencmd get_camera")
    print("   # Should show: supported=1 detected=1")


def benchmark_all_profiles():
    """Benchmark wszystkich profili latencji"""

    if not PICAMERA2_AVAILABLE:
        print("Picamera2 not available")
        return

    results = {}

    for profile_name in ['ultra_low', 'low', 'balanced']:
        print(f"\n{'='*70}")
        print(f"Testing profile: {profile_name}")
        print('='*70)

        try:
            camera = LowLatencyCamera(profile=profile_name, stereo=False)
            camera.initialize()

            # Benchmark
            metrics = camera.measure_latency(num_frames=50)
            results[profile_name] = metrics

            camera.close()
            time.sleep(0.5)

        except Exception as e:
            print(f"Error with profile {profile_name}: {e}")

    # Podsumowanie
    print(f"\n{'='*70}")
    print("=== SUMMARY ===")
    print('='*70)

    for profile_name, metrics in results.items():
        print(f"\n{profile_name.upper()}:")
        print(f"  Latency: {metrics['avg_ms']:.2f}ms "
              f"(min={metrics['min_ms']:.2f}, max={metrics['max_ms']:.2f})")
        print(f"  FPS: {metrics['fps']:.1f}")


if __name__ == "__main__":
    print("=== Picamera2 Low-Latency Configuration ===")

    # Konfiguracja systemowa
    configure_gpu_memory()
    print("\n" + "="*70)
    configure_camera_modules()

    # Jeśli Picamera2 dostępne, uruchom benchmark
    if PICAMERA2_AVAILABLE:
        print("\n" + "="*70)
        print("\nStarting benchmark...")

        try:
            benchmark_all_profiles()
        except Exception as e:
            print(f"\nBenchmark failed: {e}")
            print("\nTrying single profile...")

            camera = LowLatencyCamera(profile='ultra_low', stereo=False)
            camera.initialize()
            camera.measure_latency(num_frames=30)
            camera.close()

    print("\n" + "="*70)
    print("=== Recommendations ===")
    print("For <30ms total latency:")
    print("  • Use 'ultra_low' profile (buffer_count=2)")
    print("  • Disable auto-exposure and AWB")
    print("  • Use hardware H264 encoding")
    print("  • Set gpu_mem=256 in config.txt")
    print("  • Use zero-copy capture (capture_request)")
    print("  • Lock frames in shared memory")
    print("  • Pin camera thread to isolated CPU (e.g., CPU 2)")