#!/usr/bin/env python3
"""
Przykład 2: Nagrywanie wideo z Picamera2

Kamera: Arducam pivariety [2560x800 10-bit MONO]
"""

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, MJPEGEncoder, Quality
from picamera2.outputs import FileOutput, FfmpegOutput
import time


def record_h264_simple(duration: int = 5):
    """Nagranie H264 - najprostszy sposób."""
    with Picamera2() as picam2:
        config = picam2.create_video_configuration()
        picam2.configure(config)

        encoder = H264Encoder()

        picam2.start_recording(encoder, "video_simple.h264")
        print(f"Nagrywanie przez {duration} sekund...")
        time.sleep(duration)
        picam2.stop_recording()

        print("Zapisano: video_simple.h264")


def record_one_liner(duration: int = 5):
    """Metoda 'all in one' - najprostsza."""
    picam2 = Picamera2()
    picam2.start_and_record_video("video_quick.h264", duration=duration)
    picam2.close()
    print("Zapisano: video_quick.h264")


def record_with_quality(duration: int = 5, quality=Quality.HIGH):
    """Nagranie z określoną jakością."""
    with Picamera2() as picam2:
        config = picam2.create_video_configuration()
        picam2.configure(config)

        encoder = H264Encoder()

        picam2.start_recording(encoder, "video_high_quality.h264", quality=quality)
        print(f"Nagrywanie HIGH quality przez {duration} sekund...")
        time.sleep(duration)
        picam2.stop_recording()

        print("Zapisano: video_high_quality.h264")


def record_with_bitrate(duration: int = 5, bitrate: int = 10_000_000):
    """Nagranie z określonym bitrate (bit/s)."""
    with Picamera2() as picam2:
        config = picam2.create_video_configuration()
        picam2.configure(config)

        # Bitrate w bitach na sekundę (10 Mbps = 10_000_000)
        encoder = H264Encoder(bitrate=bitrate)

        picam2.start_recording(encoder, "video_10mbps.h264")
        print(f"Nagrywanie {bitrate / 1_000_000:.0f} Mbps przez {duration} sekund...")
        time.sleep(duration)
        picam2.stop_recording()

        print("Zapisano: video_10mbps.h264")


def record_mjpeg(duration: int = 5):
    """Nagranie MJPEG - większy plik, mniejsze obciążenie CPU."""
    with Picamera2() as picam2:
        config = picam2.create_video_configuration()
        picam2.configure(config)

        encoder = MJPEGEncoder()

        picam2.start_recording(encoder, "video.mjpeg")
        print(f"Nagrywanie MJPEG przez {duration} sekund...")
        time.sleep(duration)
        picam2.stop_recording()

        print("Zapisano: video.mjpeg")


def record_mp4_with_ffmpeg(duration: int = 5):
    """
    Nagranie MP4 używając FFmpeg.
    WYMAGA: zainstalowanego ffmpeg na systemie.
    """
    with Picamera2() as picam2:
        config = picam2.create_video_configuration()
        picam2.configure(config)

        encoder = H264Encoder()
        output = FfmpegOutput("video.mp4")

        picam2.start_recording(encoder, output)
        print(f"Nagrywanie MP4 przez {duration} sekund...")
        time.sleep(duration)
        picam2.stop_recording()

        print("Zapisano: video.mp4")


def record_with_custom_resolution(duration: int = 5):
    """Nagranie z określoną rozdzielczością."""
    with Picamera2() as picam2:
        # Dla kamery mono 2560x800 - możemy użyć mniejszych rozdzielczości
        config = picam2.create_video_configuration(
            main={
                "size": (1280, 400),  # Połowa rozdzielczości
                "format": "XBGR8888"
            }
        )
        picam2.configure(config)

        encoder = H264Encoder(bitrate=5_000_000)

        picam2.start_recording(encoder, "video_1280x400.h264")
        print(f"Nagrywanie 1280x400 przez {duration} sekund...")
        time.sleep(duration)
        picam2.stop_recording()

        print("Zapisano: video_1280x400.h264")


def record_with_framerate(duration: int = 5, fps: int = 30):
    """Nagranie z określoną liczbą klatek na sekundę."""
    with Picamera2() as picam2:
        # FrameDurationLimits w mikrosekundach
        frame_duration = int(1_000_000 / fps)  # np. 33333 dla 30fps

        config = picam2.create_video_configuration(
            controls={
                "FrameDurationLimits": (frame_duration, frame_duration)
            }
        )
        picam2.configure(config)

        encoder = H264Encoder()

        picam2.start_recording(encoder, f"video_{fps}fps.h264")
        print(f"Nagrywanie {fps}fps przez {duration} sekund...")
        time.sleep(duration)
        picam2.stop_recording()

        print(f"Zapisano: video_{fps}fps.h264")


def record_segments(segment_duration: int = 3, num_segments: int = 3):
    """Nagrywanie wielu segmentów wideo."""
    with Picamera2() as picam2:
        config = picam2.create_video_configuration()
        picam2.configure(config)
        picam2.start()

        encoder = H264Encoder()

        for i in range(num_segments):
            filename = f"video_segment_{i:03d}.h264"
            output = FileOutput(filename)

            picam2.start_encoder(encoder, output)
            print(f"Nagrywanie segmentu {i + 1}/{num_segments}...")
            time.sleep(segment_duration)
            picam2.stop_encoder()

            print(f"Zapisano: {filename}")

        picam2.stop()


def record_with_preview(duration: int = 5):
    """
    Nagranie z podglądem na ekranie.
    WYMAGA: środowiska graficznego (X11/Wayland).
    """
    with Picamera2() as picam2:
        config = picam2.create_video_configuration()
        picam2.configure(config)

        # Start z podglądem
        picam2.start_preview(True)
        picam2.start()

        encoder = H264Encoder()

        picam2.start_encoder(encoder, "video_with_preview.h264")
        print(f"Nagrywanie z podglądem przez {duration} sekund...")
        time.sleep(duration)
        picam2.stop_encoder()

        picam2.stop()
        print("Zapisano: video_with_preview.h264")


if __name__ == "__main__":
    print("=== Przykłady nagrywania wideo z Picamera2 ===\n")

    print("1. Proste nagranie H264...")
    record_h264_simple(duration=3)

    print("\n2. Metoda 'all in one'...")
    record_one_liner(duration=3)

    print("\n3. Wysokiej jakości...")
    record_with_quality(duration=3, quality=Quality.HIGH)

    print("\n4. Z określonym bitrate (10 Mbps)...")
    record_with_bitrate(duration=3, bitrate=10_000_000)

    print("\n5. MJPEG...")
    record_mjpeg(duration=3)

    print("\n6. MP4 przez FFmpeg...")
    try:
        record_mp4_with_ffmpeg(duration=3)
    except Exception as e:
        print(f"   BŁĄD (prawdopodobnie brak ffmpeg): {e}")

    print("\n7. Custom rozdzielczość...")
    record_with_custom_resolution(duration=3)

    print("\n8. Custom framerate (25fps)...")
    record_with_framerate(duration=3, fps=25)

    print("\n9. Segmenty wideo...")
    record_segments(segment_duration=2, num_segments=2)

    print("\n=== Wszystkie przykłady zakończone ===")