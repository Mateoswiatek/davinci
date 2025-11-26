#!/usr/bin/env python3
"""
Ultra-Low-Latency VR Streaming System
Integracja wszystkich optymalizacji dla <30ms latencji

Architektura:
- Process 1 (CPU 2, RT priority 90): Camera capture
- Process 2 (CPU 3, RT priority 70): YOLO detection (opcjonalne)
- Process 3 (CPU 1, RT priority 85): Network streaming
- Shared memory: Zero-copy frame transfer
"""

import os
import sys
import time
import signal
import argparse
import multiprocessing as mp
from multiprocessing import shared_memory, Process, Event
import numpy as np

# Import naszych modułów optymalizacyjnych
from cpu_pinning import CPUPinner
from realtime_scheduler import RealtimeScheduler, RealtimeThread, SCHED_FIFO
from memory_optimizations import MemoryManager, configure_vm_parameters
from network_optimizations import LowLatencySocket
from zero_copy_pipeline import ZeroCopyRingBuffer
from latency_profiler import LatencyProfiler

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠ Picamera2 not available")


class CameraProcess:
    """
    Proces camera capture (CPU 2, RT priority 90)
    """

    def __init__(self, ring_buffer_name: str, width: int, height: int,
                 stop_event: Event, profile: str = 'ultra_low'):
        self.ring_buffer_name = ring_buffer_name
        self.width = width
        self.height = height
        self.stop_event = stop_event
        self.profile = profile

    def run(self):
        """Main loop camera process"""

        # 1. CPU Pinning
        pinner = CPUPinner()
        pinner.set_cpu_affinity([2])  # CPU 2
        print(f"[Camera] Pinned to CPU 2")

        # 2. RT Scheduling
        try:
            scheduler = RealtimeScheduler()
            scheduler.set_realtime_priority(
                priority=RealtimeScheduler.PRIORITY_CAMERA_CAPTURE,
                policy=SCHED_FIFO
            )
            print(f"[Camera] RT priority set: SCHED_FIFO 90")
        except PermissionError:
            print(f"[Camera] ⚠ Cannot set RT priority (need sudo)")

        # 3. Memory locking
        try:
            mem_mgr = MemoryManager()
            mem_mgr.lock_all_memory()
            print(f"[Camera] Memory locked")
        except:
            print(f"[Camera] ⚠ Cannot lock memory")

        # 4. Podłącz się do ring buffer
        ring = ZeroCopyRingBuffer(
            self.ring_buffer_name,
            width=self.width,
            height=self.height,
            channels=3,
            num_buffers=3,
            create=False  # Główny proces utworzył
        )
        print(f"[Camera] Connected to ring buffer")

        # 5. Inicjalizuj kamerę
        if PICAMERA2_AVAILABLE:
            from picamera2_low_latency import LowLatencyCamera

            camera = LowLatencyCamera(
                camera_num=0,
                profile=self.profile,
                stereo=False
            )
            camera.initialize()
            print(f"[Camera] Camera initialized")
        else:
            print(f"[Camera] ⚠ Using fake camera (picamera2 not available)")
            camera = None

        # 6. Main capture loop
        frame_id = 0
        profiler = LatencyProfiler()

        print(f"[Camera] Starting capture loop...")

        while not self.stop_event.is_set():
            profiler.start_frame(frame_id)
            profiler.mark('capture_start')

            try:
                # Capture frame
                if camera:
                    frame, sensor_ts = camera.capture_with_timestamp()
                else:
                    # Fake frame dla testów
                    frame = np.random.randint(0, 255, (self.height, self.width, 3),
                                            dtype=np.uint8)
                    sensor_ts = time.time_ns()

                profiler.mark('capture_done')

                # Zero-copy write do shared memory
                write_buf = ring.get_write_buffer()
                write_buf.write_frame(frame, frame_id=frame_id,
                                    timestamp_ns=sensor_ts)
                ring.commit_write(frame_id)

                profiler.mark('buffer_written')
                profiler.end_frame()

                frame_id += 1

                # Print stats co 100 frames
                if frame_id % 100 == 0:
                    profiler.print_statistics(last_n=100)

            except Exception as e:
                print(f"[Camera] Error: {e}")
                break

        # Cleanup
        if camera:
            camera.close()

        print(f"[Camera] Stopped. Total frames: {frame_id}")


class YOLOProcess:
    """
    Proces YOLO detection (CPU 3, RT priority 70)
    Opcjonalny - można wyłączyć dla niższej latencji
    """

    def __init__(self, input_ring_name: str, output_ring_name: str,
                 stop_event: Event, model_size: str = 'n'):
        self.input_ring_name = input_ring_name
        self.output_ring_name = output_ring_name
        self.stop_event = stop_event
        self.model_size = model_size  # 'n', 's', 'm'

    def run(self):
        """Main loop YOLO process"""

        # CPU Pinning
        pinner = CPUPinner()
        pinner.set_cpu_affinity([3])  # CPU 3
        print(f"[YOLO] Pinned to CPU 3")

        # RT Scheduling
        try:
            scheduler = RealtimeScheduler()
            scheduler.set_realtime_priority(
                priority=RealtimeScheduler.PRIORITY_YOLO,
                policy=SCHED_FIFO
            )
            print(f"[YOLO] RT priority set: SCHED_FIFO 70")
        except:
            print(f"[YOLO] ⚠ Cannot set RT priority")

        # Podłącz do ring buffers
        input_ring = ZeroCopyRingBuffer(
            self.input_ring_name,
            width=1280, height=720, channels=3,
            num_buffers=3,
            create=False
        )

        output_ring = ZeroCopyRingBuffer(
            self.output_ring_name,
            width=1280, height=720, channels=3,
            num_buffers=3,
            create=False
        )

        print(f"[YOLO] Connected to buffers")

        # Load YOLO model
        try:
            from ultralytics import YOLO
            model = YOLO(f'yolov8{self.model_size}.pt')
            model.to('cpu')  # Lub 'cuda' jeśli masz GPU
            print(f"[YOLO] Model loaded: yolov8{self.model_size}")
        except:
            print(f"[YOLO] ⚠ YOLO not available, pass-through mode")
            model = None

        # Main loop
        profiler = LatencyProfiler()
        processed_frames = 0

        print(f"[YOLO] Starting detection loop...")

        while not self.stop_event.is_set():
            # Odczytaj frame z input buffer
            input_buf = input_ring.get_read_buffer()

            if input_buf is None:
                time.sleep(0.001)  # 1ms
                continue

            frame_view, metadata = input_buf.read_frame()
            frame_id = metadata.frame_id

            profiler.start_frame(frame_id)
            profiler.mark('yolo_start')

            # YOLO inference
            if model:
                # Resize dla szybszego inference (320x320 dla ultra-low latency)
                import cv2
                small_frame = cv2.resize(frame_view, (320, 320))

                results = model(small_frame, verbose=False)

                # Draw boxes (opcjonalnie)
                # annotated_frame = results[0].plot()
                annotated_frame = frame_view.copy()  # Pass-through dla testu
            else:
                annotated_frame = frame_view.copy()

            profiler.mark('yolo_done')

            # Write do output buffer
            output_buf = output_ring.get_write_buffer()
            output_buf.write_frame(annotated_frame, frame_id=frame_id,
                                 timestamp_ns=metadata.timestamp_capture_ns)
            output_ring.commit_write(frame_id)

            profiler.mark('output_written')
            profiler.end_frame()

            processed_frames += 1

            if processed_frames % 100 == 0:
                profiler.print_statistics(last_n=100)

        print(f"[YOLO] Stopped. Processed: {processed_frames} frames")


class NetworkProcess:
    """
    Proces network streaming (CPU 1, RT priority 85)
    """

    def __init__(self, ring_buffer_name: str, target_ip: str, target_port: int,
                 stop_event: Event, use_udp: bool = True):
        self.ring_buffer_name = ring_buffer_name
        self.target_ip = target_ip
        self.target_port = target_port
        self.stop_event = stop_event
        self.use_udp = use_udp

    def run(self):
        """Main loop network process"""

        # CPU Pinning
        pinner = CPUPinner()
        pinner.set_cpu_affinity([1])  # CPU 1
        print(f"[Network] Pinned to CPU 1")

        # RT Scheduling
        try:
            scheduler = RealtimeScheduler()
            scheduler.set_realtime_priority(
                priority=RealtimeScheduler.PRIORITY_NETWORK_SEND,
                policy=SCHED_FIFO
            )
            print(f"[Network] RT priority set: SCHED_FIFO 85")
        except:
            print(f"[Network] ⚠ Cannot set RT priority")

        # Podłącz do ring buffer
        ring = ZeroCopyRingBuffer(
            self.ring_buffer_name,
            width=1280, height=720, channels=3,
            num_buffers=3,
            create=False
        )
        print(f"[Network] Connected to ring buffer")

        # Utwórz socket
        sock = LowLatencySocket(use_udp=self.use_udp, port=self.target_port)
        print(f"[Network] Socket created: {'UDP' if self.use_udp else 'TCP'}")

        # Main loop
        profiler = LatencyProfiler()
        sent_frames = 0

        print(f"[Network] Starting streaming to {self.target_ip}:{self.target_port}...")

        while not self.stop_event.is_set():
            # Odczytaj frame
            buf = ring.get_read_buffer()

            if buf is None:
                time.sleep(0.001)
                continue

            frame_view, metadata = buf.read_frame()
            frame_id = metadata.frame_id

            profiler.start_frame(frame_id)
            profiler.mark('network_start')

            try:
                # Dla prawdziwego streaming użyj H264 encoder
                # Tu symulujemy wysyłanie surowych danych
                # W produkcji: użyj hardware H264 + UDP/RTP

                # JPEG encoding dla testu (szybkie)
                import cv2
                _, jpeg_data = cv2.imencode('.jpg', frame_view,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])

                profiler.mark('encode_done')

                # Wyślij
                if self.use_udp:
                    # UDP: podziel na pakiety jeśli > MTU (1500 bytes)
                    # Tu uproszczona wersja
                    sock.send(jpeg_data.tobytes(),
                            addr=(self.target_ip, self.target_port))
                else:
                    sock.send(jpeg_data.tobytes())

                profiler.mark('network_sent')

                # Oblicz latencję end-to-end
                now_ns = time.time_ns()
                total_latency_ms = (now_ns - metadata.timestamp_capture_ns) / 1_000_000

                profiler.end_frame()

                sent_frames += 1

                if sent_frames % 100 == 0:
                    print(f"[Network] Sent {sent_frames} frames, "
                          f"last latency: {total_latency_ms:.1f}ms")
                    profiler.print_statistics(last_n=100)

            except Exception as e:
                print(f"[Network] Error: {e}")

        sock.close()
        print(f"[Network] Stopped. Sent: {sent_frames} frames")


class VRStreamingSystem:
    """
    Główny system zarządzający wszystkimi procesami
    """

    def __init__(self, width: int = 1280, height: int = 720,
                 target_ip: str = '192.168.1.100', target_port: int = 8554,
                 enable_yolo: bool = False, use_udp: bool = True):
        """
        Args:
            width, height: Rozdzielczość frame'ów
            target_ip: IP Oculus Quest
            target_port: Port dla streaming
            enable_yolo: True = włącz YOLO detection
            use_udp: True = UDP, False = TCP
        """

        self.width = width
        self.height = height
        self.target_ip = target_ip
        self.target_port = target_port
        self.enable_yolo = enable_yolo
        self.use_udp = use_udp

        # Shared memory ring buffers
        self.camera_ring_name = "vr_camera_buffer"
        self.yolo_ring_name = "vr_yolo_buffer" if enable_yolo else None

        # Processes
        self.processes = []
        self.stop_event = mp.Event()

        # Signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n[Main] Stopping...")
        self.stop()

    def initialize(self):
        """Inicjalizuj system"""

        print("="*70)
        print("VR Streaming System - Ultra-Low Latency")
        print("="*70)

        # 1. Sprawdź system config
        print("\n[Main] Checking system configuration...")
        self._check_system_config()

        # 2. Utwórz ring buffers
        print("\n[Main] Creating shared memory buffers...")

        self.camera_ring = ZeroCopyRingBuffer(
            self.camera_ring_name,
            width=self.width,
            height=self.height,
            channels=3,
            num_buffers=3,
            create=True
        )

        if self.enable_yolo:
            self.yolo_ring = ZeroCopyRingBuffer(
                self.yolo_ring_name,
                width=self.width,
                height=self.height,
                channels=3,
                num_buffers=3,
                create=True
            )

        print("✓ Buffers created")

    def _check_system_config(self):
        """Sprawdź konfigurację systemu"""

        # Check isolated CPUs
        try:
            with open('/sys/devices/system/cpu/isolated', 'r') as f:
                isolated = f.read().strip()
                if isolated:
                    print(f"✓ Isolated CPUs: {isolated}")
                else:
                    print(f"⚠ No isolated CPUs (add isolcpus= to cmdline.txt)")
        except:
            print(f"⚠ Cannot check isolated CPUs")

        # Check RT limits
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_RTPRIO)
        if hard > 0:
            print(f"✓ RT priority limit: {hard}")
        else:
            print(f"⚠ RT priority not configured (edit /etc/security/limits.conf)")

        # Check memory
        mem_mgr = MemoryManager()
        stats = mem_mgr.get_memory_stats()
        print(f"✓ Memory: {stats.available_mb:.0f} MB available")

    def start(self):
        """Uruchom wszystkie procesy"""

        print("\n[Main] Starting processes...\n")

        # 1. Camera process (CPU 2)
        camera_proc = Process(
            target=CameraProcess(
                self.camera_ring_name,
                self.width, self.height,
                self.stop_event,
                profile='ultra_low'
            ).run,
            name="CameraProcess"
        )
        camera_proc.start()
        self.processes.append(camera_proc)
        time.sleep(0.5)  # Poczekaj na start

        # 2. YOLO process (CPU 3) - opcjonalnie
        if self.enable_yolo:
            yolo_proc = Process(
                target=YOLOProcess(
                    self.camera_ring_name,
                    self.yolo_ring_name,
                    self.stop_event,
                    model_size='n'  # YOLOv8n - najszybszy
                ).run,
                name="YOLOProcess"
            )
            yolo_proc.start()
            self.processes.append(yolo_proc)
            time.sleep(0.5)

        # 3. Network process (CPU 1)
        network_ring = self.yolo_ring_name if self.enable_yolo else self.camera_ring_name

        network_proc = Process(
            target=NetworkProcess(
                network_ring,
                self.target_ip,
                self.target_port,
                self.stop_event,
                use_udp=self.use_udp
            ).run,
            name="NetworkProcess"
        )
        network_proc.start()
        self.processes.append(network_proc)

        print(f"\n✓ All processes started")
        print(f"\nProcesses:")
        for p in self.processes:
            print(f"  - {p.name} (PID: {p.pid})")

        print(f"\nPress Ctrl+C to stop\n")

    def stop(self):
        """Zatrzymaj wszystkie procesy"""

        print("\n[Main] Stopping processes...")

        self.stop_event.set()

        # Poczekaj na zakończenie
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                print(f"⚠ Force killing {p.name}")
                p.terminate()

        # Cleanup shared memory
        print("\n[Main] Cleaning up shared memory...")
        self.camera_ring.unlink()
        if self.enable_yolo:
            self.yolo_ring.unlink()

        print("✓ Stopped")

    def run(self):
        """Uruchom system i poczekaj na stop"""

        self.initialize()
        self.start()

        # Wait for stop signal
        try:
            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        self.stop()


def main():
    parser = argparse.ArgumentParser(
        description='Ultra-Low-Latency VR Streaming System'
    )

    parser.add_argument('--width', type=int, default=1280,
                       help='Frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Frame height (default: 720)')
    parser.add_argument('--ip', type=str, default='192.168.1.100',
                       help='Target IP (Oculus Quest)')
    parser.add_argument('--port', type=int, default=8554,
                       help='Target port (default: 8554)')
    parser.add_argument('--yolo', action='store_true',
                       help='Enable YOLO detection (adds latency)')
    parser.add_argument('--tcp', action='store_true',
                       help='Use TCP instead of UDP')

    args = parser.parse_args()

    # Sprawdź czy jesteśmy root (dla RT scheduling)
    if os.geteuid() != 0:
        print("⚠ Warning: Not running as root. RT scheduling may fail.")
        print("  Run with: sudo python3 vr_streaming_optimized.py")
        print("  Or configure /etc/security/limits.conf\n")

    # Uruchom system
    system = VRStreamingSystem(
        width=args.width,
        height=args.height,
        target_ip=args.ip,
        target_port=args.port,
        enable_yolo=args.yolo,
        use_udp=not args.tcp
    )

    system.run()


if __name__ == "__main__":
    main()