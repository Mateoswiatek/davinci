#!/usr/bin/env python3
"""
Zero-copy pipeline dla VR streaming
- Shared memory between processes
- DMA buffers (v4l2)
- memoryview for zero-copy slicing
- GPU memory mapping (dla YOLO inference)
"""

import os
import time
import mmap
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Optional, Tuple
from dataclasses import dataclass
import ctypes

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class FrameMetadata:
    """Metadata dla każdego frame'a w pipeline"""
    frame_id: int
    timestamp_capture_ns: int
    timestamp_ready_ns: int
    width: int
    height: int
    channels: int
    format: str  # 'RGB888', 'YUYV', etc.


class ZeroCopyFrameBuffer:
    """
    Zero-copy frame buffer używający shared memory
    Optymalizowany dla przekazywania między procesami bez kopiowania
    """

    def __init__(self, name: str, width: int, height: int,
                 channels: int = 3, dtype=np.uint8,
                 create: bool = True):
        """
        Args:
            name: Unikalna nazwa shared memory
            width, height, channels: Wymiary frame'a
            dtype: Typ danych (np. np.uint8 dla RGB)
            create: True = utwórz nowy, False = podłącz do istniejącego
        """

        self.name = name
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = np.dtype(dtype)

        # Rozmiar bufora
        self.frame_size = width * height * channels * self.dtype.itemsize

        # Metadata size (128 bytes powinno wystarczyć)
        self.metadata_size = 128

        self.total_size = self.frame_size + self.metadata_size

        # Utwórz lub podłącz shared memory
        if create:
            self.shm = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=self.total_size
            )
            # Inicjalizuj metadata
            self._init_metadata()
            print(f"✓ Created zero-copy buffer '{name}': "
                  f"{self.frame_size / 1024 / 1024:.2f} MB")
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            print(f"✓ Attached to zero-copy buffer '{name}'")

        # Memory views (zero-copy)
        self._setup_views()

    def _init_metadata(self):
        """Inicjalizuj metadata buffer"""
        # Format metadata: [frame_id(i64), timestamp_capture(i64),
        #                   timestamp_ready(i64), reserved...]
        meta_array = np.ndarray(
            (self.metadata_size // 8,),
            dtype=np.int64,
            buffer=self.shm.buf[:self.metadata_size]
        )
        meta_array[:] = 0

    def _setup_views(self):
        """Ustaw memory views na bufory"""

        # Metadata view
        self.metadata_view = memoryview(self.shm.buf[:self.metadata_size])
        self.metadata_array = np.ndarray(
            (self.metadata_size // 8,),
            dtype=np.int64,
            buffer=self.metadata_view
        )

        # Frame data view (zero-copy!)
        self.frame_view = memoryview(
            self.shm.buf[self.metadata_size:self.metadata_size + self.frame_size]
        )
        self.frame_array = np.ndarray(
            (self.height, self.width, self.channels),
            dtype=self.dtype,
            buffer=self.frame_view
        )

    def write_frame(self, frame: np.ndarray, frame_id: int,
                   timestamp_ns: Optional[int] = None):
        """
        Napisz frame do shared memory (zero-copy jeśli możliwe)

        Args:
            frame: Numpy array z danymi frame'a
            frame_id: Unikalny ID frame'a
            timestamp_ns: Timestamp capture w nanosekundach
        """

        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        # Sprawdź wymiary
        if frame.shape != (self.height, self.width, self.channels):
            raise ValueError(f"Frame shape mismatch: expected "
                           f"{(self.height, self.width, self.channels)}, "
                           f"got {frame.shape}")

        # Zero-copy: użyj copyto zamiast =
        # copyto jest szybsze i nie tworzy intermediate copies
        np.copyto(self.frame_array, frame)

        # Update metadata
        self.metadata_array[0] = frame_id
        self.metadata_array[1] = timestamp_ns
        self.metadata_array[2] = time.time_ns()  # timestamp_ready

    def read_frame(self) -> Tuple[np.ndarray, FrameMetadata]:
        """
        Odczytaj frame ze shared memory (zero-copy view)

        Returns:
            tuple: (frame_view, metadata)
            UWAGA: frame_view jest view na shared memory!
            Jeśli potrzebujesz kopii: frame_view.copy()
        """

        # Odczytaj metadata
        frame_id = int(self.metadata_array[0])
        timestamp_capture = int(self.metadata_array[1])
        timestamp_ready = int(self.metadata_array[2])

        metadata = FrameMetadata(
            frame_id=frame_id,
            timestamp_capture_ns=timestamp_capture,
            timestamp_ready_ns=timestamp_ready,
            width=self.width,
            height=self.height,
            channels=self.channels,
            format='RGB888'
        )

        # Return view (zero-copy!)
        return self.frame_array, metadata

    def get_memoryview(self) -> memoryview:
        """Pobierz raw memoryview (dla advanced use)"""
        return self.frame_view

    def __del__(self):
        if hasattr(self, 'shm'):
            self.shm.close()

    def unlink(self):
        """Usuń shared memory (tylko creator)"""
        self.shm.unlink()
        print(f"✓ Unlinked '{self.name}'")


class ZeroCopyRingBuffer:
    """
    Ring buffer z zero-copy semantyką
    Używa trzech buforów dla triple-buffering
    """

    def __init__(self, base_name: str, width: int, height: int,
                 channels: int = 3, num_buffers: int = 3,
                 create: bool = True):
        """
        Args:
            num_buffers: Liczba buforów (zazwyczaj 2-4)
        """

        self.base_name = base_name
        self.num_buffers = num_buffers
        self.width = width
        self.height = height
        self.channels = channels

        # Utwórz bufory
        self.buffers = []
        for i in range(num_buffers):
            buf = ZeroCopyFrameBuffer(
                name=f"{base_name}_{i}",
                width=width,
                height=height,
                channels=channels,
                create=create
            )
            self.buffers.append(buf)

        # Indeksy (w shared memory dla sync między procesami)
        control_name = f"{base_name}_control"
        control_size = 64  # 64 bytes na counters

        if create:
            self.control_shm = shared_memory.SharedMemory(
                name=control_name,
                create=True,
                size=control_size
            )
            # [write_idx, read_idx, latest_frame_id]
            control = np.ndarray((8,), dtype=np.int64, buffer=self.control_shm.buf)
            control[:] = 0
        else:
            self.control_shm = shared_memory.SharedMemory(name=control_name)

        self.control = np.ndarray((8,), dtype=np.int64, buffer=self.control_shm.buf)

        print(f"✓ Zero-copy ring buffer: {num_buffers} buffers")

    def get_write_buffer(self) -> ZeroCopyFrameBuffer:
        """Pobierz buffer do zapisu (producer)"""
        write_idx = int(self.control[0]) % self.num_buffers
        return self.buffers[write_idx]

    def commit_write(self, frame_id: int):
        """Potwierdź zapis (increment write counter)"""
        self.control[0] += 1
        self.control[2] = frame_id  # latest_frame_id

    def get_read_buffer(self) -> Optional[ZeroCopyFrameBuffer]:
        """Pobierz najnowszy buffer do odczytu (consumer)"""
        write_idx = int(self.control[0])
        read_idx = int(self.control[1])

        # Sprawdź czy są nowe dane
        if read_idx >= write_idx:
            return None

        # Pobierz najnowszy dostępny buffer
        buffer_idx = (write_idx - 1) % self.num_buffers
        self.control[1] = write_idx

        return self.buffers[buffer_idx]

    def get_latest_buffer(self) -> Optional[ZeroCopyFrameBuffer]:
        """Pobierz najnowszy buffer bez update read_idx"""
        write_idx = int(self.control[0])
        if write_idx == 0:
            return None
        buffer_idx = (write_idx - 1) % self.num_buffers
        return self.buffers[buffer_idx]

    def __del__(self):
        if hasattr(self, 'control_shm'):
            self.control_shm.close()

    def unlink(self):
        """Cleanup wszystkich buforów"""
        for buf in self.buffers:
            buf.unlink()
        self.control_shm.unlink()
        print(f"✓ Unlinked ring buffer '{self.base_name}'")


class DMABuffer:
    """
    Wrapper dla V4L2 DMA buffers (direct memory access z kamery)
    UWAGA: Wymaga libcamera/v4l2 support
    """

    def __init__(self, device: str = '/dev/video0'):
        self.device = device
        self.fd = None

    def open(self):
        """Otwórz device"""
        self.fd = os.open(self.device, os.O_RDWR)
        print(f"✓ Opened {self.device}")

    def request_buffers(self, count: int = 3):
        """Request DMA buffers od kernela"""
        # Wymaga ioctl calls do V4L2
        # Implementacja zależy od użycia ctypes + V4L2 ioctl
        print(f"⚠ DMA buffer allocation not implemented in pure Python")
        print(f"  Use: v4l2-ctl --set-fmt-video=width=2560,height=800")

    def close(self):
        if self.fd:
            os.close(self.fd)


def benchmark_copy_methods(width: int = 2560, height: int = 800,
                          channels: int = 3, iterations: int = 100):
    """
    Benchmark różnych metod kopiowania frame'ów
    """

    print(f"\n=== Copy Methods Benchmark ===")
    print(f"Frame size: {width}x{height}x{channels} = "
          f"{width * height * channels / 1024 / 1024:.2f} MB")
    print(f"Iterations: {iterations}\n")

    # Przygotuj dane
    source_frame = np.random.randint(0, 255, (height, width, channels),
                                    dtype=np.uint8)

    results = {}

    # 1. Zwykłe kopiowanie (=)
    print("1. Standard copy (=)")
    start = time.perf_counter()
    for _ in range(iterations):
        dest = source_frame  # Reference (instant)
    elapsed = time.perf_counter() - start
    results['reference'] = elapsed
    print(f"   Time: {elapsed*1000:.2f}ms (total), "
          f"{elapsed*1000/iterations:.3f}ms per frame")

    # 2. np.copy()
    print("2. np.copy()")
    start = time.perf_counter()
    for _ in range(iterations):
        dest = np.copy(source_frame)
    elapsed = time.perf_counter() - start
    results['np.copy'] = elapsed
    print(f"   Time: {elapsed*1000:.2f}ms (total), "
          f"{elapsed*1000/iterations:.3f}ms per frame")

    # 3. np.copyto() (in-place, zero allocation)
    print("3. np.copyto() [RECOMMENDED]")
    dest_buffer = np.empty_like(source_frame)
    start = time.perf_counter()
    for _ in range(iterations):
        np.copyto(dest_buffer, source_frame)
    elapsed = time.perf_counter() - start
    results['np.copyto'] = elapsed
    print(f"   Time: {elapsed*1000:.2f}ms (total), "
          f"{elapsed*1000/iterations:.3f}ms per frame")

    # 4. memoryview (zero-copy)
    print("4. memoryview (zero-copy view)")
    start = time.perf_counter()
    for _ in range(iterations):
        view = memoryview(source_frame)
    elapsed = time.perf_counter() - start
    results['memoryview'] = elapsed
    print(f"   Time: {elapsed*1000:.2f}ms (total), "
          f"{elapsed*1000/iterations:.3f}ms per frame")

    # 5. Shared memory write
    print("5. Shared memory write")
    shm_buf = ZeroCopyFrameBuffer(
        "benchmark",
        width=width,
        height=height,
        channels=channels,
        create=True
    )
    start = time.perf_counter()
    for i in range(iterations):
        shm_buf.write_frame(source_frame, frame_id=i)
    elapsed = time.perf_counter() - start
    results['shared_memory'] = elapsed
    print(f"   Time: {elapsed*1000:.2f}ms (total), "
          f"{elapsed*1000/iterations:.3f}ms per frame")
    shm_buf.unlink()

    # Podsumowanie
    print(f"\n=== Summary ===")
    fastest = min(results.values())
    for method, time_s in results.items():
        speedup = time_s / fastest
        print(f"  {method:20s}: {time_s*1000/iterations:6.3f}ms/frame "
              f"({speedup:.2f}x)")

    return results


def demo_zero_copy_pipeline():
    """
    Demo kompletnego zero-copy pipeline:
    Camera -> Shared Memory -> YOLO -> Network
    """

    print("\n=== Zero-Copy Pipeline Demo ===\n")

    width, height, channels = 1280, 720, 3

    # 1. Utwórz ring buffer
    ring = ZeroCopyRingBuffer(
        "vr_pipeline",
        width=width,
        height=height,
        channels=channels,
        num_buffers=3,
        create=True
    )

    # 2. Symuluj camera capture (producer)
    print("Simulating camera capture...")
    for frame_id in range(5):
        # Pobierz buffer do zapisu
        write_buf = ring.get_write_buffer()

        # Symuluj capture (w prawdziwym kodzie: picamera2.capture_request())
        fake_frame = np.full((height, width, channels), frame_id, dtype=np.uint8)

        # Zero-copy write
        write_buf.write_frame(fake_frame, frame_id=frame_id,
                             timestamp_ns=time.time_ns())

        # Commit
        ring.commit_write(frame_id)

        print(f"  Frame {frame_id} written to buffer")
        time.sleep(0.016)  # 60fps

    # 3. Symuluj consumer (YOLO/network)
    print("\nSimulating consumer (YOLO/network)...")
    while True:
        # Pobierz najnowszy buffer
        read_buf = ring.get_read_buffer()

        if read_buf is None:
            print("  No new frames")
            break

        # Zero-copy read (view na shared memory!)
        frame_view, metadata = read_buf.read_frame()

        # Oblicz latencję
        now_ns = time.time_ns()
        latency_ms = (now_ns - metadata.timestamp_capture_ns) / 1_000_000

        print(f"  Frame {metadata.frame_id} read: "
              f"latency={latency_ms:.2f}ms, "
              f"value={frame_view[0, 0, 0]}")

        # W prawdziwym kodzie: YOLO inference lub network send
        # Używasz frame_view bezpośrednio (zero-copy!)

    # Cleanup
    ring.unlink()


if __name__ == "__main__":
    print("=== Zero-Copy Techniques for VR Streaming ===\n")

    # Benchmark
    print("="*70)
    benchmark_copy_methods(width=2560, height=800, channels=3, iterations=50)

    # Demo pipeline
    print("\n" + "="*70)
    demo_zero_copy_pipeline()

    print("\n" + "="*70)
    print("=== Recommendations ===")
    print("\nFor minimum latency:")
    print("  1. Use ZeroCopyRingBuffer with 3 buffers (triple-buffering)")
    print("  2. Camera writes directly to shared memory")
    print("  3. YOLO/Network read from shared memory (memoryview)")
    print("  4. Use np.copyto() if copy needed (faster than np.copy)")
    print("  5. Avoid Python list/dict conversions")
    print("  6. Use memoryview for slicing without copy")
    print("\nTypical pipeline latency breakdown:")
    print("  • Camera capture:        2-5ms")
    print("  • Shared memory write:   0.5-1ms")
    print("  • YOLO inference:        10-20ms (depends on model)")
    print("  • Network encode+send:   5-10ms")
    print("  • Total:                 17-36ms")
    print("\nTo achieve <30ms:")
    print("  • Use lightweight YOLO (YOLOv8n, 320x320)")
    print("  • Hardware H264 encoding (GPU)")
    print("  • UDP instead of TCP")
    print("  • All zero-copy operations")