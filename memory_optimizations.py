#!/usr/bin/env python3
"""
Memory optimizations dla ultra-niskiej latencji
- Huge pages
- Memory locking (mlockall)
- Shared memory buffers
- Zero-copy techniques
"""

import os
import ctypes
import mmap
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Optional, Tuple
from dataclasses import dataclass

# Linux memory constants
MCL_CURRENT = 1
MCL_FUTURE = 2
MCL_ONFAULT = 4


@dataclass
class MemoryStats:
    """Statystyki pamięci"""
    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    huge_pages_total: int
    huge_pages_free: int
    huge_page_size_kb: int


class MemoryManager:
    """Zarządza optymalizacjami pamięci"""

    def __init__(self):
        self.libc = ctypes.CDLL('libc.so.6')
        self.locked = False

    def lock_all_memory(self):
        """
        Blokuj całą pamięć w RAM (zapobiega swapping)
        MCL_CURRENT | MCL_FUTURE
        """
        flags = MCL_CURRENT | MCL_FUTURE

        result = self.libc.mlockall(flags)
        if result != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"mlockall failed: {os.strerror(errno)}")

        self.locked = True
        print("✓ All memory locked (mlockall)")

    def unlock_all_memory(self):
        """Odblokuj pamięć"""
        if self.locked:
            self.libc.munlockall()
            self.locked = False
            print("✓ Memory unlocked")

    def get_memory_stats(self) -> MemoryStats:
        """Pobierz statystyki pamięci"""
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().split()[0]
                    meminfo[key] = int(value)

        total_mb = meminfo.get('MemTotal', 0) / 1024
        available_mb = meminfo.get('MemAvailable', 0) / 1024
        used_mb = total_mb - available_mb
        percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0

        return MemoryStats(
            total_mb=total_mb,
            available_mb=available_mb,
            used_mb=used_mb,
            percent=percent,
            huge_pages_total=meminfo.get('HugePages_Total', 0),
            huge_pages_free=meminfo.get('HugePages_Free', 0),
            huge_page_size_kb=meminfo.get('Hugepagesize', 0)
        )

    def disable_swap(self):
        """Wyłącz swap (wymaga sudo)"""
        print("Execute: sudo swapoff -a")
        print("To make permanent, comment out swap in /etc/fstab")

    def configure_huge_pages(self, nr_hugepages: int = 128):
        """
        Konfiguruj huge pages dla dużych alokacji
        2MB pages są domyślne na ARM64
        """
        print(f"\n=== Huge Pages Configuration ===")
        print(f"Setting {nr_hugepages} huge pages (2MB each = {nr_hugepages * 2}MB total)")

        print(f"\nExecute:")
        print(f"  echo {nr_hugepages} | sudo tee /proc/sys/vm/nr_hugepages")
        print(f"\nTo make permanent, add to /etc/sysctl.conf:")
        print(f"  vm.nr_hugepages = {nr_hugepages}")

        # Sprawdź obecną konfigurację
        try:
            with open('/proc/sys/vm/nr_hugepages', 'r') as f:
                current = f.read().strip()
                print(f"\nCurrent huge pages: {current}")
        except:
            pass


class SharedMemoryBuffer:
    """
    Zero-copy shared memory buffer dla frame'ów
    Używa multiprocessing.shared_memory (Python 3.8+)
    """

    def __init__(self, name: str, shape: Tuple[int, ...],
                 dtype=np.uint8, create: bool = True):
        """
        Args:
            name: Unikalna nazwa bufora
            shape: Wymiary numpy array (np. (800, 2560, 3))
            dtype: Typ danych
            create: True = utwórz nowy, False = podłącz do istniejącego
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

        if create:
            self.shm = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=self.nbytes
            )
            print(f"✓ Created shared memory '{name}': {self.nbytes / 1024 / 1024:.2f} MB")
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            print(f"✓ Attached to shared memory '{name}'")

        # Utwórz numpy array z shared memory
        self.array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'shm'):
            self.shm.close()

    def unlink(self):
        """Usuń shared memory (tylko creator powinien to wywołać)"""
        self.shm.unlink()
        print(f"✓ Unlinked shared memory '{self.name}'")


class RingBuffer:
    """
    Lock-free ring buffer w shared memory
    Dla camera frames (producer: camera, consumer: encoder/network)
    """

    def __init__(self, name: str, buffer_count: int,
                 frame_shape: Tuple[int, ...],
                 dtype=np.uint8, create: bool = True):
        """
        Args:
            buffer_count: Liczba frame'ów w ring buffer (np. 3-4)
            frame_shape: Wymiary jednego frame'a
        """
        self.name = name
        self.buffer_count = buffer_count
        self.frame_shape = frame_shape
        self.dtype = dtype

        # Całkowity rozmiar: buffers + metadata
        frame_size = int(np.prod(frame_shape)) * np.dtype(dtype).itemsize
        self.total_size = frame_size * buffer_count + 4096  # +4KB na metadata

        if create:
            self.shm = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=self.total_size
            )
            # Inicjalizuj counters na 0
            metadata = np.ndarray((2,), dtype=np.int64, buffer=self.shm.buf[:16])
            metadata[:] = 0
        else:
            self.shm = shared_memory.SharedMemory(name=name)

        # Metadata: [write_index, read_index]
        self.metadata = np.ndarray((2,), dtype=np.int64, buffer=self.shm.buf[:16])

        # Bufory na frames
        self.buffers = []
        offset = 4096  # Zacznij po metadata
        for i in range(buffer_count):
            buf = np.ndarray(frame_shape, dtype=dtype,
                           buffer=self.shm.buf[offset:offset + frame_size])
            self.buffers.append(buf)
            offset += frame_size

        if create:
            print(f"✓ Created ring buffer '{name}': "
                  f"{buffer_count} x {frame_size / 1024 / 1024:.2f} MB")

    def write(self, frame: np.ndarray) -> int:
        """
        Napisz frame (producer)
        Returns: index gdzie zapisano
        """
        write_idx = self.metadata[0] % self.buffer_count
        np.copyto(self.buffers[write_idx], frame)
        self.metadata[0] += 1
        return write_idx

    def read(self) -> Optional[np.ndarray]:
        """
        Odczytaj frame (consumer)
        Returns: frame lub None jeśli brak nowych
        """
        read_idx = self.metadata[1]
        write_idx = self.metadata[0]

        # Sprawdź czy są nowe dane
        if read_idx >= write_idx:
            return None

        # Odczytaj najnowszy frame
        buffer_idx = (write_idx - 1) % self.buffer_count
        self.metadata[1] = write_idx
        return self.buffers[buffer_idx]

    def get_latest(self) -> Optional[np.ndarray]:
        """Pobierz najnowszy dostępny frame bez update read_idx"""
        write_idx = self.metadata[0]
        if write_idx == 0:
            return None
        buffer_idx = (write_idx - 1) % self.buffer_count
        return self.buffers[buffer_idx]

    def __del__(self):
        if hasattr(self, 'shm'):
            self.shm.close()

    def unlink(self):
        self.shm.unlink()


class MemoryPool:
    """
    Pre-allocated memory pool dla zero-copy operations
    """

    def __init__(self, buffer_size: int, pool_size: int):
        """
        Args:
            buffer_size: Rozmiar pojedynczego bufora w bajtach
            pool_size: Liczba buforów w pool
        """
        self.buffer_size = buffer_size
        self.pool_size = pool_size

        # Alokuj całą pamięć z góry
        self.memory = bytearray(buffer_size * pool_size)

        # Free list
        self.free_buffers = list(range(pool_size))
        self.used_buffers = set()

        print(f"✓ Memory pool created: {pool_size} x {buffer_size / 1024:.1f} KB")

    def acquire(self) -> Optional[memoryview]:
        """Pobierz wolny bufor"""
        if not self.free_buffers:
            return None

        idx = self.free_buffers.pop()
        self.used_buffers.add(idx)

        offset = idx * self.buffer_size
        return memoryview(self.memory[offset:offset + self.buffer_size])

    def release(self, buffer_idx: int):
        """Zwolnij bufor"""
        if buffer_idx in self.used_buffers:
            self.used_buffers.remove(buffer_idx)
            self.free_buffers.append(buffer_idx)


def configure_vm_parameters():
    """Konfiguruj parametry VM dla niskiej latencji"""

    print("\n=== VM Parameters dla Low-Latency ===\n")

    params = {
        # Wyłącz swap
        "vm.swappiness": "0",

        # Dirty page writeback (flush co 1s zamiast 30s)
        "vm.dirty_expire_centisecs": "100",  # 1s
        "vm.dirty_writeback_centisecs": "100",

        # Cache pressure (preferuj page cache nad swapping)
        "vm.vfs_cache_pressure": "50",

        # Overcommit memory
        "vm.overcommit_memory": "1",

        # Huge pages
        "vm.nr_hugepages": "128",

        # Transparent huge pages (użyj madvise zamiast always)
        "/sys/kernel/mm/transparent_hugepage/enabled": "madvise",
    }

    print("Add to /etc/sysctl.conf:")
    for param, value in params.items():
        if param.startswith('/sys'):
            print(f"  echo {value} | sudo tee {param}")
        else:
            print(f"  {param} = {value}")

    print("\nApply immediately:")
    print("  sudo sysctl -p")


def demo_shared_memory():
    """Demo użycia shared memory dla frame'ów"""

    print("\n=== Shared Memory Demo ===\n")

    # Przykładowe wymiary frame'a: 800x2560x3 (RGB)
    frame_shape = (800, 2560, 3)

    # 1. Prosty shared buffer
    print("1. Creating simple shared buffer...")
    buf = SharedMemoryBuffer("vr_frame", frame_shape, create=True)

    # Zapisz dane
    test_frame = np.random.randint(0, 255, frame_shape, dtype=np.uint8)
    np.copyto(buf.array, test_frame)
    print(f"   Written frame: {buf.array.shape}, sum={buf.array.sum()}")

    # 2. Ring buffer
    print("\n2. Creating ring buffer...")
    ring = RingBuffer("vr_ring", buffer_count=3, frame_shape=frame_shape, create=True)

    # Zapisz kilka frame'ów
    for i in range(5):
        frame = np.full(frame_shape, i, dtype=np.uint8)
        idx = ring.write(frame)
        print(f"   Wrote frame {i} to buffer {idx}")

    # Odczytaj
    latest = ring.get_latest()
    if latest is not None:
        print(f"   Latest frame value: {latest[0, 0, 0]}")

    # Cleanup
    buf.unlink()
    ring.unlink()


if __name__ == "__main__":
    print("=== Memory Optimization Setup ===\n")

    mem_mgr = MemoryManager()

    # Statystyki
    stats = mem_mgr.get_memory_stats()
    print(f"Memory: {stats.used_mb:.0f} / {stats.total_mb:.0f} MB ({stats.percent:.1f}%)")
    print(f"Available: {stats.available_mb:.0f} MB")
    print(f"Huge pages: {stats.huge_pages_free} / {stats.huge_pages_total} "
          f"({stats.huge_page_size_kb} KB each)")

    # Konfiguracja
    print("\n" + "="*60)
    configure_vm_parameters()

    print("\n" + "="*60)
    mem_mgr.configure_huge_pages(128)

    print("\n" + "="*60)
    print("=== Memory Locking ===")
    print("\nTo lock all memory in RAM:")
    print("  1. Set limits in /etc/security/limits.conf:")
    print("     pi  -  memlock  unlimited")
    print("  2. In code: mem_mgr.lock_all_memory()")
    print("\nWarning: mlockall requires sufficient RAM!")

    # Demo
    print("\n" + "="*60)
    demo_shared_memory()

    print("\n=== Summary ===")
    print("For production VR streaming:")
    print("  • Use RingBuffer for camera frames (3-4 buffers)")
    print("  • Lock memory with mlockall() to prevent page faults")
    print("  • Configure huge pages for large allocations")
    print("  • Disable swap completely")
    print("  • Use zero-copy numpy operations with shared memory")