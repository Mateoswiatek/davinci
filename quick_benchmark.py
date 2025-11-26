#!/usr/bin/env python3
"""
Quick benchmark suite dla VR streaming system
Testuje wszystkie komponenty i pokazuje czy osiągamy <30ms
"""

import time
import sys
import os
import subprocess
from typing import Dict, List
import statistics


class Color:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class QuickBenchmark:
    """Szybki benchmark wszystkich komponentów"""

    def __init__(self):
        self.results = {}
        self.passed = []
        self.warnings = []
        self.failed = []

    def print_header(self, text: str):
        """Wydrukuj nagłówek sekcji"""
        print(f"\n{Color.BLUE}{'='*70}{Color.NC}")
        print(f"{Color.BLUE}{text}{Color.NC}")
        print(f"{Color.BLUE}{'='*70}{Color.NC}\n")

    def check(self, name: str, condition: bool, message: str = "",
             warning: bool = False):
        """
        Sprawdź warunek i dodaj do wyników

        Args:
            name: Nazwa testu
            condition: True = pass, False = fail
            message: Dodatkowy komunikat
            warning: True = warning jeśli fail, False = error
        """
        if condition:
            print(f"{Color.GREEN}✓{Color.NC} {name}")
            if message:
                print(f"  {message}")
            self.passed.append(name)
        else:
            if warning:
                print(f"{Color.YELLOW}⚠{Color.NC} {name}")
                if message:
                    print(f"  {message}")
                self.warnings.append(name)
            else:
                print(f"{Color.RED}✗{Color.NC} {name}")
                if message:
                    print(f"  {message}")
                self.failed.append(name)

    def measure_time(self, name: str, func, iterations: int = 100,
                    target_ms: float = None) -> float:
        """
        Zmierz czas wykonania funkcji

        Args:
            name: Nazwa pomiaru
            func: Funkcja do zmierzenia
            iterations: Liczba iteracji
            target_ms: Oczekiwany max czas (ms)

        Returns:
            Średni czas w ms
        """
        times = []

        # Warmup
        for _ in range(10):
            func()

        # Measure
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        avg = statistics.mean(times)
        median = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        p95 = sorted(times)[int(len(times) * 0.95)]

        self.results[name] = {
            'avg': avg,
            'median': median,
            'min': min_time,
            'max': max_time,
            'p95': p95
        }

        # Check against target
        if target_ms:
            passed = avg <= target_ms
            self.check(
                f"{name}: {avg:.2f}ms avg (target: <{target_ms}ms)",
                passed,
                f"  min={min_time:.2f}ms, max={max_time:.2f}ms, p95={p95:.2f}ms",
                warning=not passed
            )
        else:
            print(f"{Color.BLUE}ℹ{Color.NC} {name}: {avg:.2f}ms avg "
                  f"(min={min_time:.2f}, max={max_time:.2f}, p95={p95:.2f})")

        return avg

    # ========================================================================
    # System Checks
    # ========================================================================

    def check_system_config(self):
        """Sprawdź konfigurację systemu"""
        self.print_header("1. System Configuration")

        # Isolated CPUs
        try:
            with open('/sys/devices/system/cpu/isolated', 'r') as f:
                isolated = f.read().strip()
                self.check("Isolated CPUs", bool(isolated),
                          f"CPUs: {isolated}" if isolated else
                          "Add isolcpus=2,3 to /boot/firmware/cmdline.txt")
        except:
            self.check("Isolated CPUs", False,
                      "Cannot read /sys/devices/system/cpu/isolated", warning=True)

        # RT limits
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_RTPRIO)
        self.check("RT priority limits", hard > 0,
                  f"rtprio={hard}" if hard > 0 else
                  "Configure /etc/security/limits.conf")

        # Swap
        try:
            result = subprocess.run(['free', '-m'], capture_output=True, text=True)
            swap_line = [l for l in result.stdout.split('\n') if 'Swap' in l][0]
            swap_total = int(swap_line.split()[1])
            self.check("Swap disabled", swap_total == 0,
                      "Swap is 0" if swap_total == 0 else
                      f"Swap is {swap_total}MB - run: sudo swapoff -a")
        except:
            self.check("Swap check", False, "Cannot check swap", warning=True)

        # CPU governor
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
                governor = f.read().strip()
                self.check("CPU governor", governor == 'performance',
                          f"Governor: {governor}")
        except:
            self.check("CPU governor", False, "Cannot read governor", warning=True)

        # BBR congestion control
        try:
            result = subprocess.run(['sysctl', 'net.ipv4.tcp_congestion_control'],
                                  capture_output=True, text=True)
            bbr_enabled = 'bbr' in result.stdout
            self.check("BBR congestion control", bbr_enabled,
                      "BBR enabled" if bbr_enabled else
                      "Enable BBR in /etc/sysctl.conf")
        except:
            self.check("BBR check", False, "Cannot check BBR", warning=True)

        # Huge pages
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'HugePages_Total' in line:
                        total = int(line.split()[1])
                        self.check("Huge pages", total > 0,
                                  f"Total: {total}" if total > 0 else
                                  "Configure vm.nr_hugepages in sysctl.conf")
                        break
        except:
            self.check("Huge pages", False, "Cannot read /proc/meminfo", warning=True)

    # ========================================================================
    # Memory Performance
    # ========================================================================

    def benchmark_memory(self):
        """Benchmark operacji pamięciowych"""
        self.print_header("2. Memory Performance")

        import numpy as np

        # Frame size: 1280x720x3 (RGB)
        width, height = 1280, 720
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        dest = np.empty_like(frame)

        print(f"Frame size: {width}x{height}x3 = "
              f"{frame.nbytes / 1024 / 1024:.2f} MB\n")

        # np.copy()
        self.measure_time(
            "np.copy()",
            lambda: np.copy(frame),
            iterations=50,
            target_ms=2.0
        )

        # np.copyto()
        self.measure_time(
            "np.copyto()",
            lambda: np.copyto(dest, frame),
            iterations=50,
            target_ms=1.5
        )

        # memoryview
        self.measure_time(
            "memoryview",
            lambda: memoryview(frame),
            iterations=100,
            target_ms=0.01
        )

        # Shared memory (jeśli dostępne)
        try:
            from zero_copy_pipeline import ZeroCopyFrameBuffer

            shm = ZeroCopyFrameBuffer("benchmark", width, height, 3, create=True)

            self.measure_time(
                "Shared memory write",
                lambda: shm.write_frame(frame, frame_id=0),
                iterations=50,
                target_ms=2.0
            )

            shm.unlink()
        except Exception as e:
            print(f"{Color.YELLOW}⚠{Color.NC} Shared memory test skipped: {e}")

    # ========================================================================
    # Camera Performance
    # ========================================================================

    def benchmark_camera(self):
        """Benchmark kamery (jeśli dostępna)"""
        self.print_header("3. Camera Performance")

        try:
            from picamera2 import Picamera2
            from picamera2_low_latency import LowLatencyCamera

            print("Testing camera profiles...\n")

            for profile in ['ultra_low', 'low']:
                try:
                    camera = LowLatencyCamera(
                        camera_num=0,
                        profile=profile,
                        stereo=False
                    )
                    camera.initialize()

                    print(f"Profile: {profile}")

                    # Measure capture latency
                    latencies = []
                    for _ in range(30):
                        start = time.perf_counter()
                        frame, ts = camera.capture_with_timestamp()
                        elapsed = (time.perf_counter() - start) * 1000
                        latencies.append(elapsed)

                    avg_latency = statistics.mean(latencies)

                    target = 10.0 if profile == 'ultra_low' else 20.0
                    self.check(
                        f"Camera {profile} latency",
                        avg_latency <= target,
                        f"{avg_latency:.2f}ms avg (target: <{target}ms)"
                    )

                    camera.close()
                    time.sleep(0.5)

                except Exception as e:
                    print(f"{Color.YELLOW}⚠{Color.NC} Profile {profile} failed: {e}")

        except ImportError:
            print(f"{Color.YELLOW}⚠{Color.NC} Picamera2 not available - skipping camera tests")
        except Exception as e:
            print(f"{Color.YELLOW}⚠{Color.NC} Camera test failed: {e}")

    # ========================================================================
    # Network Performance
    # ========================================================================

    def benchmark_network(self):
        """Benchmark sieci"""
        self.print_header("4. Network Performance")

        try:
            from network_optimizations import LowLatencySocket

            # UDP socket
            print("Testing UDP socket...\n")
            udp_sock = LowLatencySocket(use_udp=True, port=9999)

            test_data = b'X' * 1024  # 1KB

            self.measure_time(
                "UDP send (1KB)",
                lambda: udp_sock.send(test_data, addr=('127.0.0.1', 9999)),
                iterations=100,
                target_ms=0.1
            )

            udp_sock.close()

            # TCP socket
            print("\nTesting TCP socket...\n")
            # Skip TCP test (wymaga server)
            print(f"{Color.BLUE}ℹ{Color.NC} TCP test skipped (requires server)")

        except Exception as e:
            print(f"{Color.YELLOW}⚠{Color.NC} Network test failed: {e}")

        # Ping test (jeśli podano IP)
        print("\n" + "-"*70)
        print("For network RTT test, run:")
        print("  ping -c 100 <oculus_ip> | tail -1")
        print("Target: avg <5ms, max <10ms")

    # ========================================================================
    # Run All
    # ========================================================================

    def run_all(self):
        """Uruchom wszystkie benchmarki"""

        print(f"\n{Color.GREEN}{'='*70}{Color.NC}")
        print(f"{Color.GREEN}VR Streaming System - Quick Benchmark{Color.NC}")
        print(f"{Color.GREEN}{'='*70}{Color.NC}")

        self.check_system_config()
        self.benchmark_memory()
        self.benchmark_camera()
        self.benchmark_network()

        # Summary
        self.print_summary()

    def print_summary(self):
        """Wydrukuj podsumowanie"""

        self.print_header("Summary")

        total = len(self.passed) + len(self.warnings) + len(self.failed)

        print(f"Total checks: {total}")
        print(f"{Color.GREEN}✓ Passed: {len(self.passed)}{Color.NC}")
        print(f"{Color.YELLOW}⚠ Warnings: {len(self.warnings)}{Color.NC}")
        print(f"{Color.RED}✗ Failed: {len(self.failed)}{Color.NC}")

        if self.warnings:
            print(f"\n{Color.YELLOW}Warnings:{Color.NC}")
            for w in self.warnings:
                print(f"  - {w}")

        if self.failed:
            print(f"\n{Color.RED}Failed:{Color.NC}")
            for f in self.failed:
                print(f"  - {f}")

        # Estimate total latency
        print(f"\n{Color.BLUE}{'='*70}{Color.NC}")
        print(f"{Color.BLUE}Estimated Total Latency{Color.NC}")
        print(f"{Color.BLUE}{'='*70}{Color.NC}\n")

        estimated_latency = 0.0

        # Camera
        if 'Camera ultra_low latency' in self.results:
            camera_lat = self.results['Camera ultra_low latency']['avg']
        else:
            camera_lat = 5.0  # Estimate
        estimated_latency += camera_lat
        print(f"Camera capture:        {camera_lat:6.2f} ms")

        # Memory copy
        if 'np.copyto()' in self.results:
            copy_lat = self.results['np.copyto()']['avg']
        else:
            copy_lat = 1.0
        estimated_latency += copy_lat
        print(f"Memory operations:     {copy_lat:6.2f} ms")

        # Encoding (estimate)
        encode_lat = 5.0
        estimated_latency += encode_lat
        print(f"H264 encoding:         {encode_lat:6.2f} ms (estimated)")

        # Network send
        if 'UDP send (1KB)' in self.results:
            net_lat = self.results['UDP send (1KB)']['avg'] * 10  # 1KB -> ~10KB frame
        else:
            net_lat = 3.0
        estimated_latency += net_lat
        print(f"Network send:          {net_lat:6.2f} ms")

        # Network RTT
        rtt_lat = 2.0
        estimated_latency += rtt_lat
        print(f"Network RTT:           {rtt_lat:6.2f} ms (estimated)")

        print(f"─────────────────────────────")
        print(f"TOTAL:                {estimated_latency:6.2f} ms")

        if estimated_latency < 30:
            print(f"\n{Color.GREEN}✓ Target <30ms achieved!{Color.NC}")
        else:
            print(f"\n{Color.YELLOW}⚠ Estimated latency >30ms{Color.NC}")
            print(f"  Recommendations:")
            if camera_lat > 10:
                print(f"  - Optimize camera (use ultra_low profile)")
            if encode_lat > 7:
                print(f"  - Use hardware H264 encoding")
            if net_lat > 5:
                print(f"  - Check network (use 5GHz WiFi, disable power save)")

        print(f"\n{Color.BLUE}Note:{Color.NC} This is an estimate. Run full system for accurate measurement.")
        print(f"      Command: sudo python3 vr_streaming_optimized.py --ip <oculus_ip>")


def main():
    """Main"""

    benchmark = QuickBenchmark()
    benchmark.run_all()

    print()


if __name__ == "__main__":
    main()