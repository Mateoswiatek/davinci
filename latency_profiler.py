#!/usr/bin/env python3
"""
End-to-end latency profiling dla VR streaming pipeline
- Timestamping każdego etapu
- perf, py-spy integration
- Flame graphs
- Real-time monitoring
"""

import time
import sys
import os
import subprocess
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class TimingPoint:
    """Pojedynczy punkt pomiarowy"""
    name: str
    timestamp_ns: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameTiming:
    """Kompletny timing dla jednego frame'a"""
    frame_id: int
    points: List[TimingPoint] = field(default_factory=list)

    def add_point(self, name: str, metadata: Optional[Dict] = None):
        """Dodaj punkt pomiarowy"""
        point = TimingPoint(
            name=name,
            timestamp_ns=time.time_ns(),
            metadata=metadata or {}
        )
        self.points.append(point)

    def get_latency(self, start_point: str, end_point: str) -> float:
        """
        Oblicz latencję między dwoma punktami (w ms)

        Args:
            start_point: Nazwa punktu startowego
            end_point: Nazwa punktu końcowego

        Returns:
            Latencja w milisekundach
        """
        start_ts = None
        end_ts = None

        for point in self.points:
            if point.name == start_point:
                start_ts = point.timestamp_ns
            elif point.name == end_point:
                end_ts = point.timestamp_ns

        if start_ts is None or end_ts is None:
            return 0.0

        return (end_ts - start_ts) / 1_000_000  # ns -> ms

    def get_all_latencies(self) -> Dict[str, float]:
        """Oblicz latencje między kolejnymi punktami"""
        latencies = {}

        for i in range(len(self.points) - 1):
            start = self.points[i]
            end = self.points[i + 1]

            stage_name = f"{start.name} -> {end.name}"
            latency_ms = (end.timestamp_ns - start.timestamp_ns) / 1_000_000

            latencies[stage_name] = latency_ms

        return latencies

    def get_total_latency(self) -> float:
        """Całkowita latencja (pierwszy -> ostatni punkt)"""
        if len(self.points) < 2:
            return 0.0

        first = self.points[0].timestamp_ns
        last = self.points[-1].timestamp_ns

        return (last - first) / 1_000_000  # ms


class LatencyProfiler:
    """
    Profiler latencji dla całego pipeline
    """

    # Standardowe punkty pomiarowe dla VR streaming
    STAGE_CAMERA_CAPTURE = "camera_capture"
    STAGE_CAMERA_READY = "camera_ready"
    STAGE_YOLO_START = "yolo_start"
    STAGE_YOLO_END = "yolo_end"
    STAGE_ENCODE_START = "encode_start"
    STAGE_ENCODE_END = "encode_end"
    STAGE_NETWORK_SEND = "network_send"
    STAGE_NETWORK_ACK = "network_ack"

    def __init__(self, buffer_size: int = 1000):
        """
        Args:
            buffer_size: Liczba frame'ów do zapamiętania (rolling buffer)
        """
        self.buffer_size = buffer_size
        self.frames: deque[FrameTiming] = deque(maxlen=buffer_size)
        self.current_frame: Optional[FrameTiming] = None
        self.frame_counter = 0

    def start_frame(self, frame_id: Optional[int] = None) -> FrameTiming:
        """
        Rozpocznij timing dla nowego frame'a

        Args:
            frame_id: ID frame'a (auto-increment jeśli None)

        Returns:
            FrameTiming object
        """
        if frame_id is None:
            frame_id = self.frame_counter
            self.frame_counter += 1

        self.current_frame = FrameTiming(frame_id=frame_id)
        return self.current_frame

    def mark(self, stage_name: str, metadata: Optional[Dict] = None):
        """
        Zaznacz punkt pomiarowy w bieżącym frame

        Args:
            stage_name: Nazwa etapu (np. STAGE_CAMERA_CAPTURE)
            metadata: Dodatkowe dane (opcjonalne)
        """
        if self.current_frame is None:
            self.start_frame()

        self.current_frame.add_point(stage_name, metadata)

    def end_frame(self):
        """Zakończ timing dla bieżącego frame'a"""
        if self.current_frame:
            self.frames.append(self.current_frame)
            self.current_frame = None

    def get_statistics(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Oblicz statystyki dla ostatnich N frames

        Args:
            last_n: Liczba ostatnich frames (None = wszystkie)

        Returns:
            Dict ze statystykami
        """
        if not self.frames:
            return {}

        frames_to_analyze = list(self.frames)
        if last_n:
            frames_to_analyze = frames_to_analyze[-last_n:]

        # Zbierz latencje dla każdego stage
        stage_latencies: Dict[str, List[float]] = {}

        for frame in frames_to_analyze:
            latencies = frame.get_all_latencies()
            for stage, latency in latencies.items():
                if stage not in stage_latencies:
                    stage_latencies[stage] = []
                stage_latencies[stage].append(latency)

        # Oblicz total latencies
        total_latencies = [frame.get_total_latency() for frame in frames_to_analyze]

        # Statystyki
        stats = {
            'num_frames': len(frames_to_analyze),
            'total': self._compute_stats(total_latencies),
            'stages': {}
        }

        for stage, latencies in stage_latencies.items():
            stats['stages'][stage] = self._compute_stats(latencies)

        return stats

    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        """Oblicz statystyki dla listy wartości"""
        if not values:
            return {}

        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'p95': self._percentile(values, 95),
            'p99': self._percentile(values, 99),
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Oblicz percentyl"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100.0))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def print_statistics(self, last_n: Optional[int] = None):
        """Wydrukuj statystyki w czytelnym formacie"""
        stats = self.get_statistics(last_n)

        if not stats:
            print("No data collected yet")
            return

        print(f"\n=== Latency Statistics (last {stats['num_frames']} frames) ===\n")

        # Total latency
        total = stats['total']
        print(f"TOTAL LATENCY:")
        print(f"  Mean:   {total['mean']:7.2f} ms")
        print(f"  Median: {total['median']:7.2f} ms")
        print(f"  Min:    {total['min']:7.2f} ms")
        print(f"  Max:    {total['max']:7.2f} ms")
        print(f"  StdDev: {total['stdev']:7.2f} ms")
        print(f"  P95:    {total['p95']:7.2f} ms")
        print(f"  P99:    {total['p99']:7.2f} ms")

        # Per-stage latencies
        print(f"\nPER-STAGE LATENCIES:")
        for stage, stage_stats in stats['stages'].items():
            print(f"\n  {stage}:")
            print(f"    Mean:   {stage_stats['mean']:7.2f} ms")
            print(f"    Median: {stage_stats['median']:7.2f} ms")
            print(f"    Min:    {stage_stats['min']:7.2f} ms")
            print(f"    Max:    {stage_stats['max']:7.2f} ms")
            print(f"    P95:    {stage_stats['p95']:7.2f} ms")

    def export_json(self, filename: str, last_n: Optional[int] = None):
        """Eksportuj statystyki do JSON"""
        stats = self.get_statistics(last_n)

        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Statistics exported to {filename}")

    def export_csv(self, filename: str):
        """Eksportuj surowe dane do CSV dla analizy"""
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['frame_id', 'stage', 'latency_ms'])

            # Data
            for frame in self.frames:
                latencies = frame.get_all_latencies()
                for stage, latency in latencies.items():
                    writer.writerow([frame.frame_id, stage, latency])

        print(f"✓ Raw data exported to {filename}")


class PerfProfiler:
    """
    Wrapper dla Linux perf tool
    """

    @staticmethod
    def record(pid: int, duration_seconds: int = 10,
              output_file: str = "perf.data"):
        """
        Nagraj perf data dla procesu

        Args:
            pid: Process ID
            duration_seconds: Czas nagrywania
            output_file: Plik wyjściowy
        """

        cmd = [
            'perf', 'record',
            '-F', '999',  # Sampling frequency (Hz)
            '-g',  # Call graph
            '-p', str(pid),
            '-o', output_file,
            '--', 'sleep', str(duration_seconds)
        ]

        print(f"Recording perf data for PID {pid} ({duration_seconds}s)...")
        print(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Perf data saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print("Make sure 'perf' is installed: sudo apt-get install linux-tools-generic")

    @staticmethod
    def report(perf_data: str = "perf.data"):
        """Wyświetl perf report"""
        cmd = ['perf', 'report', '-i', perf_data]
        subprocess.run(cmd)

    @staticmethod
    def generate_flamegraph(perf_data: str = "perf.data",
                           output_svg: str = "flamegraph.svg"):
        """
        Wygeneruj flame graph z perf data

        Wymaga: https://github.com/brendangregg/FlameGraph
        """

        print(f"\nGenerating flame graph...")

        # Sprawdź czy FlameGraph jest zainstalowane
        flamegraph_path = "/opt/FlameGraph"
        if not os.path.exists(flamegraph_path):
            print(f"\n⚠ FlameGraph not found at {flamegraph_path}")
            print("Install:")
            print("  cd /opt")
            print("  sudo git clone https://github.com/brendangregg/FlameGraph.git")
            return

        # Konwertuj perf.data -> folded format
        folded_file = "perf.folded"
        cmd1 = f"perf script -i {perf_data} | {flamegraph_path}/stackcollapse-perf.pl > {folded_file}"

        # Wygeneruj SVG
        cmd2 = f"{flamegraph_path}/flamegraph.pl {folded_file} > {output_svg}"

        print(f"Executing:")
        print(f"  {cmd1}")
        print(f"  {cmd2}")

        subprocess.run(cmd1, shell=True)
        subprocess.run(cmd2, shell=True)

        print(f"✓ Flame graph saved to {output_svg}")


class PySpy Profiler:
    """
    Wrapper dla py-spy (Python profiler)
    """

    @staticmethod
    def record(pid: int, duration_seconds: int = 10,
              output_file: str = "profile.svg",
              rate: int = 100):
        """
        Profil Python procesu i wygeneruj flame graph

        Args:
            pid: Process ID
            duration_seconds: Czas profilowania
            output_file: Plik SVG z flame graph
            rate: Sampling rate (Hz)
        """

        cmd = [
            'py-spy', 'record',
            '--pid', str(pid),
            '--duration', str(duration_seconds),
            '--rate', str(rate),
            '--format', 'speedscope',
            '--output', output_file
        ]

        print(f"Profiling Python process {pid} with py-spy...")
        print(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Profile saved to {output_file}")
        except FileNotFoundError:
            print("⚠ py-spy not installed")
            print("Install: pip install py-spy")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

    @staticmethod
    def top(pid: int):
        """Live profiling (top-like interface)"""
        cmd = ['py-spy', 'top', '--pid', str(pid)]
        subprocess.run(cmd)


def demo_latency_profiling():
    """Demo użycia LatencyProfiler"""

    print("\n=== Latency Profiling Demo ===\n")

    profiler = LatencyProfiler(buffer_size=100)

    # Symuluj 50 frame'ów
    for i in range(50):
        timing = profiler.start_frame(i)

        # Camera capture
        profiler.mark(LatencyProfiler.STAGE_CAMERA_CAPTURE)
        time.sleep(0.003)  # 3ms

        profiler.mark(LatencyProfiler.STAGE_CAMERA_READY)
        time.sleep(0.001)  # 1ms

        # YOLO
        profiler.mark(LatencyProfiler.STAGE_YOLO_START)
        time.sleep(0.015)  # 15ms
        profiler.mark(LatencyProfiler.STAGE_YOLO_END)

        # Encoding
        profiler.mark(LatencyProfiler.STAGE_ENCODE_START)
        time.sleep(0.005)  # 5ms
        profiler.mark(LatencyProfiler.STAGE_ENCODE_END)

        # Network
        profiler.mark(LatencyProfiler.STAGE_NETWORK_SEND)
        time.sleep(0.002)  # 2ms

        profiler.end_frame()

    # Wyświetl statystyki
    profiler.print_statistics(last_n=50)

    # Eksportuj
    profiler.export_json("latency_stats.json")
    profiler.export_csv("latency_raw.csv")


if __name__ == "__main__":
    print("=== VR Streaming Latency Profiler ===")

    # Demo
    demo_latency_profiling()

    print("\n" + "="*70)
    print("=== Profiling Tools Guide ===\n")

    print("1. LATENCY PROFILING (end-to-end):")
    print("   Use LatencyProfiler in your code:")
    print("   ```python")
    print("   profiler = LatencyProfiler()")
    print("   profiler.start_frame()")
    print("   profiler.mark('camera_capture')")
    print("   # ... your code ...")
    print("   profiler.mark('network_send')")
    print("   profiler.end_frame()")
    print("   profiler.print_statistics()")
    print("   ```")

    print("\n2. PERF (CPU profiling):")
    print("   # Record:")
    print("   sudo perf record -F 999 -g -p <PID> -- sleep 10")
    print("   # Report:")
    print("   sudo perf report")
    print("   # Flame graph:")
    print("   # (see PerfProfiler.generate_flamegraph())")

    print("\n3. PY-SPY (Python profiling):")
    print("   # Install:")
    print("   pip install py-spy")
    print("   # Record:")
    print("   py-spy record --pid <PID> --duration 10 --output profile.svg")
    print("   # Live:")
    print("   py-spy top --pid <PID>")

    print("\n4. CPROFILE (Python built-in):")
    print("   python -m cProfile -o profile.stats your_script.py")
    print("   # Analyze:")
    print("   python -c \"import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)\"")

    print("\n5. HTOP (system monitoring):")
    print("   htop -p <PID>")
    print("   # Press F2 -> Display options -> Show custom thread names")

    print("\n6. FRAME TIMESTAMPING:")
    print("   # Add metadata to each frame:")
    print("   frame_metadata = {")
    print("       'capture_ts': time.time_ns(),")
    print("       'frame_id': frame_id,")
    print("   }")
    print("   # Embed in frame header or side-channel")

    print("\n7. NETWORK LATENCY:")
    print("   # Measure RTT:")
    print("   ping -c 100 <oculus_ip>")
    print("   # TCP dump:")
    print("   sudo tcpdump -i wlan0 -w capture.pcap")
    print("   # Analyze with Wireshark")

    print("\n=== Target Latency Breakdown (for <30ms total) ===")
    print("  Camera capture:       2-5ms")
    print("  YOLO inference:      10-15ms  (YOLOv8n @ 320x320)")
    print("  H264 encoding:        3-5ms   (hardware)")
    print("  Network send:         2-5ms   (UDP, 5GHz WiFi)")
    print("  Network RTT:          1-3ms   (local WiFi)")
    print("  Display (Oculus):     ~10ms   (internal)")
    print("  ────────────────────────────")
    print("  TOTAL:               28-43ms")
    print("\n  To achieve <30ms: skip YOLO or use async processing")