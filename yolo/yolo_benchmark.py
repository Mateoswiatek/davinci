#!/usr/bin/env python3
"""
YOLO Model Benchmark Suite for Raspberry Pi 5
Compare different model formats and configurations

Benchmarks:
- PyTorch (.pt)
- ONNX (.onnx)
- NCNN (.param + .bin)
- TFLite (.tflite)
- EdgeTPU (.tflite)
- Hailo (.hef)

Author: DaVinci VR Project
Date: 2025-11-26
"""

import time
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    model_path: str
    model_format: str
    input_size: tuple
    num_runs: int
    warmup_runs: int

    # Timing statistics (ms)
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float

    # Throughput
    fps: float

    # System metrics
    cpu_percent: float
    memory_mb: float
    power_watts: Optional[float] = None
    temperature_c: Optional[float] = None

    # Model info
    model_size_mb: float = 0.0
    num_parameters: Optional[int] = None


class YOLOBenchmark:
    """YOLO model benchmarking utility"""

    def __init__(
        self,
        input_size: tuple = (640, 640),
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        monitor_system: bool = True
    ):
        self.input_size = input_size
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.monitor_system = monitor_system

        # Create dummy input
        self.dummy_image = np.random.randint(
            0, 255,
            (input_size[1], input_size[0], 3),
            dtype=np.uint8
        )

        logger.info(f"Benchmark config:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Warmup runs: {warmup_runs}")
        logger.info(f"  Benchmark runs: {benchmark_runs}")
        logger.info(f"  System monitoring: {monitor_system}")

    def _get_model_size(self, model_path: str) -> float:
        """Get model file size in MB"""
        path = Path(model_path)
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
        return 0.0

    def _get_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature (Pi 5 specific)"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000.0
                return temp
        except:
            return None

    def _get_power_usage(self) -> Optional[float]:
        """Estimate power usage (if available)"""
        # Pi 5 doesn't expose power directly
        # Would need external power meter
        return None

    def _monitor_system_start(self) -> Dict:
        """Start system monitoring"""
        if not HAS_PSUTIL or not self.monitor_system:
            return {}

        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / (1024 * 1024),
            'temperature_c': self._get_cpu_temp()
        }

    def _monitor_system_end(self, start_metrics: Dict) -> Dict:
        """End system monitoring and calculate averages"""
        if not HAS_PSUTIL or not self.monitor_system:
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'temperature_c': None,
                'power_watts': None
            }

        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / (1024 * 1024),
            'temperature_c': self._get_cpu_temp(),
            'power_watts': self._get_power_usage()
        }

    def benchmark_pytorch(
        self,
        model_path: str,
        device: str = "cpu",
        half: bool = False
    ) -> BenchmarkResult:
        """Benchmark PyTorch model"""
        if not HAS_ULTRALYTICS:
            raise RuntimeError("ultralytics not installed")

        logger.info(f"Benchmarking PyTorch: {model_path}")

        # Load model
        model = YOLO(model_path)
        if half:
            model.model.half()

        # Warmup
        logger.info(f"  Warmup ({self.warmup_runs} runs)...")
        for _ in range(self.warmup_runs):
            model(self.dummy_image, verbose=False, device=device)

        # Start monitoring
        start_metrics = self._monitor_system_start()

        # Benchmark
        logger.info(f"  Benchmark ({self.benchmark_runs} runs)...")
        times = []
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            results = model(self.dummy_image, verbose=False, device=device)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        # End monitoring
        end_metrics = self._monitor_system_end(start_metrics)

        # Calculate statistics
        times_array = np.array(times)

        result = BenchmarkResult(
            model_path=model_path,
            model_format="pytorch" + ("-fp16" if half else "-fp32"),
            input_size=self.input_size,
            num_runs=self.benchmark_runs,
            warmup_runs=self.warmup_runs,
            mean_ms=float(np.mean(times_array)),
            std_ms=float(np.std(times_array)),
            min_ms=float(np.min(times_array)),
            max_ms=float(np.max(times_array)),
            median_ms=float(np.median(times_array)),
            p95_ms=float(np.percentile(times_array, 95)),
            p99_ms=float(np.percentile(times_array, 99)),
            fps=1000.0 / float(np.mean(times_array)),
            cpu_percent=end_metrics['cpu_percent'],
            memory_mb=end_metrics['memory_mb'],
            temperature_c=end_metrics['temperature_c'],
            power_watts=end_metrics['power_watts'],
            model_size_mb=self._get_model_size(model_path)
        )

        self._print_result(result)
        return result

    def benchmark_onnx(self, model_path: str) -> BenchmarkResult:
        """Benchmark ONNX model"""
        if not HAS_ULTRALYTICS:
            raise RuntimeError("ultralytics not installed")

        logger.info(f"Benchmarking ONNX: {model_path}")

        # Ultralytics can load ONNX
        model = YOLO(model_path, task='detect')

        # Warmup
        logger.info(f"  Warmup ({self.warmup_runs} runs)...")
        for _ in range(self.warmup_runs):
            model(self.dummy_image, verbose=False)

        # Start monitoring
        start_metrics = self._monitor_system_start()

        # Benchmark
        logger.info(f"  Benchmark ({self.benchmark_runs} runs)...")
        times = []
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            results = model(self.dummy_image, verbose=False)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        # End monitoring
        end_metrics = self._monitor_system_end(start_metrics)

        times_array = np.array(times)

        result = BenchmarkResult(
            model_path=model_path,
            model_format="onnx",
            input_size=self.input_size,
            num_runs=self.benchmark_runs,
            warmup_runs=self.warmup_runs,
            mean_ms=float(np.mean(times_array)),
            std_ms=float(np.std(times_array)),
            min_ms=float(np.min(times_array)),
            max_ms=float(np.max(times_array)),
            median_ms=float(np.median(times_array)),
            p95_ms=float(np.percentile(times_array, 95)),
            p99_ms=float(np.percentile(times_array, 99)),
            fps=1000.0 / float(np.mean(times_array)),
            cpu_percent=end_metrics['cpu_percent'],
            memory_mb=end_metrics['memory_mb'],
            temperature_c=end_metrics['temperature_c'],
            power_watts=end_metrics['power_watts'],
            model_size_mb=self._get_model_size(model_path)
        )

        self._print_result(result)
        return result

    def benchmark_ncnn(self, param_path: str, bin_path: str) -> BenchmarkResult:
        """Benchmark NCNN model"""
        logger.warning("NCNN benchmark not fully implemented yet")
        # TODO: Implement NCNN benchmark
        raise NotImplementedError("NCNN benchmark requires python-ncnn bindings")

    def benchmark_edgetpu(self, model_path: str) -> BenchmarkResult:
        """Benchmark Coral EdgeTPU model"""
        logger.warning("EdgeTPU benchmark not fully implemented yet")
        # TODO: Implement EdgeTPU benchmark
        raise NotImplementedError("EdgeTPU benchmark requires pycoral setup")

    def _print_result(self, result: BenchmarkResult):
        """Pretty print benchmark result"""
        logger.info("=" * 60)
        logger.info(f"Model: {result.model_path}")
        logger.info(f"Format: {result.model_format}")
        logger.info(f"Size: {result.model_size_mb:.1f} MB")
        logger.info("-" * 60)
        logger.info(f"Inference Time:")
        logger.info(f"  Mean:   {result.mean_ms:.2f}ms ± {result.std_ms:.2f}ms")
        logger.info(f"  Median: {result.median_ms:.2f}ms")
        logger.info(f"  P95:    {result.p95_ms:.2f}ms")
        logger.info(f"  P99:    {result.p99_ms:.2f}ms")
        logger.info(f"  Range:  {result.min_ms:.2f}ms - {result.max_ms:.2f}ms")
        logger.info("-" * 60)
        logger.info(f"Throughput: {result.fps:.2f} FPS")
        logger.info("-" * 60)
        logger.info(f"System:")
        logger.info(f"  CPU:    {result.cpu_percent:.1f}%")
        logger.info(f"  Memory: {result.memory_mb:.1f} MB")
        if result.temperature_c:
            logger.info(f"  Temp:   {result.temperature_c:.1f}°C")
        logger.info("=" * 60)

    def save_results(self, results: List[BenchmarkResult], output_path: str):
        """Save benchmark results to JSON"""
        data = {
            'benchmark_config': {
                'input_size': self.input_size,
                'warmup_runs': self.warmup_runs,
                'benchmark_runs': self.benchmark_runs
            },
            'results': [asdict(r) for r in results]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    def compare_results(self, results: List[BenchmarkResult]):
        """Compare multiple benchmark results"""
        if not results:
            return

        logger.info("=" * 80)
        logger.info("BENCHMARK COMPARISON")
        logger.info("=" * 80)

        # Sort by FPS (descending)
        sorted_results = sorted(results, key=lambda x: x.fps, reverse=True)

        # Print table header
        logger.info(
            f"{'Format':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} "
            f"{'FPS':<8} {'CPU %':<8} {'Size (MB)':<10}"
        )
        logger.info("-" * 80)

        # Print rows
        for r in sorted_results:
            logger.info(
                f"{r.model_format:<20} "
                f"{r.mean_ms:>10.1f}  "
                f"{r.p95_ms:>10.1f}  "
                f"{r.fps:>6.1f}  "
                f"{r.cpu_percent:>6.1f}  "
                f"{r.model_size_mb:>8.1f}"
            )

        logger.info("=" * 80)

        # Winner
        fastest = sorted_results[0]
        logger.info(f"FASTEST: {fastest.model_format} @ {fastest.fps:.1f} FPS")

        # Recommendations
        logger.info("\nRECOMMENDATIONS:")
        if fastest.mean_ms < 50:
            logger.info("  ✅ Suitable for VR (< 50ms latency)")
        else:
            logger.info("  ❌ Too slow for VR (> 50ms latency)")
            logger.info("  → Consider frame skipping or hardware accelerator")

        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark YOLO models on Raspberry Pi 5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark single PyTorch model
  python yolo_benchmark.py --pytorch yolov8n.pt

  # Benchmark ONNX model
  python yolo_benchmark.py --onnx yolov8n.onnx

  # Compare multiple models
  python yolo_benchmark.py --pytorch yolov8n.pt --onnx yolov8n.onnx

  # Save results to file
  python yolo_benchmark.py --pytorch yolov8n.pt --output results.json

  # Custom input size and run count
  python yolo_benchmark.py --pytorch yolov8n.pt --size 320 --runs 200
        """
    )

    # Model paths
    parser.add_argument('--pytorch', type=str, help='PyTorch model (.pt)')
    parser.add_argument('--onnx', type=str, help='ONNX model (.onnx)')
    parser.add_argument('--ncnn-param', type=str, help='NCNN param file (.param)')
    parser.add_argument('--ncnn-bin', type=str, help='NCNN bin file (.bin)')
    parser.add_argument('--edgetpu', type=str, help='EdgeTPU model (.tflite)')

    # Benchmark options
    parser.add_argument('--size', type=int, default=640, help='Input size (square)')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup runs')
    parser.add_argument('--runs', type=int, default=100, help='Benchmark runs')
    parser.add_argument('--no-monitor', action='store_true', help='Disable system monitoring')

    # PyTorch options
    parser.add_argument('--device', type=str, default='cpu', help='PyTorch device')
    parser.add_argument('--half', action='store_true', help='Use FP16 for PyTorch')

    # Output
    parser.add_argument('--output', type=str, help='Save results to JSON file')

    args = parser.parse_args()

    # Validate input
    if not any([args.pytorch, args.onnx, args.ncnn_param, args.edgetpu]):
        parser.error("At least one model must be specified")

    if args.ncnn_param and not args.ncnn_bin:
        parser.error("NCNN requires both --ncnn-param and --ncnn-bin")

    # Create benchmark
    benchmark = YOLOBenchmark(
        input_size=(args.size, args.size),
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        monitor_system=not args.no_monitor
    )

    results = []

    # Run benchmarks
    try:
        if args.pytorch:
            result = benchmark.benchmark_pytorch(
                args.pytorch,
                device=args.device,
                half=args.half
            )
            results.append(result)

        if args.onnx:
            result = benchmark.benchmark_onnx(args.onnx)
            results.append(result)

        if args.ncnn_param:
            result = benchmark.benchmark_ncnn(args.ncnn_param, args.ncnn_bin)
            results.append(result)

        if args.edgetpu:
            result = benchmark.benchmark_edgetpu(args.edgetpu)
            results.append(result)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1

    # Compare results
    if len(results) > 1:
        benchmark.compare_results(results)

    # Save results
    if args.output:
        benchmark.save_results(results, args.output)

    return 0


if __name__ == "__main__":
    exit(main())