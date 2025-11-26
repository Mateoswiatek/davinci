#!/usr/bin/env python3
"""
Quick YOLO Test - Check what works on your Pi 5

This script quickly tests all available YOLO formats and recommends the best one.

Usage:
    python quick_yolo_test.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Color output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_ok(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_fail(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}→ {text}{Colors.ENDC}")


# Test functions
def test_pytorch():
    """Test PyTorch YOLO"""
    print_header("Testing PyTorch")

    try:
        from ultralytics import YOLO
        print_ok("ultralytics installed")
    except ImportError:
        print_fail("ultralytics not installed")
        print_info("Install: pip install ultralytics")
        return None

    # Check if model exists
    if not Path("yolov8n.pt").exists():
        print_warning("yolov8n.pt not found, will auto-download...")

    try:
        model = YOLO('yolov8n.pt')
        print_ok("Model loaded")

        # Quick benchmark (5 runs)
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        times = []

        print_info("Running quick benchmark (5 iterations)...")
        for _ in range(5):
            start = time.perf_counter()
            model(dummy, verbose=False)
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        fps = 1000 / avg_ms

        print_ok(f"Average: {avg_ms:.1f}ms ({fps:.1f} FPS)")

        if fps < 5:
            print_fail("Too slow for VR!")
            return {'format': 'pytorch', 'fps': fps, 'latency_ms': avg_ms, 'viable': False}
        elif fps < 10:
            print_warning("Marginal - use with frame skipping")
            return {'format': 'pytorch', 'fps': fps, 'latency_ms': avg_ms, 'viable': True}
        else:
            print_ok("Good performance!")
            return {'format': 'pytorch', 'fps': fps, 'latency_ms': avg_ms, 'viable': True}

    except Exception as e:
        print_fail(f"Test failed: {e}")
        return None


def test_coral():
    """Test Coral EdgeTPU"""
    print_header("Testing Coral Edge TPU")

    try:
        from pycoral.utils import edgetpu
        print_ok("pycoral installed")
    except ImportError:
        print_fail("pycoral not installed")
        print_info("Install: sudo apt install python3-pycoral")
        return None

    # Check for EdgeTPU device
    try:
        devices = edgetpu.list_edge_tpus()
        if not devices:
            print_fail("No Edge TPU device found")
            print_info("Connect Coral USB Accelerator")
            return None

        print_ok(f"Found Edge TPU: {devices[0]['type']}")

        # Check for model
        edgetpu_models = list(Path(".").glob("*edgetpu.tflite"))
        if not edgetpu_models:
            print_warning("No EdgeTPU model found")
            print_info("Create with: yolo export model=yolov8n.pt format=edgetpu")
            return None

        model_path = str(edgetpu_models[0])
        print_ok(f"Found model: {model_path}")

        # Load model
        from pycoral.adapters import common
        interpreter = edgetpu.make_interpreter(model_path)
        interpreter.allocate_tensors()
        print_ok("Model loaded on EdgeTPU")

        # Quick benchmark
        input_details = interpreter.get_input_details()[0]
        _, height, width, _ = input_details['shape']

        dummy = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        times = []

        print_info("Running quick benchmark (10 iterations)...")
        for _ in range(10):
            start = time.perf_counter()
            interpreter.set_tensor(input_details['index'], dummy[np.newaxis, ...])
            interpreter.invoke()
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        fps = 1000 / avg_ms

        print_ok(f"Average: {avg_ms:.1f}ms ({fps:.1f} FPS)")

        if fps > 40:
            print_ok("EXCELLENT - Can process every frame!")
            return {'format': 'edgetpu', 'fps': fps, 'latency_ms': avg_ms, 'viable': True}
        elif fps > 20:
            print_ok("Good - Can process every 2nd frame")
            return {'format': 'edgetpu', 'fps': fps, 'latency_ms': avg_ms, 'viable': True}
        else:
            print_warning("Slower than expected")
            return {'format': 'edgetpu', 'fps': fps, 'latency_ms': avg_ms, 'viable': True}

    except Exception as e:
        print_fail(f"Test failed: {e}")
        return None


def test_hailo():
    """Test Hailo-8L"""
    print_header("Testing Hailo-8L AI Kit")

    try:
        from hailo_platform import Device
        print_ok("hailo_platform installed")
    except ImportError:
        print_fail("hailo_platform not installed")
        print_info("Install: sudo apt install hailo-all && pip install hailo-platform")
        return None

    # Check for device
    try:
        devices = Device.scan()
        if not devices:
            print_fail("No Hailo device found")
            print_info("Install Hailo AI Kit M.2 HAT")
            return None

        print_ok(f"Found Hailo device: {devices[0]}")

        # Check for model
        hef_models = list(Path(".").glob("*.hef"))
        if not hef_models:
            print_warning("No HEF model found")
            print_info("Download from Hailo Model Zoo or compile")
            return None

        print_ok(f"Found model: {hef_models[0]}")

        # Quick test (can't easily benchmark without full setup)
        print_ok("Hailo hardware detected and ready!")
        print_info("Expected performance: ~400+ FPS")

        return {'format': 'hailo', 'fps': 400, 'latency_ms': 2.5, 'viable': True, 'estimated': True}

    except Exception as e:
        print_fail(f"Test failed: {e}")
        return None


def test_system():
    """Test system capabilities"""
    print_header("System Information")

    # CPU
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Cortex-A76' in cpuinfo:
                print_ok("CPU: Raspberry Pi 5 (Cortex-A76)")
            else:
                print_warning("CPU: Unknown (not Pi 5?)")

        # Count cores
        import os
        cores = os.cpu_count()
        print_ok(f"CPU Cores: {cores}")

    except:
        pass

    # Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024 * 1024)
                    print_ok(f"Memory: {mem_gb:.1f} GB")
                    if mem_gb < 4:
                        print_warning("Less than 4GB RAM - may struggle with YOLO")
                    break
    except:
        pass

    # Temperature
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read().strip()) / 1000.0
            print_info(f"Temperature: {temp:.1f}°C")
            if temp > 70:
                print_warning("High temperature - add cooling!")
    except:
        pass

    # Check for accelerators
    print("\n" + Colors.BOLD + "Hardware Accelerators:" + Colors.ENDC)

    # USB devices
    import subprocess
    try:
        lsusb = subprocess.check_output(['lsusb'], text=True)
        if 'Global Unichip' in lsusb:
            print_ok("Coral USB Accelerator detected")
        else:
            print_info("No Coral USB detected")
    except:
        pass

    # PCIe
    try:
        lspci = subprocess.check_output(['lspci'], text=True)
        if 'Hailo' in lspci:
            print_ok("Hailo AI Kit detected")
        else:
            print_info("No Hailo device detected")
    except:
        pass


def main():
    print(f"\n{Colors.BOLD}YOLO Quick Test - Raspberry Pi 5{Colors.ENDC}")
    print("This will test all available YOLO formats and recommend the best.\n")

    # System info
    test_system()

    # Test all formats
    results = []

    # PyTorch (always available)
    pt_result = test_pytorch()
    if pt_result:
        results.append(pt_result)

    # Coral EdgeTPU (if hardware present)
    coral_result = test_coral()
    if coral_result:
        results.append(coral_result)

    # Hailo (if hardware present)
    hailo_result = test_hailo()
    if hailo_result:
        results.append(hailo_result)

    # Summary
    print_header("Summary & Recommendations")

    if not results:
        print_fail("No working YOLO setup found!")
        print_info("Install at minimum: pip install ultralytics")
        return 1

    # Sort by FPS
    results.sort(key=lambda x: x['fps'], reverse=True)

    print(f"\n{Colors.BOLD}Available Formats:{Colors.ENDC}\n")
    for r in results:
        estimated = " (estimated)" if r.get('estimated') else ""
        status = "✓" if r['viable'] else "✗"
        print(f"  {status} {r['format']:12} - {r['fps']:6.1f} FPS ({r['latency_ms']:.1f}ms){estimated}")

    # Recommendation
    best = results[0]
    print(f"\n{Colors.BOLD}RECOMMENDATION:{Colors.ENDC}\n")

    if best['format'] == 'hailo':
        print_ok(f"Use Hailo-8L - BEST PERFORMANCE")
        print_info("Command:")
        print(f"  python backend/vr_yolo_streamer_v2.py --model yolov8n.hef --format hef --skip 1")

    elif best['format'] == 'edgetpu':
        print_ok(f"Use Coral EdgeTPU - EXCELLENT for budget")
        print_info("Command:")
        print(f"  python backend/vr_yolo_streamer_v2.py --model model_edgetpu.tflite --format edgetpu --skip 2")

    elif best['format'] == 'pytorch':
        if best['fps'] < 10:
            print_warning("CPU-only - USE FRAME SKIPPING")
            print_info("Recommended command:")
            print(f"  python backend/vr_yolo_streamer_v2.py --model yolov8n.pt --format pt --skip 5")
            print_info("\nConsider upgrading to:")
            print("  - Coral USB Accelerator ($60) - 10x faster")
            print("  - Hailo AI Kit ($70) - 100x faster")
        else:
            print_ok("CPU performance acceptable with frame skipping")
            print_info("Command:")
            print(f"  python backend/vr_yolo_streamer_v2.py --model yolov8n.pt --format pt --skip 3")

    # VR suitability
    print(f"\n{Colors.BOLD}VR Streaming Suitability:{Colors.ENDC}\n")

    if best['fps'] > 60:
        print_ok("Can process EVERY frame in real-time (30 FPS streaming)")
    elif best['fps'] > 30:
        print_ok("Can process every 2nd frame (15 FPS detection)")
    elif best['fps'] > 10:
        print_warning("Need to skip 3-5 frames (6-10 FPS detection)")
    else:
        print_fail("Too slow - need hardware accelerator")

    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}\n")
    print("1. Run full benchmark:")
    print(f"   python backend/yolo_benchmark.py --{best['format']} <model_path> --runs 100")
    print("\n2. Test VR streaming:")
    print("   (see recommendation above)")
    print("\n3. Check latency in production:")
    print("   Watch for 'Total latency' in logs (target: <50ms)")
    print("\nFor detailed setup instructions, see: docs/YOLO_SETUP_GUIDE.md\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
        sys.exit(1)