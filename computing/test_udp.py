#!/usr/bin/env python3
"""
UDP Streaming Test

Tests the UDP streaming functionality by:
1. Starting a UDP streamer (sender)
2. Starting a UDP receiver
3. Sending test frames and measuring latency

Usage:
    # Terminal 1 - Receiver
    python test_udp.py --mode receiver --port 5000

    # Terminal 2 - Sender
    python test_udp.py --mode sender --target localhost --port 5000

    # Or run both in one process (for testing)
    python test_udp.py --mode both
"""

import asyncio
import argparse
import time
import sys
import logging
from collections import deque

import numpy as np

sys.path.insert(0, str(__file__).rsplit('/', 1)[0])

from protocols.udp_streamer import UDPStreamer, UDPConfig, UDPReceiver
from protocols.base import CompressionFormat

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def create_test_frame(frame_id: int, width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test frame with embedded frame ID."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient background
    for y in range(height):
        frame[y, :, 0] = int(255 * y / height)  # Red gradient

    # Moving element
    t = time.time()
    x = int((np.sin(t * 2) + 1) * (width - 100) / 2)
    y = int((np.cos(t * 2) + 1) * (height - 100) / 2)
    frame[y:y+50, x:x+50] = [0, 255, 0]  # Green square

    # Add frame ID text (if OpenCV available)
    try:
        import cv2
        cv2.putText(frame, f"Frame {frame_id}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time.time():.3f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    except ImportError:
        pass

    return frame


async def run_sender(target: str, port: int, count: int, fps: int):
    """Run UDP sender test."""
    config = UDPConfig(
        host="0.0.0.0",
        port=port + 1000,  # Different port for sending
        target_host=target,
        target_port=port,
        compression=CompressionFormat.JPEG,
        jpeg_quality=80
    )

    streamer = UDPStreamer(config)

    if not await streamer.start():
        logger.error("Failed to start UDP sender")
        return

    logger.info(f"UDP Sender started, sending to {target}:{port}")

    frame_interval = 1.0 / fps
    latencies = deque(maxlen=100)

    for i in range(count):
        start = time.perf_counter()

        # Create and send frame
        frame = create_test_frame(i)
        success = await streamer.send_frame(frame, frame_id=i, timestamp=time.time())

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        if i % 30 == 0:
            avg = sum(latencies) / len(latencies) if latencies else 0
            stats = streamer.get_stats()
            logger.info(f"Frame {i}: send_time={elapsed:.1f}ms, avg={avg:.1f}ms, "
                       f"bytes={stats.bytes_sent}, fps={stats.current_fps:.1f}")

        # Frame rate control
        sleep_time = frame_interval - (time.perf_counter() - start)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    await streamer.stop()

    # Final stats
    stats = streamer.get_stats()
    logger.info(f"\n=== Sender Stats ===")
    logger.info(f"Frames sent: {stats.frames_sent}")
    logger.info(f"Bytes sent: {stats.bytes_sent}")
    logger.info(f"Avg frame size: {stats.avg_frame_size_bytes} bytes")
    logger.info(f"Avg send time: {stats.avg_send_time_ms:.2f}ms")
    logger.info(f"FPS: {stats.current_fps:.1f}")


def run_receiver(port: int, duration: float):
    """Run UDP receiver test."""
    receiver = UDPReceiver(host="0.0.0.0", port=port)
    receiver.start()

    logger.info(f"UDP Receiver listening on port {port}")

    start_time = time.time()
    latencies = []
    frame_count = 0

    try:
        while time.time() - start_time < duration:
            result = receiver.receive_frame_sync(timeout=0.5)

            if result:
                frame, info = result
                frame_count += 1

                # Calculate latency
                if info.get('timestamp'):
                    latency = (time.time() - info['timestamp']) * 1000
                    latencies.append(latency)

                if frame_count % 30 == 0:
                    avg_latency = sum(latencies[-30:]) / len(latencies[-30:]) if latencies else 0
                    logger.info(f"Frame {info.get('frame_id', '?')}: "
                               f"size={frame.shape}, latency={avg_latency:.1f}ms")

                # Display frame (if OpenCV available)
                try:
                    import cv2
                    cv2.imshow("UDP Receiver", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except ImportError:
                    pass

    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass

    # Stats
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    logger.info(f"\n=== Receiver Stats ===")
    logger.info(f"Frames received: {frame_count}")
    logger.info(f"Duration: {elapsed:.1f}s")
    logger.info(f"FPS: {fps:.1f}")
    if latencies:
        logger.info(f"Avg latency: {sum(latencies)/len(latencies):.1f}ms")
        logger.info(f"Min latency: {min(latencies):.1f}ms")
        logger.info(f"Max latency: {max(latencies):.1f}ms")


async def run_both(port: int, count: int, fps: int):
    """Run both sender and receiver in same process (for testing)."""
    import threading

    # Start receiver in thread
    receiver_thread = threading.Thread(
        target=run_receiver,
        args=(port, count / fps + 5),
        daemon=True
    )
    receiver_thread.start()

    # Wait for receiver to start
    await asyncio.sleep(1)

    # Run sender
    await run_sender("127.0.0.1", port, count, fps)

    # Wait for receiver
    receiver_thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="UDP Streaming Test")
    parser.add_argument("--mode", choices=["sender", "receiver", "both"],
                        default="both", help="Test mode")
    parser.add_argument("--target", default="localhost",
                        help="Target host (sender mode)")
    parser.add_argument("--port", type=int, default=5000,
                        help="UDP port")
    parser.add_argument("--count", type=int, default=300,
                        help="Number of frames to send")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS")
    parser.add_argument("--duration", type=float, default=30,
                        help="Receiver duration (seconds)")

    args = parser.parse_args()

    print("\n" + "="*50)
    print("  UDP STREAMING TEST")
    print("="*50)
    print(f"  Mode: {args.mode}")
    print(f"  Port: {args.port}")
    if args.mode == "sender":
        print(f"  Target: {args.target}")
    print("="*50 + "\n")

    if args.mode == "sender":
        asyncio.run(run_sender(args.target, args.port, args.count, args.fps))
    elif args.mode == "receiver":
        run_receiver(args.port, args.duration)
    else:
        asyncio.run(run_both(args.port, args.count, args.fps))


if __name__ == "__main__":
    main()