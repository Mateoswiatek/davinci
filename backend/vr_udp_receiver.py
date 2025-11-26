#!/usr/bin/env python3
"""
VR UDP Frame Receiver
Receives and reconstructs frames from UDP streaming server

Compatible with vr_udp_streamer.py
Displays received frames with OpenCV

Author: DaVinci VR Project
Date: 2025-11-26
"""

import socket
import struct
import time
import logging
from collections import defaultdict
from typing import Optional, Dict, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UDPFrameReceiver:
    """
    Receives and reconstructs frames from UDP packets

    Protocol (must match vr_udp_streamer.py):
    1. Header: [frame_id(4B), width(2B), height(2B), channels(1B), total_packets(2B)]
    2. Data:   [frame_id(4B), packet_num(2B), data]
    """

    def __init__(self, port: int = 5000, timeout: float = 1.0):
        self.port = port
        self.timeout = timeout

        # Setup UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', port))
        self.sock.settimeout(timeout)

        # Increase receive buffer
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)  # 1MB

        # Frame reconstruction buffers
        self.frame_buffers: Dict[int, dict] = defaultdict(dict)

        # Statistics
        self.frames_received = 0
        self.packets_received = 0
        self.incomplete_frames = 0
        self.last_frame_time = time.time()

        logger.info(f"UDP receiver listening on port {port}")

    def receive_frame(self) -> Optional[np.ndarray]:
        """
        Receive and reconstruct a complete frame

        Returns:
            numpy array (height, width, channels) or None if timeout
        """
        max_attempts = 100  # Prevent infinite loop

        for _ in range(max_attempts):
            try:
                data, addr = self.sock.recvfrom(65535)
                self.packets_received += 1

            except socket.timeout:
                # Check for incomplete frames and cleanup
                self._cleanup_old_frames()
                continue

            # Parse packet
            if len(data) == 11:  # Header packet
                frame_id, width, height, channels, total_packets = struct.unpack('!IHHBH', data)

                self.frame_buffers[frame_id]['meta'] = {
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'total_packets': total_packets,
                    'start_time': time.time()
                }
                self.frame_buffers[frame_id]['packets'] = {}

            else:  # Data packet
                if len(data) < 6:
                    continue

                frame_id, packet_num = struct.unpack('!IH', data[:6])
                chunk = data[6:]

                if frame_id not in self.frame_buffers:
                    continue

                self.frame_buffers[frame_id]['packets'][packet_num] = chunk

            # Check if frame is complete
            if frame_id in self.frame_buffers:
                meta = self.frame_buffers[frame_id].get('meta')
                if not meta:
                    continue

                packets = self.frame_buffers[frame_id]['packets']
                total_packets = meta['total_packets']

                if len(packets) == total_packets:
                    # Reconstruct frame
                    frame = self._reconstruct_frame(frame_id, meta, packets)

                    # Cleanup
                    del self.frame_buffers[frame_id]

                    if frame is not None:
                        self.frames_received += 1
                        self.last_frame_time = time.time()
                        return frame

        return None

    def _reconstruct_frame(
        self,
        frame_id: int,
        meta: dict,
        packets: Dict[int, bytes]
    ) -> Optional[np.ndarray]:
        """Reconstruct frame from packets"""
        try:
            # Concatenate packets in order
            frame_bytes = b''.join(packets[i] for i in range(meta['total_packets']))

            # Convert to numpy array
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = frame.reshape((meta['height'], meta['width'], meta['channels']))

            return frame

        except Exception as e:
            logger.error(f"Error reconstructing frame {frame_id}: {e}")
            return None

    def _cleanup_old_frames(self, max_age: float = 2.0):
        """Remove incomplete frames older than max_age seconds"""
        current_time = time.time()
        to_remove = []

        for frame_id, buffer in self.frame_buffers.items():
            meta = buffer.get('meta')
            if meta and (current_time - meta['start_time']) > max_age:
                to_remove.append(frame_id)

        for frame_id in to_remove:
            del self.frame_buffers[frame_id]
            self.incomplete_frames += 1

    def get_stats(self) -> dict:
        """Get receiver statistics"""
        return {
            'frames_received': self.frames_received,
            'packets_received': self.packets_received,
            'incomplete_frames': self.incomplete_frames,
            'buffered_frames': len(self.frame_buffers)
        }

    def close(self):
        """Close socket"""
        self.sock.close()
        logger.info("UDP receiver closed")


class VRFrameViewer:
    """Display received VR frames with OpenCV"""

    def __init__(self, port: int = 5000):
        self.receiver = UDPFrameReceiver(port)

        # Display settings
        self.window_name = 'VR UDP Stream'
        self.display_stereo_split = True
        self.show_fps = True

        # Metrics
        self.fps = 0.0
        self.latency_ms = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_fps_count = 0

    def _split_stereo(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split side-by-side stereo frame"""
        height, width, channels = frame.shape
        mid = width // 2

        left = frame[:, :mid, :]
        right = frame[:, mid:, :]

        return left, right

    def _add_overlay(self, frame: np.ndarray):
        """Add performance overlay to frame"""
        if not self.show_fps:
            return

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # FPS
        cv2.putText(
            frame, f"FPS: {self.fps:.1f}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # Latency
        cv2.putText(
            frame, f"Latency: {self.latency_ms:.1f}ms",
            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

        # Stats
        stats = self.receiver.get_stats()
        cv2.putText(
            frame, f"Frames: {stats['frames_received']}",
            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    def _update_metrics(self):
        """Update FPS calculation"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed >= 1.0:
            frames = self.frame_count - self.last_fps_count
            self.fps = frames / elapsed

            self.last_fps_time = current_time
            self.last_fps_count = self.frame_count

    def run(self):
        """Main display loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 400)

        logger.info("VR viewer started. Press 'q' to quit, 's' to toggle stereo split")

        try:
            while True:
                start_time = time.time()

                # Receive frame
                frame = self.receiver.receive_frame()

                if frame is None:
                    continue

                self.frame_count += 1

                # Process frame
                if self.display_stereo_split and frame.shape[1] > frame.shape[0]:
                    # Side-by-side stereo
                    left, right = self._split_stereo(frame)

                    # Stack vertically for display
                    display_frame = np.vstack([left, right])
                else:
                    display_frame = frame

                # Add overlay
                self._add_overlay(display_frame)

                # Display
                cv2.imshow(self.window_name, display_frame)

                # Calculate latency
                self.latency_ms = (time.time() - start_time) * 1000

                # Update metrics
                self._update_metrics()

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    self.display_stereo_split = not self.display_stereo_split
                    logger.info(f"Stereo split: {self.display_stereo_split}")
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                    logger.info(f"FPS overlay: {self.show_fps}")

        except KeyboardInterrupt:
            logger.info("Viewer stopped by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        cv2.destroyAllWindows()
        self.receiver.close()

        # Final stats
        stats = self.receiver.get_stats()
        logger.info("=" * 60)
        logger.info("Final Statistics:")
        logger.info(f"  Frames received: {stats['frames_received']}")
        logger.info(f"  Packets received: {stats['packets_received']}")
        logger.info(f"  Incomplete frames: {stats['incomplete_frames']}")
        logger.info(f"  Final FPS: {self.fps:.1f}")
        logger.info("=" * 60)


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='VR UDP Frame Receiver')
    parser.add_argument(
        '--port', type=int, default=5000,
        help='UDP port to listen on (default: 5000)'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("VR UDP Frame Viewer")
    logger.info("=" * 60)
    logger.info(f"Listening on port: {args.port}")
    logger.info("")
    logger.info("Controls:")
    logger.info("  Q - Quit")
    logger.info("  S - Toggle stereo split view")
    logger.info("  F - Toggle FPS overlay")
    logger.info("=" * 60)

    viewer = VRFrameViewer(port=args.port)
    viewer.run()


if __name__ == "__main__":
    main()