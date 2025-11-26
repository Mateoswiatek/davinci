#!/usr/bin/env python3
"""
Ultra-Low-Latency VR Streaming Server
Uses raw UDP streaming for <30ms latency

Target: Raspberry Pi 5 + Arducam 2560x800 → Oculus VR
Latency: ~20-30ms
FPS: 30 stable

Author: DaVinci VR Project
Date: 2025-11-26
"""

import time
import socket
import struct
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
from picamera2 import Picamera2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Streaming configuration"""
    width: int = 2560
    height: int = 800
    fps: int = 30
    format: str = "RGB888"
    vr_host: str = "192.168.1.100"
    vr_port: int = 5000
    max_packet_size: int = 8192


class UDPFrameStreamer:
    """
    Ultra-low-latency UDP frame streaming

    Protocol:
    1. Header packet: [frame_id(4B), width(2B), height(2B), channels(1B), total_packets(2B)]
    2. Data packets: [frame_id(4B), packet_num(2B), data]
    """

    def __init__(self, config: StreamConfig):
        self.config = config

        # Setup UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Increase send buffer to 1MB (reduce packet loss)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2**20)

        self.dest = (config.vr_host, config.vr_port)
        self.frame_counter = 0

        logger.info(f"UDP streamer initialized → {config.vr_host}:{config.vr_port}")
        logger.info(f"Max packet size: {config.max_packet_size} bytes")

    def send_frame(self, frame: np.ndarray) -> int:
        """
        Send frame via UDP in chunks

        Args:
            frame: numpy array (height, width, channels)

        Returns:
            Number of packets sent
        """
        height, width, channels = frame.shape
        frame_bytes = frame.tobytes()
        total_size = len(frame_bytes)

        # Calculate packets
        data_per_packet = self.config.max_packet_size - 6  # Header overhead
        total_packets = (total_size + data_per_packet - 1) // data_per_packet

        frame_id = self.frame_counter
        self.frame_counter = (self.frame_counter + 1) & 0xFFFFFFFF

        # Send header
        header = struct.pack('!IHHBH', frame_id, width, height, channels, total_packets)
        self.sock.sendto(header, self.dest)

        # Send data packets
        packets_sent = 1
        for i in range(total_packets):
            start = i * data_per_packet
            end = min((i + 1) * data_per_packet, total_size)
            chunk = frame_bytes[start:end]

            packet = struct.pack('!IH', frame_id, i) + chunk
            self.sock.sendto(packet, self.dest)
            packets_sent += 1

        return packets_sent

    def close(self):
        """Close socket"""
        self.sock.close()
        logger.info("UDP streamer closed")


class VRCameraStreamer:
    """
    Main VR camera streaming pipeline
    Handles capture, optional processing, and UDP streaming
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self.streamer = UDPFrameStreamer(config)

        # Initialize camera
        self.picam2 = Picamera2()
        self._configure_camera()

        # Metrics
        self.fps = 0.0
        self.latency_ms = 0.0
        self.frames_sent = 0
        self.last_metric_time = time.time()
        self.last_metric_frames = 0

    def _configure_camera(self):
        """Configure camera for low-latency VR capture"""
        frame_duration_us = int(1_000_000 / self.config.fps)

        config = self.picam2.create_video_configuration(
            main={
                "size": (self.config.width, self.config.height),
                "format": self.config.format
            },
            buffer_count=2,  # Double buffering - minimal latency
            controls={
                "FrameDurationLimits": (frame_duration_us, frame_duration_us),
                "ExposureTime": 10000,  # 10ms exposure
                "AnalogueGain": 2.0,    # Compensate for short exposure
            }
        )

        self.picam2.configure(config)
        logger.info(f"Camera configured: {self.config.width}x{self.config.height} @ {self.config.fps}fps")

    def _update_metrics(self):
        """Update and log performance metrics"""
        current_time = time.time()
        elapsed = current_time - self.last_metric_time

        if elapsed >= 5.0:  # Report every 5 seconds
            frames = self.frames_sent - self.last_metric_frames
            self.fps = frames / elapsed

            logger.info(
                f"Performance: {self.fps:.1f} FPS | "
                f"Latency: {self.latency_ms:.1f}ms | "
                f"Total frames: {self.frames_sent}"
            )

            self.last_metric_time = current_time
            self.last_metric_frames = self.frames_sent

    def run(self):
        """
        Main streaming loop

        Expected latency breakdown:
        - Capture: ~2-5ms
        - UDP send: ~10-15ms
        - Network: ~5-10ms (LAN)
        Total: ~20-30ms
        """
        self.picam2.start()
        logger.info("VR streaming started")

        frame_time = 1.0 / self.config.fps

        try:
            while True:
                start_time = time.time()

                # 1. Capture frame (zero-copy from camera buffer)
                frame = self.picam2.capture_array("main")

                # 2. Send via UDP
                packets_sent = self.streamer.send_frame(frame)

                # 3. Update metrics
                self.frames_sent += 1
                elapsed = time.time() - start_time
                self.latency_ms = elapsed * 1000

                # 4. Maintain target FPS
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                self._update_metrics()

        except KeyboardInterrupt:
            logger.info("Streaming stopped by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.picam2.stop()
        self.streamer.close()
        logger.info(f"Total frames sent: {self.frames_sent}")


def main():
    """Entry point"""
    # Configuration for VR streaming
    config = StreamConfig(
        width=2560,
        height=800,
        fps=30,
        format="RGB888",
        vr_host="192.168.1.100",  # Update with your VR headset IP
        vr_port=5000,
        max_packet_size=8192
    )

    logger.info("=" * 60)
    logger.info("VR UDP Streaming Server")
    logger.info("=" * 60)
    logger.info(f"Resolution: {config.width}x{config.height}")
    logger.info(f"FPS: {config.fps}")
    logger.info(f"Format: {config.format}")
    logger.info(f"Destination: {config.vr_host}:{config.vr_port}")
    logger.info("=" * 60)

    # Calculate bandwidth
    bytes_per_frame = config.width * config.height * 3  # RGB888
    mbps = (bytes_per_frame * config.fps * 8) / 1_000_000
    logger.info(f"Expected bandwidth: {mbps:.0f} Mbps")

    if mbps > 900:
        logger.warning("Bandwidth exceeds Gigabit Ethernet! Consider:")
        logger.warning("  - Lower resolution (e.g., 1280x400)")
        logger.warning("  - Lower FPS (e.g., 15-20)")
        logger.warning("  - YUV420 format instead of RGB888")

    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Start streaming
    streamer = VRCameraStreamer(config)
    streamer.run()


if __name__ == "__main__":
    main()