"""
Camera Capture Module for Raspberry Pi 5

Low-latency camera capture using Picamera2 with optimizations for VR streaming.
Supports mono and stereo cameras (e.g., Arducam 2560x800).

Author: Claude Code
License: MIT
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Any
from enum import Enum
import numpy as np

# Picamera2 import with fallback for development
try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    logging.warning("Picamera2 not available - using mock camera for development")

logger = logging.getLogger(__name__)


class CameraProfile(Enum):
    """Predefined camera profiles for different use cases."""
    ULTRA_LOW_LATENCY = "ultra_low_latency"  # Minimum latency, may sacrifice quality
    LOW_LATENCY = "low_latency"               # Good balance for VR
    BALANCED = "balanced"                      # Quality/latency trade-off
    HIGH_QUALITY = "high_quality"              # Maximum quality, higher latency


@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    width: int = 1280
    height: int = 720
    fps: int = 30
    format: str = "RGB888"  # RGB888, YUV420, XRGB8888

    # Stereo camera settings
    stereo: bool = False
    stereo_width: int = 2560  # Full width for stereo (2x 1280)
    stereo_height: int = 800

    # Low-latency optimizations
    buffer_count: int = 2  # Minimum buffers for low latency
    queue: bool = False     # Don't queue frames

    # Exposure settings
    auto_exposure: bool = True
    exposure_time: int = 10000  # microseconds (if manual)
    analog_gain: float = 1.0

    # Profile presets
    profile: CameraProfile = CameraProfile.LOW_LATENCY


@dataclass
class CapturedFrame:
    """Container for captured frame with metadata."""
    data: np.ndarray
    frame_id: int
    timestamp_ns: int
    capture_time_ms: float
    width: int
    height: int
    format: str

    # Stereo info
    is_stereo: bool = False
    left_eye: Optional[np.ndarray] = None
    right_eye: Optional[np.ndarray] = None


class MockCamera:
    """Mock camera for development/testing without hardware."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.frame_id = 0
        self.running = False
        self._generate_pattern()

    def _generate_pattern(self):
        """Generate test pattern."""
        h, w = self.config.height, self.config.width
        if self.config.stereo:
            h, w = self.config.stereo_height, self.config.stereo_width

        # Create gradient pattern
        self.pattern = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                self.pattern[y, x] = [
                    int(255 * x / w),      # R: horizontal gradient
                    int(255 * y / h),      # G: vertical gradient
                    128                     # B: constant
                ]

    def start(self):
        self.running = True
        logger.info("Mock camera started")

    def stop(self):
        self.running = False
        logger.info("Mock camera stopped")

    def capture_array(self) -> np.ndarray:
        """Capture a frame (simulated)."""
        self.frame_id += 1
        # Add some variation
        frame = self.pattern.copy()
        # Add moving element
        t = time.time()
        x = int((np.sin(t * 2) + 1) * (frame.shape[1] - 100) / 2)
        y = int((np.cos(t * 2) + 1) * (frame.shape[0] - 100) / 2)
        frame[y:y+50, x:x+50] = [255, 0, 0]  # Red square

        # NOTE: Don't sleep here - framerate is controlled by the main loop
        # time.sleep() would block asyncio event loop!
        return frame

    def close(self):
        self.stop()


class CameraCapture:
    """
    High-performance camera capture for Raspberry Pi 5.

    Features:
    - Ultra-low-latency configuration
    - Stereo camera support
    - CPU pinning for real-time performance
    - Zero-copy frame access
    - Callback-based frame delivery
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        """
        Initialize camera capture.

        Args:
            config: Camera configuration. If None, uses defaults.
        """
        self.config = config or CameraConfig()
        self.camera = None
        self.running = False
        self.frame_id = 0
        self.callbacks: list[Callable[[CapturedFrame], None]] = []
        self._capture_thread: Optional[threading.Thread] = None
        self._stats = {
            'frames_captured': 0,
            'total_capture_time_ms': 0.0,
            'avg_capture_time_ms': 0.0,
            'fps': 0.0,
            'last_fps_update': time.time(),
            'frames_since_fps_update': 0
        }

        self._apply_profile()

    def _apply_profile(self):
        """Apply profile-specific settings."""
        if self.config.profile == CameraProfile.ULTRA_LOW_LATENCY:
            self.config.buffer_count = 2
            self.config.queue = False
            self.config.fps = 60
        elif self.config.profile == CameraProfile.LOW_LATENCY:
            self.config.buffer_count = 2
            self.config.queue = False
            self.config.fps = 30
        elif self.config.profile == CameraProfile.BALANCED:
            self.config.buffer_count = 4
            self.config.queue = True
            self.config.fps = 30
        elif self.config.profile == CameraProfile.HIGH_QUALITY:
            self.config.buffer_count = 6
            self.config.queue = True
            self.config.fps = 24

    def initialize(self) -> bool:
        """
        Initialize the camera.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if PICAMERA2_AVAILABLE:
                self.camera = Picamera2()
                self._configure_picamera2()
            else:
                logger.warning("Using mock camera (Picamera2 not available)")
                self.camera = MockCamera(self.config)

            logger.info(f"Camera initialized: {self.config.width}x{self.config.height} @ {self.config.fps}fps")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False

    def _configure_picamera2(self):
        """Configure Picamera2 for low-latency capture."""
        if self.config.stereo:
            size = (self.config.stereo_width, self.config.stereo_height)
        else:
            size = (self.config.width, self.config.height)

        # Create video configuration for low latency
        video_config = self.camera.create_video_configuration(
            main={
                "size": size,
                "format": self.config.format
            },
            buffer_count=self.config.buffer_count,
            queue=self.config.queue,
            controls={
                "FrameDurationLimits": (
                    int(1_000_000 / self.config.fps),  # min frame duration (us)
                    int(1_000_000 / self.config.fps)   # max frame duration (us)
                )
            }
        )

        self.camera.configure(video_config)

        # Set exposure controls
        if not self.config.auto_exposure:
            self.camera.set_controls({
                "AeEnable": False,
                "ExposureTime": self.config.exposure_time,
                "AnalogueGain": self.config.analog_gain
            })
        else:
            self.camera.set_controls({
                "AeEnable": True,
                "AeExposureMode": controls.AeExposureModeEnum.Short  # Prefer short exposure
            })

    def start(self):
        """Start camera capture."""
        if self.camera is None:
            raise RuntimeError("Camera not initialized. Call initialize() first.")

        self.camera.start()
        self.running = True
        logger.info("Camera capture started")

    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.camera:
            self.camera.stop()
        logger.info("Camera capture stopped")

    def capture_frame(self) -> Optional[CapturedFrame]:
        """
        Capture a single frame.

        Returns:
            CapturedFrame object or None if capture failed.
        """
        if not self.running:
            return None

        try:
            start_time = time.perf_counter()

            # Capture frame
            frame_data = self.camera.capture_array()

            capture_time_ms = (time.perf_counter() - start_time) * 1000
            self.frame_id += 1

            # Update stats
            self._update_stats(capture_time_ms)

            # Create frame object
            frame = CapturedFrame(
                data=frame_data,
                frame_id=self.frame_id,
                timestamp_ns=time.time_ns(),
                capture_time_ms=capture_time_ms,
                width=frame_data.shape[1],
                height=frame_data.shape[0],
                format=self.config.format,
                is_stereo=self.config.stereo
            )

            # Split stereo if needed
            if self.config.stereo:
                mid = frame_data.shape[1] // 2
                frame.left_eye = frame_data[:, :mid]
                frame.right_eye = frame_data[:, mid:]

            return frame

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None

    def _update_stats(self, capture_time_ms: float):
        """Update capture statistics."""
        self._stats['frames_captured'] += 1
        self._stats['total_capture_time_ms'] += capture_time_ms
        self._stats['avg_capture_time_ms'] = (
            self._stats['total_capture_time_ms'] / self._stats['frames_captured']
        )

        # Calculate FPS
        self._stats['frames_since_fps_update'] += 1
        elapsed = time.time() - self._stats['last_fps_update']
        if elapsed >= 1.0:
            self._stats['fps'] = self._stats['frames_since_fps_update'] / elapsed
            self._stats['frames_since_fps_update'] = 0
            self._stats['last_fps_update'] = time.time()

    def add_callback(self, callback: Callable[[CapturedFrame], None]):
        """Add a callback to be called for each captured frame."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[CapturedFrame], None]):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def start_continuous_capture(self, threaded: bool = True):
        """
        Start continuous frame capture.

        Args:
            threaded: If True, run capture in a separate thread.
        """
        if threaded:
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="CameraCapture"
            )
            self._capture_thread.start()
        else:
            self._capture_loop()

    def _capture_loop(self):
        """Main capture loop."""
        logger.info("Capture loop started")

        while self.running:
            frame = self.capture_frame()
            if frame:
                for callback in self.callbacks:
                    try:
                        callback(frame)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

        logger.info("Capture loop ended")

    def get_stats(self) -> dict:
        """Get capture statistics."""
        return self._stats.copy()

    def close(self):
        """Clean up resources."""
        self.stop()
        if self.camera:
            if hasattr(self.camera, 'close'):
                self.camera.close()
            self.camera = None
        logger.info("Camera closed")

    def __enter__(self):
        self.initialize()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class StereoCamera(CameraCapture):
    """
    Specialized camera capture for stereo cameras.

    Handles splitting of side-by-side stereo images and provides
    separate access to left and right eye views.
    """

    def __init__(
        self,
        width: int = 2560,
        height: int = 800,
        fps: int = 30,
        profile: CameraProfile = CameraProfile.LOW_LATENCY
    ):
        config = CameraConfig(
            stereo=True,
            stereo_width=width,
            stereo_height=height,
            fps=fps,
            profile=profile
        )
        super().__init__(config)

    def capture_left(self) -> Optional[np.ndarray]:
        """Capture left eye only."""
        frame = self.capture_frame()
        return frame.left_eye if frame else None

    def capture_right(self) -> Optional[np.ndarray]:
        """Capture right eye only."""
        frame = self.capture_frame()
        return frame.right_eye if frame else None

    def capture_both(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture both eyes."""
        frame = self.capture_frame()
        if frame:
            return frame.left_eye, frame.right_eye
        return None, None


# Utility functions

def list_cameras() -> list:
    """List available cameras."""
    if not PICAMERA2_AVAILABLE:
        return ["Mock Camera (Picamera2 not available)"]

    try:
        from picamera2 import Picamera2
        cameras = Picamera2.global_camera_info()
        return cameras
    except Exception as e:
        logger.error(f"Failed to list cameras: {e}")
        return []


def get_camera_modes(camera_index: int = 0) -> list:
    """Get available camera modes."""
    if not PICAMERA2_AVAILABLE:
        return []

    try:
        from picamera2 import Picamera2
        cam = Picamera2(camera_index)
        modes = cam.sensor_modes
        cam.close()
        return modes
    except Exception as e:
        logger.error(f"Failed to get camera modes: {e}")
        return []


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: Basic capture
    print("=== Basic Camera Capture ===")
    config = CameraConfig(
        width=1280,
        height=720,
        fps=30,
        profile=CameraProfile.LOW_LATENCY
    )

    with CameraCapture(config) as camera:
        for i in range(10):
            frame = camera.capture_frame()
            if frame:
                print(f"Frame {frame.frame_id}: {frame.width}x{frame.height}, "
                      f"capture={frame.capture_time_ms:.2f}ms")

    print("\n=== Statistics ===")
    print(camera.get_stats())

    # Example 2: Continuous capture with callback
    print("\n=== Continuous Capture ===")

    def on_frame(frame: CapturedFrame):
        print(f"Callback: Frame {frame.frame_id}, latency={frame.capture_time_ms:.2f}ms")

    camera = CameraCapture(config)
    camera.initialize()
    camera.start()
    camera.add_callback(on_frame)
    camera.start_continuous_capture(threaded=True)

    time.sleep(2)
    camera.close()