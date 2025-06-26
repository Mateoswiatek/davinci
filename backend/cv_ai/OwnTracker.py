#!/usr/bin/env python3
"""
Dual Camera Object Detection and Tracking System
Captures stereo camera feed and performs:
- Object detection using background subtraction
- Feature detection using ORB
- Simple centroid-based object tracking (no external trackers required)
- Motion analysis and trajectory tracking
"""

import cv2
import numpy as np
import time
import logging
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectedObject:
    """Container for detected object information"""
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    area: float
    timestamp: float
    camera: str  # 'left' or 'right'

@dataclass
class TrackedObject:
    """Container for tracked object information using centroid tracking"""
    id: int
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    last_seen: float
    track_history: deque
    camera: str
    velocity: Tuple[float, float]  # dx/dt, dy/dt
    area_history: deque
    confidence_history: deque

class SimpleCentroidTracker:
    """Simple centroid-based tracker that doesn't require opencv-contrib"""

    def __init__(self, max_disappeared=20, max_distance=100):
        self.next_object_id = 1
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, camera):
        """Register a new object"""
        object_id = self.next_object_id
        self.objects[object_id] = TrackedObject(
            id=object_id,
            center=centroid,
            bbox=bbox,
            last_seen=time.time(),
            track_history=deque(maxlen=30),
            camera=camera,
            velocity=(0.0, 0.0),
            area_history=deque(maxlen=10),
            confidence_history=deque(maxlen=10)
        )
        self.disappeared[object_id] = 0
        self.next_object_id += 1

        logger.debug(f"Registered new object {object_id} in {camera} camera")
        return object_id

    def deregister(self, object_id):
        """Remove an object from tracking"""
        if object_id in self.objects:
            del self.objects[object_id]
            del self.disappeared[object_id]
            logger.debug(f"Deregistered object {object_id}")

    def update(self, detections):
        """Update tracker with new detections"""
        current_time = time.time()

        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Initialize centroids from detections
        input_centroids = []
        input_bboxes = []
        input_areas = []

        for detection in detections:
            input_centroids.append(detection.center)
            input_bboxes.append(detection.bbox)
            input_areas.append(detection.area)

        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, input_bboxes[i], detections[i].camera)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id].center for obj_id in object_ids]

            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] -
                               np.array(input_centroids), axis=2)

            # Find minimum distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # Keep track of used indices
            used_row_indices = set()
            used_col_indices = set()

            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                # Update object
                object_id = object_ids[row]
                old_center = self.objects[object_id].center
                new_center = input_centroids[col]

                # Calculate velocity
                dt = current_time - self.objects[object_id].last_seen
                if dt > 0:
                    velocity = ((new_center[0] - old_center[0]) / dt,
                                (new_center[1] - old_center[1]) / dt)
                else:
                    velocity = self.objects[object_id].velocity

                # Update object
                self.objects[object_id].center = new_center
                self.objects[object_id].bbox = input_bboxes[col]
                self.objects[object_id].last_seen = current_time
                self.objects[object_id].velocity = velocity
                self.objects[object_id].track_history.append(new_center)
                self.objects[object_id].area_history.append(input_areas[col])

                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            # Mark unmatched objects as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            # Register new objects
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col], input_bboxes[col], detections[col].camera)

        return self.objects

class DualCameraAnalyzer:
    def __init__(self, camera_source=0, use_yolo=False, use_tracking=True):
        """
        Initialize the dual camera analyzer

        Args:
            camera_source: Camera index or video file path
            use_yolo: Enable YOLO object detection (disabled by default)
            use_tracking: Enable object tracking
        """
        self.camera_source = camera_source
        self.use_yolo = use_yolo
        self.use_tracking = use_tracking

        # Initialize camera
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.left_width = 0
        self.right_width = 0

        # Background subtraction for motion detection
        self.bg_subtractor_left = None
        self.bg_subtractor_right = None

        # Feature detection
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.show_features = False

        # Object tracking using custom centroid tracker
        self.tracker_left = SimpleCentroidTracker(max_disappeared=20, max_distance=80)
        self.tracker_right = SimpleCentroidTracker(max_disappeared=20, max_distance=80)

        # Detection parameters
        self.min_contour_area = 300
        self.max_contour_area = 50000

        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

        # UI state
        self.show_detection = True
        self.show_tracking = True
        self.show_trajectories = True

        # Mouse interaction
        self.mouse_start = None
        self.mouse_end = None
        self.drawing_box = False

    def initialize_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.camera_source}")

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate left and right camera regions
        self.left_width = self.frame_width // 2
        self.right_width = self.frame_width - self.left_width

        logger.info(f"Camera initialized: {self.frame_width}x{self.frame_height}")
        logger.info(f"Left camera: {self.left_width}x{self.frame_height}")
        logger.info(f"Right camera: {self.right_width}x{self.frame_height}")

    def initialize_detection(self):
        """Initialize detection methods"""
        # Initialize background subtractors
        self.bg_subtractor_left = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=200)
        self.bg_subtractor_right = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=200)

        logger.info("Background subtraction initialized for both cameras")

    def split_stereo_frame(self, frame):
        """Split combined stereo frame into left and right images"""
        left_frame = frame[:, :self.left_width]
        right_frame = frame[:, self.left_width:]
        return left_frame, right_frame

    def detect_objects_background_subtraction(self, frame, camera_side="unknown"):
        """Detect moving objects using background subtraction"""
        if camera_side == "left":
            bg_mask = self.bg_subtractor_left.apply(frame)
        else:
            bg_mask = self.bg_subtractor_right.apply(frame)

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)

        # Remove shadows (value 127 in MOG2)
        bg_mask[bg_mask == 127] = 0

        # Find contours
        contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by aspect ratio (avoid very thin/wide objects)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:

                    detected_objects.append(DetectedObject(
                        id=i,
                        class_id=0,
                        class_name="moving_object",
                        confidence=min(area / 2000.0, 1.0),
                        bbox=(x, y, w, h),
                        center=(x + w//2, y + h//2),
                        area=area,
                        timestamp=time.time(),
                        camera=camera_side
                    ))

        return detected_objects, bg_mask

    def detect_features(self, frame):
        """Detect key features in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def calculate_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        self.frame_count += 1

        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def draw_detections(self, frame, detections, camera_side):
        """Draw detected objects on frame"""
        if not self.show_detection:
            return

        for obj in detections:
            x, y, w, h = obj.bbox

            # Color coding: green for left, blue for right
            color = (0, 255, 0) if camera_side == "left" else (255, 0, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw center point
            cv2.circle(frame, obj.center, 4, color, -1)

            # Draw label with confidence and area
            label = f"{obj.class_name}: {obj.confidence:.2f} ({int(obj.area)})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 8),
                          (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_tracking(self, frame, tracked_objects, camera_side):
        """Draw tracked objects on frame"""
        if not self.show_tracking:
            return

        for obj_id, tracked_obj in tracked_objects.items():
            x, y, w, h = tracked_obj.bbox

            # Color coding for tracking: cyan for left, yellow for right
            track_color = (255, 255, 0) if camera_side == "left" else (0, 255, 255)

            # Draw tracking box
            cv2.rectangle(frame, (x, y), (x + w, y + h), track_color, 2)

            # Draw center point
            cv2.circle(frame, tracked_obj.center, 6, track_color, -1)

            # Draw tracker ID and velocity
            velocity_mag = math.sqrt(tracked_obj.velocity[0]**2 + tracked_obj.velocity[1]**2)
            label = f"T{obj_id} v:{velocity_mag:.1f}"
            cv2.putText(frame, label, (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 2)

            # Draw trajectory
            if self.show_trajectories and len(tracked_obj.track_history) > 1:
                points = list(tracked_obj.track_history)
                for i in range(1, len(points)):
                    # Fade older points
                    alpha = i / len(points)
                    color_faded = tuple(int(c * alpha) for c in track_color)
                    cv2.line(frame, points[i-1], points[i], color_faded, 2)

    def draw_features(self, frame, keypoints):
        """Draw detected features on frame"""
        if self.show_features and keypoints:
            # Draw keypoints as small circles
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        return frame

    def draw_motion_mask(self, frame, mask, camera_side):
        """Draw motion detection mask overlay"""
        # Create colored overlay for motion areas
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)

        # Blend with original frame
        alpha = 0.3
        overlay = cv2.addWeighted(frame, 1-alpha, mask_colored, alpha, 0)

        return overlay

    def draw_info_overlay(self, frame, camera_side, detections, tracked_objects):
        """Draw information overlay on frame"""
        height, width = frame.shape[:2]

        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # System info
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Camera: {camera_side.upper()}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Detection info
        detection_count = len(detections) if detections else 0
        cv2.putText(frame, f"Detections: {detection_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Tracking info
        tracker_count = len(tracked_objects) if tracked_objects else 0
        cv2.putText(frame, f"Tracked: {tracker_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Status indicators
        status_y = 110
        detection_status = "ON" if self.show_detection else "OFF"
        tracking_status = "ON" if self.show_tracking else "OFF"

        cv2.putText(frame, f"Det:{detection_status} Track:{tracking_status}", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Controls help
        help_y = height - 100
        help_texts = [
            "Controls: Q-Quit R-Reset D-Detection T-Tracking",
            "F-Features M-Motion SPACE-Trajectories",
            "Click+Drag - Manual selection (future feature)"
        ]

        for i, help_text in enumerate(help_texts):
            cv2.putText(frame, help_text, (10, help_y + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    def process_frame(self, frame):
        """Process a single frame"""
        # Split stereo frame
        left_frame, right_frame = self.split_stereo_frame(frame)

        # Create copies for processing
        left_display = left_frame.copy()
        right_display = right_frame.copy()

        # Object detection using background subtraction
        left_detections, left_mask = self.detect_objects_background_subtraction(left_frame, "left")
        right_detections, right_mask = self.detect_objects_background_subtraction(right_frame, "right")

        # Feature detection
        left_keypoints, left_descriptors = self.detect_features(left_frame)
        right_keypoints, right_descriptors = self.detect_features(right_frame)

        # Object tracking
        tracked_left = {}
        tracked_right = {}

        if self.use_tracking:
            tracked_left = self.tracker_left.update(left_detections)
            tracked_right = self.tracker_right.update(right_detections)

        # Draw motion masks (optional)
        # left_display = self.draw_motion_mask(left_display, left_mask, "left")
        # right_display = self.draw_motion_mask(right_display, right_mask, "right")

        # Draw detections
        self.draw_detections(left_display, left_detections, "left")
        self.draw_detections(right_display, right_detections, "right")

        # Draw tracking
        if self.use_tracking:
            self.draw_tracking(left_display, tracked_left, "left")
            self.draw_tracking(right_display, tracked_right, "right")

        # Draw features
        if self.show_features:
            left_display = self.draw_features(left_display, left_keypoints)
            right_display = self.draw_features(right_display, right_keypoints)

        # Draw info overlays
        self.draw_info_overlay(left_display, "left", left_detections, tracked_left)
        self.draw_info_overlay(right_display, "right", right_detections, tracked_right)

        # Add separator line
        separator = np.ones((self.frame_height, 2, 3), dtype=np.uint8) * 255

        # Combine frames for display
        combined_frame = np.hstack([left_display, separator, right_display])

        return combined_frame, (left_detections, right_detections), (left_keypoints, right_keypoints)

    def setup_mouse_callback(self):
        """Setup mouse callback for manual interaction"""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mouse_start = (x, y)
                self.drawing_box = True
                logger.info(f"Mouse down at {x}, {y}")

            elif event == cv2.EVENT_MOUSEMOVE and self.drawing_box:
                self.mouse_end = (x, y)

            elif event == cv2.EVENT_LBUTTONUP and self.drawing_box:
                self.mouse_end = (x, y)
                self.drawing_box = False
                logger.info(f"Mouse up at {x}, {y}")

                # Future: implement manual tracker initialization

        cv2.setMouseCallback('Dual Camera Analysis', mouse_callback)

    def run(self):
        """Main processing loop"""
        try:
            self.initialize_camera()
            self.initialize_detection()

            # Create window
            cv2.namedWindow('Dual Camera Analysis', cv2.WINDOW_AUTOSIZE)
            self.setup_mouse_callback()

            logger.info("Starting dual camera analysis...")
            logger.info("Background subtraction detection active")
            logger.info("Custom centroid tracking active")
            logger.info("Controls:")
            logger.info("  Q/ESC - Quit")
            logger.info("  R - Reset trackers")
            logger.info("  D - Toggle detection display")
            logger.info("  T - Toggle tracking display")
            logger.info("  F - Toggle feature display")
            logger.info("  M - Toggle motion overlay")
            logger.info("  SPACE - Toggle trajectories")

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break

                # Process frame
                processed_frame, detections, features = self.process_frame(frame)

                # Draw selection box if drawing
                if self.drawing_box and self.mouse_start and self.mouse_end:
                    cv2.rectangle(processed_frame, self.mouse_start, self.mouse_end, (255, 255, 255), 2)

                # Update FPS
                self.calculate_fps()

                # Display result
                cv2.imshow('Dual Camera Analysis', processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):  # Reset trackers
                    self.tracker_left = SimpleCentroidTracker(max_disappeared=20, max_distance=80)
                    self.tracker_right = SimpleCentroidTracker(max_disappeared=20, max_distance=80)
                    logger.info("All trackers reset")
                elif key == ord('d'):  # Toggle detection display
                    self.show_detection = not self.show_detection
                    logger.info(f"Detection display: {'ON' if self.show_detection else 'OFF'}")
                elif key == ord('t'):  # Toggle tracking display
                    self.show_tracking = not self.show_tracking
                    logger.info(f"Tracking display: {'ON' if self.show_tracking else 'OFF'}")
                elif key == ord('f'):  # Toggle feature display
                    self.show_features = not self.show_features
                    logger.info(f"Feature display: {'ON' if self.show_features else 'OFF'}")
                elif key == ord(' '):  # Toggle trajectories
                    self.show_trajectories = not self.show_trajectories
                    logger.info(f"Trajectory display: {'ON' if self.show_trajectories else 'OFF'}")
                elif key == ord('m'):  # Toggle motion overlay
                    # This would toggle motion mask overlay (currently commented out)
                    logger.info("Motion overlay toggle (not implemented)")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Dual Camera Object Detection and Tracking")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--video", type=str,
                        help="Video file path instead of camera")
    parser.add_argument("--min-area", type=int, default=300,
                        help="Minimum object area for detection")
    parser.add_argument("--max-area", type=int, default=50000,
                        help="Maximum object area for detection")

    args = parser.parse_args()

    # Determine camera source
    camera_source = args.video if args.video else args.camera

    print("Dual Camera Object Detection and Tracking System")
    print("=" * 50)
    print(f"Camera source: {camera_source}")
    print("Detection: Background Subtraction (MOG2)")
    print("Tracking: Custom Centroid Tracker")
    print("Features: ORB Detector")
    print(f"Detection area range: {args.min_area} - {args.max_area} pixels")
    print("=" * 50)

    # Create and run analyzer
    analyzer = DualCameraAnalyzer(
        camera_source=camera_source,
        use_yolo=False,  # Disabled by default
        use_tracking=True
    )

    # Set detection parameters
    analyzer.min_contour_area = args.min_area
    analyzer.max_contour_area = args.max_area

    analyzer.run()

if __name__ == "__main__":
    main()