#!/usr/bin/env python3
"""
Dual Camera Object Detection and Tracking System
Captures stereo camera feed and performs:
- Object detection using YOLO
- Feature detection using ORB/SIFT
- Object tracking using multiple algorithms
- Depth estimation from stereo vision
"""

import cv2
import numpy as np
import time
import logging
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import threading
import queue

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
    timestamp: float
    camera: str  # 'left' or 'right'

@dataclass
class TrackedObject:
    """Container for tracked object information"""
    id: int
    tracker: cv2.Tracker
    bbox: Tuple[int, int, int, int]
    last_seen: float
    track_history: deque
    camera: str

class DualCameraAnalyzer:
    def __init__(self, camera_source=0, use_yolo=True, use_tracking=True):
        """
        Initialize the dual camera analyzer

        Args:
            camera_source: Camera index or video file path
            use_yolo: Enable YOLO object detection
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

        # YOLO detection
        self.net = None
        self.output_layers = None
        self.classes = None

        # Feature detection
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Object tracking
        self.trackers_left = {}
        self.trackers_right = {}
        self.next_tracker_id = 1
        self.max_tracker_age = 30  # frames

        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

        # Threading
        self.frame_queue = queue.Queue(maxsize=5)
        self.is_running = False

    def initialize_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.camera_source}")

        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate left and right camera regions
        self.left_width = self.frame_width // 2
        self.right_width = self.frame_width - self.left_width

        logger.info(f"Camera initialized: {self.frame_width}x{self.frame_height}")
        logger.info(f"Left camera: {self.left_width}x{self.frame_height}")
        logger.info(f"Right camera: {self.right_width}x{self.frame_height}")

    def initialize_yolo(self, config_path="yolo/yolov4.cfg",
                        weights_path="yolo/yolov4.weights",
                        names_path="yolo/coco.names"):
        """
        Initialize YOLO object detection or fallback methods
        """
        # Always initialize background subtractor as fallback
        self.bg_subtractor_left = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50)
        self.bg_subtractor_right = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50)

        # Try to initialize YOLO
        try:
            # Try to load YOLO
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Get output layer names
            layer_names = self.net.getLayerNames()
            try:
                # Try new format first (OpenCV 4.x)
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            except:
                # Fallback to old format (OpenCV 3.x)
                self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

            # Load class names
            with open(names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]

            logger.info(f"YOLO initialized with {len(self.classes)} classes")

        except Exception as e:
            logger.warning(f"Could not load YOLO: {e}")
            logger.info("Using background subtraction for detection")
            self.use_yolo = False

    def split_stereo_frame(self, frame):
        """Split combined stereo frame into left and right images"""
        left_frame = frame[:, :self.left_width]
        right_frame = frame[:, self.left_width:]
        return left_frame, right_frame

    def detect_objects_yolo(self, frame, camera_side="unknown"):
        """Detect objects using YOLO"""
        if not self.use_yolo or self.net is None:
            return []

        height, width = frame.shape[:2]

        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Run detection
        outputs = self.net.forward(self.output_layers)

        # Parse detections
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    # Scale bounding box back to frame size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate top-left corner
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detected_objects = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]] if self.classes else f"class_{class_ids[i]}"

                detected_objects.append(DetectedObject(
                    id=len(detected_objects),
                    class_id=class_ids[i],
                    class_name=class_name,
                    confidence=confidences[i],
                    bbox=(x, y, w, h),
                    center=(x + w//2, y + h//2),
                    timestamp=time.time(),
                    camera=camera_side
                ))

        return detected_objects

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

        # Find contours
        contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []
        min_area = 500  # Minimum area threshold

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)

                detected_objects.append(DetectedObject(
                    id=i,
                    class_id=0,
                    class_name="moving_object",
                    confidence=min(area / 5000.0, 1.0),  # Simple confidence based on size
                    bbox=(x, y, w, h),
                    center=(x + w//2, y + h//2),
                    timestamp=time.time(),
                    camera=camera_side
                ))

        return detected_objects

    def detect_features(self, frame):
        """Detect key features in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match features between two frames"""
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        return matches

    def create_tracker(self):
        """Create a tracker based on available OpenCV version"""
        # Try different tracker types based on OpenCV version
        tracker_types = [
            # OpenCV 4.5.1+
            ('cv2.TrackerCSRT_create', lambda: cv2.TrackerCSRT_create()),
            ('cv2.TrackerKCF_create', lambda: cv2.TrackerKCF_create()),
            # OpenCV 4.5.0 and earlier
            ('cv2.legacy.TrackerCSRT_create', lambda: cv2.legacy.TrackerCSRT_create()),
            ('cv2.legacy.TrackerKCF_create', lambda: cv2.legacy.TrackerKCF_create()),
            ('cv2.legacy.TrackerMOSSE_create', lambda: cv2.legacy.TrackerMOSSE_create()),
            # Even older versions
            ('cv2.TrackerMOSSE_create', lambda: cv2.TrackerMOSSE_create()),
            ('cv2.TrackerBoosting_create', lambda: cv2.TrackerBoosting_create()),
        ]

        for name, creator in tracker_types:
            try:
                tracker = creator()
                logger.info(f"Using tracker: {name}")
                return tracker
            except AttributeError:
                continue
            except Exception as e:
                logger.warning(f"Failed to create {name}: {e}")
                continue

        # If no tracker available, return None
        logger.error("No suitable tracker found in OpenCV installation")
        return None

    def initialize_tracker(self, frame, bbox, camera_side):
        """Initialize a new object tracker"""
        tracker = self.create_tracker()
        if tracker is None:
            logger.warning("Cannot create tracker - tracking disabled")
            return None

        success = tracker.init(frame, bbox)
        if success:
            tracker_id = self.next_tracker_id
            self.next_tracker_id += 1

            tracked_obj = TrackedObject(
                id=tracker_id,
                tracker=tracker,
                bbox=bbox,
                last_seen=time.time(),
                track_history=deque(maxlen=30),
                camera=camera_side
            )

            if camera_side == "left":
                self.trackers_left[tracker_id] = tracked_obj
            else:
                self.trackers_right[tracker_id] = tracked_obj

            logger.info(f"Initialized tracker {tracker_id} for {camera_side} camera")
            return tracker_id

        return None

    def update_trackers(self, frame, camera_side):
        """Update all trackers for the given camera side"""
        trackers = self.trackers_left if camera_side == "left" else self.trackers_right
        current_time = time.time()

        # Update existing trackers
        to_remove = []
        for tracker_id, tracked_obj in trackers.items():
            success, bbox = tracked_obj.tracker.update(frame)

            if success:
                tracked_obj.bbox = tuple(map(int, bbox))
                tracked_obj.last_seen = current_time
                tracked_obj.track_history.append(tracked_obj.bbox)
            else:
                # Mark for removal if tracking failed
                if current_time - tracked_obj.last_seen > 2.0:  # 2 seconds timeout
                    to_remove.append(tracker_id)

        # Remove failed trackers
        for tracker_id in to_remove:
            del trackers[tracker_id]
            logger.info(f"Removed tracker {tracker_id} from {camera_side} camera")

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
        for obj in detections:
            x, y, w, h = obj.bbox

            # Draw bounding box
            color = (0, 255, 0) if camera_side == "left" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f"{obj.class_name}: {obj.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10),
                          (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_trackers(self, frame, camera_side):
        """Draw tracked objects on frame"""
        trackers = self.trackers_left if camera_side == "left" else self.trackers_right

        for tracker_id, tracked_obj in trackers.items():
            x, y, w, h = tracked_obj.bbox

            # Draw tracking box
            color = (0, 255, 255) if camera_side == "left" else (255, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw tracker ID
            cv2.putText(frame, f"T{tracker_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw tracking history
            if len(tracked_obj.track_history) > 1:
                points = [(bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
                          for bbox in tracked_obj.track_history]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)

    def draw_features(self, frame, keypoints):
        """Draw detected features on frame"""
        if keypoints:
            frame = cv2.drawKeypoints(frame, keypoints, None,
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return frame

    def draw_info_overlay(self, frame, camera_side):
        """Draw information overlay on frame"""
        height, width = frame.shape[:2]

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Camera side
        cv2.putText(frame, f"Camera: {camera_side.upper()}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Tracker count
        tracker_count = len(self.trackers_left if camera_side == "left" else self.trackers_right)
        cv2.putText(frame, f"Trackers: {tracker_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Controls
        controls = [
            "Controls:",
            "CLICK+DRAG - Add tracker",
            "R - Reset all trackers",
            "D - Toggle detection",
            "T - Toggle tracking",
            "Q/ESC - Quit"
        ]

        for i, control in enumerate(controls):
            cv2.putText(frame, control, (10, height - 120 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def process_frame(self, frame):
        """Process a single frame"""
        # Split stereo frame
        left_frame, right_frame = self.split_stereo_frame(frame)

        # Object detection
        if self.use_yolo:
            left_detections = self.detect_objects_yolo(left_frame, "left")
            right_detections = self.detect_objects_yolo(right_frame, "right")
        else:
            left_detections = self.detect_objects_background_subtraction(left_frame, "left")
            right_detections = self.detect_objects_background_subtraction(right_frame, "right")

        # Feature detection
        left_keypoints, left_descriptors = self.detect_features(left_frame)
        right_keypoints, right_descriptors = self.detect_features(right_frame)

        # Object tracking
        if self.use_tracking:
            self.update_trackers(left_frame, "left")
            self.update_trackers(right_frame, "right")

            # Initialize new trackers for detected objects (limit to avoid too many)
            max_new_trackers = 3

            for i, detection in enumerate(left_detections[:max_new_trackers]):
                bbox = detection.bbox
                if self.initialize_tracker(left_frame, bbox, "left"):
                    logger.debug(f"New tracker initialized for left detection {i}")

            for i, detection in enumerate(right_detections[:max_new_trackers]):
                bbox = detection.bbox
                if self.initialize_tracker(right_frame, bbox, "right"):
                    logger.debug(f"New tracker initialized for right detection {i}")

        # Draw visualizations
        self.draw_detections(left_frame, left_detections, "left")
        self.draw_detections(right_frame, right_detections, "right")

        if self.use_tracking:
            self.draw_trackers(left_frame, "left")
            self.draw_trackers(right_frame, "right")

        # Optionally draw features
        # left_frame = self.draw_features(left_frame, left_keypoints)
        # right_frame = self.draw_features(right_frame, right_keypoints)

        # Draw info overlays
        self.draw_info_overlay(left_frame, "left")
        self.draw_info_overlay(right_frame, "right")

        # Combine frames for display
        combined_frame = np.hstack([left_frame, right_frame])

        return combined_frame, (left_detections, right_detections), (left_keypoints, right_keypoints)

    def run(self):
        """Main processing loop"""
        try:
            self.initialize_camera()
            if self.use_yolo:
                self.initialize_yolo()

            self.is_running = True

            # Create windows
            cv2.namedWindow('Dual Camera Analysis', cv2.WINDOW_AUTOSIZE)

            logger.info("Starting dual camera analysis...")
            logger.info("Controls:")
            logger.info("  Q/ESC - Quit")
            logger.info("  R - Reset trackers")
            logger.info("  D - Toggle detection")
            logger.info("  T - Toggle tracking")
            logger.info("  SPACE - Manual tracker (click and drag)")

            # Mouse callback for manual tracker initialization
            self.mouse_start = None
            self.mouse_end = None
            self.drawing_box = False

            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.mouse_start = (x, y)
                    self.drawing_box = True
                elif event == cv2.EVENT_MOUSEMOVE:
                    if self.drawing_box:
                        self.mouse_end = (x, y)
                elif event == cv2.EVENT_LBUTTONUP:
                    if self.drawing_box and self.mouse_start and abs(x - self.mouse_start[0]) > 10:
                        self.mouse_end = (x, y)
                        self.drawing_box = False

                        # Determine which camera and create tracker
                        frame_width = processed_frame.shape[1] if 'processed_frame' in locals() else 640
                        left_width = frame_width // 2

                        if x < left_width:  # Left camera
                            bbox = (min(self.mouse_start[0], x), min(self.mouse_start[1], y),
                                    abs(x - self.mouse_start[0]), abs(y - self.mouse_start[1]))
                            if hasattr(self, '_current_left_frame'):
                                self.initialize_tracker(self._current_left_frame, bbox, "left")
                        else:  # Right camera
                            adj_x = x - left_width
                            adj_start_x = self.mouse_start[0] - left_width
                            bbox = (min(adj_start_x, adj_x), min(self.mouse_start[1], y),
                                    abs(adj_x - adj_start_x), abs(y - self.mouse_start[1]))
                            if hasattr(self, '_current_right_frame'):
                                self.initialize_tracker(self._current_right_frame, bbox, "right")

            cv2.setMouseCallback('Dual Camera Analysis', mouse_callback)

            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break

                # Process frame
                processed_frame, detections, features = self.process_frame(frame)

                # Store current frames for manual tracker creation
                left_frame, right_frame = self.split_stereo_frame(frame)
                self._current_left_frame = left_frame.copy()
                self._current_right_frame = right_frame.copy()

                # Draw selection box if drawing
                if hasattr(self, 'drawing_box') and self.drawing_box and self.mouse_start and self.mouse_end:
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
                    self.trackers_left.clear()
                    self.trackers_right.clear()
                    logger.info("All trackers reset")
                elif key == ord('d'):  # Toggle detection
                    self.use_yolo = not self.use_yolo
                    logger.info(f"Detection: {'ON' if self.use_yolo else 'OFF'}")
                elif key == ord('t'):  # Toggle tracking
                    self.use_tracking = not self.use_tracking
                    logger.info(f"Tracking: {'ON' if self.use_tracking else 'OFF'}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False

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
    parser.add_argument("--no-yolo", action="store_true",
                        help="Disable YOLO detection")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable object tracking")

    args = parser.parse_args()

    # Determine camera source
    camera_source = args.video if args.video else args.camera

    print("Dual Camera Object Detection and Tracking System")
    print("=" * 50)
    print(f"Camera source: {camera_source}")
    print(f"YOLO detection: {'Disabled' if args.no_yolo else 'Enabled'}")
    print(f"Object tracking: {'Disabled' if args.no_tracking else 'Enabled'}")
    print("=" * 50)

    # Create and run analyzer
    analyzer = DualCameraAnalyzer(
        camera_source=camera_source,
        use_yolo=not args.no_yolo,
        use_tracking=not args.no_tracking
    )

    analyzer.run()

if __name__ == "__main__":
    main()