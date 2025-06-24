#!/usr/bin/env python3
"""
VR Motion and Feature Detection Client
Captures dual camera stream and performs:
- Feature point detection (corners, keypoints)
- Motion detection with object tracking
- Background dimming option
- Separate processing for left and right camera feeds
"""

import asyncio
import websockets
import json
import base64
import io
import cv2
import numpy as np
import time
import logging
from PIL import Image
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MotionFeatureDetector:
    def __init__(self, server_host='192.168.88.252', server_port=8765):
        self.server_host = server_host
        self.server_port = server_port
        self.websocket = None
        self.is_running = False

        # Frame processing
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        # Detection settings
        self.enable_motion_detection = True
        self.enable_feature_detection = True
        self.enable_background_dimming = False
        self.motion_threshold = 2000  # Minimum contour area for motion detection

        # Background subtraction for motion detection
        self.bg_subtractor_left = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        self.bg_subtractor_right = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )

        # Feature detection - SIFT detector
        self.sift_detector = cv2.SIFT_create(nfeatures=100)

        # Corner detection parameters
        self.corner_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3
        )

        # Previous frames for optical flow
        self.prev_left = None
        self.prev_right = None

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Color schemes
        self.colors = {
            'motion': (0, 255, 0),      # Green for motion
            'features': (255, 0, 0),    # Blue for features
            'corners': (0, 255, 255),   # Yellow for corners
            'tracking': (255, 255, 0),  # Cyan for tracking
            'background': 0.3           # Dimming factor
        }

    async def connect_to_server(self):
        """Connect to the camera server"""
        uri = f"ws://{self.server_host}:{self.server_port}"
        logger.info(f"Connecting to camera server at {uri}")

        try:
            self.websocket = await websockets.connect(uri)
            logger.info("Connected to camera server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False

    def decode_image(self, base64_data):
        """Decode base64 image data to OpenCV format"""
        try:
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes))
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

    def split_dual_camera_feed(self, frame):
        """Split the dual camera feed into left and right images"""
        if frame is None:
            return None, None

        height, width = frame.shape[:2]
        mid_point = width // 2

        left_frame = frame[:, :mid_point]
        right_frame = frame[:, mid_point:]

        return left_frame, right_frame

    def detect_motion(self, frame, bg_subtractor, side=""):
        """Detect motion using background subtraction"""
        if frame is None:
            return [], None

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        motion_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.motion_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                motion_objects.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour
                })

        return motion_objects, fg_mask

    def detect_features(self, frame):
        """Detect feature points using SIFT"""
        if frame is None:
            return [], []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # SIFT keypoints
        keypoints, descriptors = self.sift_detector.detectAndCompute(gray, None)

        # Good features to track (corners)
        corners = cv2.goodFeaturesToTrack(gray, **self.corner_params)

        return keypoints, corners

    def track_optical_flow(self, current_frame, prev_frame):
        """Track features using Lucas-Kanade optical flow"""
        if prev_frame is None or current_frame is None:
            return []

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Detect features in previous frame
        prev_corners = cv2.goodFeaturesToTrack(prev_gray, **self.corner_params)

        if prev_corners is not None:
            # Calculate optical flow
            next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, current_gray, prev_corners, None, **self.lk_params
            )

            # Select good points
            good_new = next_corners[status == 1]
            good_old = prev_corners[status == 1]

            tracks = []
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)

                # Calculate motion vector length
                motion_length = np.sqrt((a - c)**2 + (b - d)**2)

                if motion_length > 2:  # Filter out small movements
                    tracks.append({
                        'start': (c, d),
                        'end': (a, b),
                        'length': motion_length
                    })

            return tracks

        return []

    def apply_background_dimming(self, frame, motion_objects, motion_mask):
        """Apply background dimming effect"""
        if not self.enable_background_dimming or frame is None:
            return frame

        result = frame.copy()

        if motion_mask is not None:
            # Create inverse mask (background areas)
            background_mask = cv2.bitwise_not(motion_mask)

            # Dim the background
            background_dimmed = (result * self.colors['background']).astype(np.uint8)

            # Apply dimming only to background areas
            result = cv2.bitwise_and(result, result, mask=motion_mask)
            background_part = cv2.bitwise_and(background_dimmed, background_dimmed, mask=background_mask)

            result = cv2.add(result, background_part)

        return result

    def draw_detections(self, frame, motion_objects, keypoints, corners, optical_flow_tracks, side=""):
        """Draw all detections on the frame"""
        if frame is None:
            return None

        result = frame.copy()

        # Draw motion detection
        if self.enable_motion_detection and motion_objects:
            for obj in motion_objects:
                x, y, w, h = obj['bbox']
                area = obj['area']

                # Draw bounding box
                cv2.rectangle(result, (x, y), (x + w, y + h), self.colors['motion'], 2)

                # Draw area label
                cv2.putText(result, f"Motion: {int(area)}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['motion'], 1)

        # Draw SIFT keypoints
        if self.enable_feature_detection and keypoints:
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(result, (x, y), 3, self.colors['features'], -1)

        # Draw corner features
        if self.enable_feature_detection and corners is not None:
            for corner in corners:
                x, y = int(corner[0][0]), int(corner[0][1])
                cv2.circle(result, (x, y), 4, self.colors['corners'], 2)

        # Draw optical flow tracks
        if optical_flow_tracks:
            for track in optical_flow_tracks:
                start_point = track['start']
                end_point = track['end']

                # Draw motion vector
                cv2.arrowedLine(result, start_point, end_point,
                                self.colors['tracking'], 2, tipLength=0.3)

        return result

    def add_info_overlay(self, frame, motion_count, feature_count, corner_count, track_count, side=""):
        """Add information overlay to the frame"""
        if frame is None:
            return None

        # Calculate FPS
        current_time = time.time()
        self.frame_count += 1

        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

        # Add overlay information
        info_y = 30
        line_height = 25

        overlay_info = [
            f"{side} Camera - FPS: {self.fps:.1f}",
            f"Motion Objects: {motion_count}",
            f"SIFT Features: {feature_count}",
            f"Corners: {corner_count}",
            f"Optical Flow: {track_count}",
            f"Motion Detection: {'ON' if self.enable_motion_detection else 'OFF'}",
            f"Feature Detection: {'ON' if self.enable_feature_detection else 'OFF'}",
            f"Background Dimming: {'ON' if self.enable_background_dimming else 'OFF'}"
        ]

        for i, info in enumerate(overlay_info):
            cv2.putText(frame, info, (10, info_y + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def add_controls_help(self, frame):
        """Add controls help text"""
        if frame is None:
            return None

        help_y = frame.shape[0] - 160
        help_texts = [
            "Controls:",
            "M - Toggle Motion Detection",
            "F - Toggle Feature Detection",
            "B - Toggle Background Dimming",
            "T - Adjust Motion Threshold (+/-)",
            "R - Reset Background Model",
            "Q/ESC - Quit"
        ]

        for i, text in enumerate(help_texts):
            cv2.putText(frame, text, (10, help_y + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def handle_keyboard_input(self, key):
        """Handle keyboard input for control"""
        if key == ord('m'):
            self.enable_motion_detection = not self.enable_motion_detection
            logger.info(f"Motion detection: {'ON' if self.enable_motion_detection else 'OFF'}")

        elif key == ord('f'):
            self.enable_feature_detection = not self.enable_feature_detection
            logger.info(f"Feature detection: {'ON' if self.enable_feature_detection else 'OFF'}")

        elif key == ord('b'):
            self.enable_background_dimming = not self.enable_background_dimming
            logger.info(f"Background dimming: {'ON' if self.enable_background_dimming else 'OFF'}")

        elif key == ord('+') or key == ord('='):
            self.motion_threshold += 500
            logger.info(f"Motion threshold increased to: {self.motion_threshold}")

        elif key == ord('-'):
            self.motion_threshold = max(500, self.motion_threshold - 500)
            logger.info(f"Motion threshold decreased to: {self.motion_threshold}")

        elif key == ord('r'):
            # Reset background models
            self.bg_subtractor_left = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, varThreshold=50, history=500
            )
            self.bg_subtractor_right = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, varThreshold=50, history=500
            )
            logger.info("Background models reset")

    def process_frame(self, frame, side):
        """Process a single camera frame"""
        if frame is None:
            return None

        # Select appropriate background subtractor and previous frame
        if side == "Left":
            bg_subtractor = self.bg_subtractor_left
            prev_frame = self.prev_left
        else:
            bg_subtractor = self.bg_subtractor_right
            prev_frame = self.prev_right

        # Motion detection
        motion_objects = []
        motion_mask = None
        if self.enable_motion_detection:
            motion_objects, motion_mask = self.detect_motion(frame, bg_subtractor, side)

        # Feature detection
        keypoints = []
        corners = None
        if self.enable_feature_detection:
            keypoints, corners = self.detect_features(frame)

        # Optical flow tracking
        optical_flow_tracks = []
        if prev_frame is not None:
            optical_flow_tracks = self.track_optical_flow(frame, prev_frame)

        # Apply background dimming
        processed_frame = self.apply_background_dimming(frame, motion_objects, motion_mask)

        # Draw all detections
        result_frame = self.draw_detections(
            processed_frame, motion_objects, keypoints, corners, optical_flow_tracks, side
        )

        # Add information overlay
        result_frame = self.add_info_overlay(
            result_frame,
            len(motion_objects),
            len(keypoints),
            len(corners) if corners is not None else 0,
            len(optical_flow_tracks),
            side
        )

        # Update previous frame
        if side == "Left":
            self.prev_left = frame.copy()
        else:
            self.prev_right = frame.copy()

        return result_frame

    async def receive_and_process_frames(self):
        """Receive frames and perform detection"""
        logger.info("Starting frame reception and processing")

        cv2.namedWindow('Left Camera - Motion & Features', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Right Camera - Motion & Features', cv2.WINDOW_AUTOSIZE)

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    if data.get('type') == 'camera_frame':
                        image_data = data.get('image')
                        if image_data:
                            # Decode the dual camera frame
                            full_frame = self.decode_image(image_data)
                            if full_frame is not None:
                                # Split into left and right frames
                                _, right_frame = self.split_dual_camera_feed(full_frame)

                                # Process each frame separately
                                # left_processed = self.process_frame(left_frame, "Left")
                                right_processed = self.process_frame(right_frame, "Right")

                                # # Add controls help to left frame
                                # if left_processed is not None:
                                #     left_processed = self.add_controls_help(left_processed)

                                # Display both processed frames
                                # if left_processed is not None:
                                #     cv2.imshow('Left Camera - Motion & Features', left_processed)
                                if right_processed is not None:
                                    cv2.imshow('Right Camera - Motion & Features', right_processed)

                                # Handle keyboard input
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q') or key == 27:  # 'q' or ESC
                                    logger.info("Exit requested by user")
                                    break
                                elif key != 255:
                                    self.handle_keyboard_input(key)

                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON data")
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection to server closed")
        except Exception as e:
            logger.error(f"Error in frame reception: {e}")
        finally:
            cv2.destroyAllWindows()

    async def run_detector(self):
        """Run the motion and feature detection client"""
        if not await self.connect_to_server():
            return

        self.is_running = True

        try:
            await self.receive_and_process_frames()
        except KeyboardInterrupt:
            logger.info("Detector interrupted by user")
        finally:
            self.is_running = False
            if self.websocket:
                await self.websocket.close()
            logger.info("Motion and Feature Detector stopped")

def main():
    # Configuration
    PI_SERVER_HOST = '192.168.88.252'
    PI_SERVER_PORT = 8765

    print("VR Motion and Feature Detection Client")
    print("=" * 50)
    print(f"Connecting to: {PI_SERVER_HOST}:{PI_SERVER_PORT}")
    print("\nFeatures:")
    print("- Motion detection with background subtraction")
    print("- SIFT feature point detection")
    print("- Corner detection (good features to track)")
    print("- Optical flow tracking")
    print("- Background dimming for moving objects")
    print("- Separate processing for dual camera feed")
    print("\nControls:")
    print("- M: Toggle Motion Detection")
    print("- F: Toggle Feature Detection")
    print("- B: Toggle Background Dimming")
    print("- +/-: Adjust Motion Threshold")
    print("- R: Reset Background Model")
    print("- Q/ESC: Quit")
    print("=" * 50)

    # Create and run detector
    detector = MotionFeatureDetector(server_host=PI_SERVER_HOST, server_port=PI_SERVER_PORT)

    try:
        asyncio.run(detector.run_detector())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")

if __name__ == "__main__":
    main()