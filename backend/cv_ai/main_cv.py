#!/usr/bin/env python3
"""
VR Client Test Application with Advanced Stereo Image Processing
Connects to the Raspberry Pi camera server and displays received images
Features:
- Stereo depth map calculation
- Feature point detection and matching
- Motion detection
- Foreground object detection
- Real-time visualization overlays
"""

import asyncio
import websockets
import json
import base64
import io
import cv2
import numpy as np
import time
import math
import logging
from PIL import Image
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StereoImageProcessor:
    """Advanced stereo image processing for VR applications"""

    def __init__(self):
        # Stereo matcher for depth calculation
        self.stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)

        # Feature detector (ORB - fast and robust)
        self.feature_detector = cv2.ORB_create(nfeatures=500)

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )

        # Motion detection parameters
        self.motion_history = deque(maxlen=5)
        self.motion_threshold = 1000  # Minimum contour area for motion

        # Processing modes
        self.processing_modes = {
            'depth': True,
            'features': True,
            'motion': True,
            'foreground': True
        }

        # Calibration parameters (should be calibrated for your specific cameras)
        self.stereo_params = {
            'baseline': 65.0,  # Distance between cameras in mm
            'focal_length': 500.0,  # Approximate focal length in pixels
            'cx': 320,  # Principal point x
            'cy': 240   # Principal point y
        }

        # Processing statistics
        self.stats = {
            'features_left': 0,
            'features_right': 0,
            'matches': 0,
            'motion_objects': 0,
            'depth_points': 0
        }

    def split_stereo_image(self, image):
        """Split vertically divided stereo image into left and right frames"""
        if image is None:
            return None, None

        height, width = image.shape[:2]
        mid_height = height // 2

        left_image = image[:mid_height, :]
        right_image = image[mid_height:, :]

        return left_image, right_image

    def calculate_depth_map(self, left_gray, right_gray):
        """Calculate depth map from stereo pair"""
        try:
            # Compute disparity map
            disparity = self.stereo.compute(left_gray, right_gray)

            # Normalize disparity for visualization
            disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # Apply colormap for better visualization
            depth_colored = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)

            # Calculate actual depth values (in mm)
            # depth = (baseline * focal_length) / disparity
            with np.errstate(divide='ignore', invalid='ignore'):
                depth_real = (self.stereo_params['baseline'] * self.stereo_params['focal_length']) / (disparity + 1e-6)
                depth_real = np.clip(depth_real, 0, 5000)  # Clip to 5 meters

            # Count valid depth points
            self.stats['depth_points'] = np.count_nonzero(disparity > 0)

            return depth_colored, depth_real, disparity

        except Exception as e:
            logger.error(f"Error calculating depth map: {e}")
            return None, None, None

    def detect_and_match_features(self, left_gray, right_gray):
        """Detect and match features between stereo pair"""
        try:
            # Detect keypoints and descriptors
            kp1, des1 = self.feature_detector.detectAndCompute(left_gray, None)
            kp2, des2 = self.feature_detector.detectAndCompute(right_gray, None)

            self.stats['features_left'] = len(kp1)
            self.stats['features_right'] = len(kp2)

            matches = []
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                # Match descriptors
                matches = self.matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                # Keep only good matches (distance threshold)
                good_matches = [m for m in matches if m.distance < 50]
                self.stats['matches'] = len(good_matches)

                return kp1, kp2, good_matches

            return kp1, kp2, []

        except Exception as e:
            logger.error(f"Error in feature detection: {e}")
            return [], [], []

    def detect_motion(self, frame):
        """Detect motion using background subtraction"""
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)

            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area
            motion_objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.motion_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_objects.append((x, y, w, h, area))

            self.stats['motion_objects'] = len(motion_objects)
            return motion_objects, fg_mask

        except Exception as e:
            logger.error(f"Error in motion detection: {e}")
            return [], None

    def detect_foreground_objects(self, left_frame, depth_map):
        """Detect foreground objects using depth information"""
        try:
            if depth_map is None:
                return []

            # Threshold for foreground (objects closer than 1.5 meters)
            foreground_threshold = 1500  # mm

            # Create foreground mask
            fg_mask = (depth_map > 0) & (depth_map < foreground_threshold)
            fg_mask = fg_mask.astype(np.uint8) * 255

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and analyze contours
            foreground_objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum object size
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate average depth of the object
                    object_region = depth_map[y:y+h, x:x+w]
                    valid_depths = object_region[object_region > 0]
                    avg_depth = np.mean(valid_depths) if len(valid_depths) > 0 else 0

                    foreground_objects.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'depth': avg_depth,
                        'contour': contour
                    })

            return foreground_objects

        except Exception as e:
            logger.error(f"Error in foreground detection: {e}")
            return []

    def create_visualization(self, left_frame, right_frame, depth_colored, motion_objects,
                             foreground_objects, keypoints_left, keypoints_right, matches):
        """Create compact visualization of all processing results"""
        try:
            height, width = left_frame.shape[:2]

            # Create a more compact layout - 2x2 grid instead of sparse layout
            canvas_width = width * 2
            canvas_height = height * 2
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # Position 1 (top-left): Original left frame with overlays
            frame_with_overlays = left_frame.copy()

            # Draw motion detection
            for x, y, w, h, area in motion_objects:
                cv2.rectangle(frame_with_overlays, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame_with_overlays, f"M:{int(area)}",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Draw foreground objects
            for obj in foreground_objects:
                x, y, w, h = obj['bbox']
                depth = obj['depth']
                cv2.rectangle(frame_with_overlays, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_with_overlays, f"FG:{int(depth)}mm",
                            (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw feature points (smaller circles)
            if self.processing_modes['features'] and len(keypoints_left) > 0:
                for kp in keypoints_left[:30]:  # Show fewer points for cleaner look
                    x, y = map(int, kp.pt)
                    cv2.circle(frame_with_overlays, (x, y), 2, (255, 0, 0), -1)

            canvas[0:height, 0:width] = frame_with_overlays

            # Position 2 (top-right): Right frame with feature points
            right_with_features = right_frame.copy()
            if self.processing_modes['features'] and len(keypoints_right) > 0:
                for kp in keypoints_right[:30]:
                    x, y = map(int, kp.pt)
                    cv2.circle(right_with_features, (x, y), 2, (255, 0, 0), -1)

            canvas[0:height, width:width*2] = right_with_features

            # Position 3 (bottom-left): Depth map
            if depth_colored is not None:
                canvas[height:height*2, 0:width] = depth_colored
            else:
                # If no depth map, show a placeholder or info panel
                info_panel = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(info_panel, "Depth Map", (width//2-50, height//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_panel, "Press 'D' to enable", (width//2-80, height//2+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                canvas[height:height*2, 0:width] = info_panel

            # Position 4 (bottom-right): Feature matches or statistics
            if len(matches) > 10 and len(keypoints_left) > 0 and len(keypoints_right) > 0:
                # Create a smaller match visualization
                match_region = np.zeros((height, width, 3), dtype=np.uint8)

                # Draw simplified matches
                for i, match in enumerate(matches[:15]):  # Show fewer matches
                    if i >= 15:
                        break

                    # Get keypoint coordinates
                    pt1 = keypoints_left[match.queryIdx].pt
                    pt2 = keypoints_right[match.trainIdx].pt

                    # Scale coordinates to fit in the region
                    x1, y1 = int(pt1[0] * 0.5), int(pt1[1])
                    x2, y2 = int(pt2[0] * 0.5 + width * 0.5), int(pt2[1])

                    # Draw keypoints and connection
                    cv2.circle(match_region, (x1, y1), 2, (0, 255, 0), -1)
                    cv2.circle(match_region, (x2, y2), 2, (0, 255, 0), -1)
                    cv2.line(match_region, (x1, y1), (x2, y2), (255, 255, 0), 1)

                # Add labels
                cv2.putText(match_region, "Feature Matches", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(match_region, f"Count: {len(matches)}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                canvas[height:height*2, width:width*2] = match_region
            else:
                # Create statistics panel
                stats_panel = np.zeros((height, width, 3), dtype=np.uint8)
                self.add_compact_stats_overlay(stats_panel)
                canvas[height:height*2, width:width*2] = stats_panel

            # Add minimal info overlay
            self.add_info_overlay(canvas)

            return canvas

        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return left_frame

    def add_compact_stats_overlay(self, panel):
        """Add compact processing statistics overlay"""
        try:
            height, width = panel.shape[:2]

            # Title
            cv2.putText(panel, "Processing Stats", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Statistics with more compact layout
            stats_lines = [
                f"Features L: {self.stats['features_left']}",
                f"Features R: {self.stats['features_right']}",
                f"Matches: {self.stats['matches']}",
                f"Motion obj: {self.stats['motion_objects']}",
                f"Depth pts: {self.stats['depth_points']}",
                "",
                "Processing modes:",
                f"Depth: {'ON' if self.processing_modes['depth'] else 'OFF'}",
                f"Features: {'ON' if self.processing_modes['features'] else 'OFF'}",
                f"Motion: {'ON' if self.processing_modes['motion'] else 'OFF'}",
                f"Foreground: {'ON' if self.processing_modes['foreground'] else 'OFF'}"
            ]

            for i, line in enumerate(stats_lines):
                y_pos = 55 + i * 25
                if y_pos > height - 20:
                    break

                if line == "":
                    continue
                elif line == "Processing modes:":
                    color = (255, 255, 0)
                    font_scale = 0.5
                elif "ON" in line:
                    color = (0, 255, 0)
                    font_scale = 0.4
                elif "OFF" in line:
                    color = (0, 0, 255)
                    font_scale = 0.4
                else:
                    color = (255, 255, 255)
                    font_scale = 0.4

                cv2.putText(panel, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

            # Add controls help at bottom
            help_y = height - 60
            cv2.putText(panel, "Controls:", (10, help_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
            cv2.putText(panel, "D/F/M/G - Toggle modes", (10, help_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            cv2.putText(panel, "C - Manual/Auto", (10, help_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            cv2.putText(panel, "T - Toggle size", (10, help_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        except Exception as e:
            logger.error(f"Error adding compact stats overlay: {e}")

    def add_info_overlay(self, canvas):
        """Add minimal info overlay for the main display"""
        try:
            # Much smaller info overlay in top-left corner
            height, width = canvas.shape[:2]

            # Semi-transparent background
            overlay = canvas.copy()
            cv2.rectangle(overlay, (5, 5), (250, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

            # Essential info only
            cv2.putText(canvas, f"Features: L{self.stats['features_left']} R{self.stats['features_right']} M{self.stats['matches']}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(canvas, f"Motion: {self.stats['motion_objects']} | Depth: {self.stats['depth_points']}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Processing status indicators
            status_text = ""
            if self.processing_modes['depth']: status_text += "D "
            if self.processing_modes['features']: status_text += "F "
            if self.processing_modes['motion']: status_text += "M "
            if self.processing_modes['foreground']: status_text += "G "

            cv2.putText(canvas, f"Active: {status_text}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        except Exception as e:
            logger.error(f"Error adding info overlay: {e}")

    def toggle_processing_mode(self, key):
        """Toggle processing modes based on key input"""
        if key == ord('d'):
            self.processing_modes['depth'] = not self.processing_modes['depth']
            logger.info(f"Depth processing: {'ON' if self.processing_modes['depth'] else 'OFF'}")
        elif key == ord('f'):
            self.processing_modes['features'] = not self.processing_modes['features']
            logger.info(f"Feature processing: {'ON' if self.processing_modes['features'] else 'OFF'}")
        elif key == ord('m'):
            self.processing_modes['motion'] = not self.processing_modes['motion']
            logger.info(f"Motion processing: {'ON' if self.processing_modes['motion'] else 'OFF'}")
        elif key == ord('g'):
            self.processing_modes['foreground'] = not self.processing_modes['foreground']
            logger.info(f"Foreground processing: {'ON' if self.processing_modes['foreground'] else 'OFF'}")

    def process_stereo_frame(self, stereo_image):
        """Main processing function for stereo frame"""
        try:
            # Split stereo image
            left_frame, right_frame = self.split_stereo_image(stereo_image)

            if left_frame is None or right_frame is None:
                return stereo_image

            # Convert to grayscale for processing
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            # Initialize results
            depth_colored = None
            depth_real = None
            motion_objects = []
            foreground_objects = []
            keypoints_left = []
            keypoints_right = []
            matches = []

            # Process depth map
            if self.processing_modes['depth']:
                depth_colored, depth_real, _ = self.calculate_depth_map(left_gray, right_gray)

            # Process feature detection and matching
            if self.processing_modes['features']:
                keypoints_left, keypoints_right, matches = self.detect_and_match_features(left_gray, right_gray)

            # Process motion detection
            if self.processing_modes['motion']:
                motion_objects, _ = self.detect_motion(left_frame)

            # Process foreground objects
            if self.processing_modes['foreground'] and depth_real is not None:
                foreground_objects = self.detect_foreground_objects(left_frame, depth_real)

            # Create comprehensive visualization
            result = self.create_visualization(left_frame, right_frame, depth_colored,
                                               motion_objects, foreground_objects,
                                               keypoints_left, keypoints_right, matches)

            return result

        except Exception as e:
            logger.error(f"Error in stereo processing: {e}")
            return stereo_image


class VRClient:
    def __init__(self, server_host='192.168.113.209', server_port=8765):
        self.server_host = server_host
        self.server_port = server_port
        self.websocket = None
        self.is_running = False
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Initialize stereo processor
        self.stereo_processor = StereoImageProcessor()

        # Manual control angles
        self.manual_pitch = 0.0
        self.manual_yaw = 0.0
        self.manual_roll = 0.0
        self.manual_control = True

        # Angle limits (from center position)
        self.max_pitch_up = 90.0    # Max up
        self.max_pitch_down = -75.0 # Max down
        self.max_yaw = 70.0         # ±70° (140° field of view)
        self.max_roll = 70.0        # ±70° roll

    async def connect_to_server(self):
        """Connect to the Raspberry Pi camera server"""
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
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)

            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Convert PIL to OpenCV format (BGR)
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            return opencv_image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

    def simulate_head_movement(self):
        """Simulate head movement angles for testing"""
        current_time = time.time()

        # Generate smooth sinusoidal movement
        pitch = math.sin(current_time * 0.5) * 30  # ±30 degrees
        yaw = math.cos(current_time * 0.3) * 45    # ±45 degrees
        roll = math.sin(current_time * 0.2) * 15   # ±15 degrees

        return pitch, yaw, roll

    def handle_keyboard_input(self, key):
        """Handle keyboard input for manual angle control and processing modes"""
        angle_step = 5.0

        # Check for processing mode toggles first
        if key in [ord('d'), ord('f'), ord('m'), ord('g')]:
            self.stereo_processor.toggle_processing_mode(key)
            return

        if key == ord('c'):  # Toggle manual control (changed from 'm' to avoid conflict)
            self.manual_control = not self.manual_control
            mode = "Manual" if self.manual_control else "Automatic"
            logger.info(f"Switched to {mode} control mode")

        elif self.manual_control:
            # Arrow keys for pitch and yaw
            if key == 82 or key == 0:  # Up arrow - pitch up (positive)
                self.manual_pitch = min(self.max_pitch_up, self.manual_pitch + angle_step)
                logger.info(f"Pitch: {self.manual_pitch:.1f}°")

            elif key == 84 or key == 1:  # Down arrow - pitch down (negative)
                self.manual_pitch = max(self.max_pitch_down, self.manual_pitch - angle_step)
                logger.info(f"Pitch: {self.manual_pitch:.1f}°")

            elif key == 81 or key == 2:  # Left arrow - yaw left (negative)
                self.manual_yaw = max(-self.max_yaw, self.manual_yaw - angle_step)
                logger.info(f"Yaw: {self.manual_yaw:.1f}°")

            elif key == 83 or key == 3:  # Right arrow - yaw right (positive)
                self.manual_yaw = min(self.max_yaw, self.manual_yaw + angle_step)
                logger.info(f"Yaw: {self.manual_yaw:.1f}°")

            # < and > for roll
            elif key == ord(',') or key == ord('<'):  # < key - roll left (negative)
                self.manual_roll = max(-self.max_roll, self.manual_roll - angle_step)
                logger.info(f"Roll: {self.manual_roll:.1f}°")

            elif key == ord('.') or key == ord('>'):  # > key - roll right (positive)
                self.manual_roll = min(self.max_roll, self.manual_roll + angle_step)
                logger.info(f"Roll: {self.manual_roll:.1f}°")

            # Reset angles
            elif key == ord('r'):
                self.manual_pitch = 0.0
                self.manual_yaw = 0.0
                self.manual_roll = 0.0
                logger.info("Angles reset to 0°")

    def get_current_angles(self):
        """Get current angles based on control mode"""
        if self.manual_control:
            return self.manual_pitch, self.manual_yaw, self.manual_roll
        else:
            return self.simulate_head_movement()

    async def send_head_angles(self):
        """Send head angles to the server"""
        while self.is_running and self.websocket:
            try:
                pitch, yaw, roll = self.get_current_angles()

                message = {
                    'type': 'head_angles',
                    'pitch': round(pitch, 2),
                    'yaw': round(yaw, 2),
                    'roll': round(roll, 2),
                    'timestamp': time.time()
                }

                await self.websocket.send(json.dumps(message))

                # Send angles every 100ms
                await asyncio.sleep(0.1)

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed while sending head angles")
                break
            except Exception as e:
                logger.error(f"Error sending head angles: {e}")
                break

    def display_frame_info(self, frame):
        """Add frame information overlay to the image"""
        if frame is None:
            return None

        # Calculate FPS
        current_time = time.time()
        self.frame_count += 1

        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.fps = fps
            self.frame_count = 0
            self.last_fps_time = current_time

        # Add overlay text in top-right corner
        if hasattr(self, 'fps'):
            text = f"FPS: {self.fps:.1f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, text, (frame.shape[1] - text_size[0] - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add connection status
        status_text = f"Connected: {self.server_host}:{self.server_port}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add control mode and angles
        mode_text = "Manual" if self.manual_control else "Auto"
        pitch, yaw, roll = self.get_current_angles()

        cv2.putText(frame, f"Mode: {mode_text} | P:{pitch:.1f}° Y:{yaw:.1f}° R:{roll:.1f}°",
                    (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Add controls help in bottom-right
        help_texts = [
            "Controls: C-Manual/Auto, R-Reset, Q-Quit, T-Toggle Size",
            "Processing: D-Depth, F-Features, M-Motion, G-Foreground"
        ]

        for i, help_text in enumerate(help_texts):
            text_size = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(frame, help_text,
                        (frame.shape[1] - text_size[0] - 10, frame.shape[0] - 60 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    async def receive_frames(self):
        """Receive and display camera frames from the server"""
        logger.info("Starting frame reception with stereo processing")

        # Create OpenCV window
        cv2.namedWindow('VR Stereo Camera Processing', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('VR Stereo Camera Processing', 1200, 800)

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    if data.get('type') == 'camera_frame':
                        # Decode the image
                        image_data = data.get('image')
                        if image_data:
                            frame = self.decode_image(image_data)
                            if frame is not None:
                                # Process stereo frame
                                processed_frame = self.stereo_processor.process_stereo_frame(frame)

                                # Add basic frame information
                                processed_frame = self.display_frame_info(processed_frame)

                                # Display the processed frame
                                cv2.imshow('VR Stereo Camera Processing', processed_frame)

                                # Check for keyboard input
                                key = cv2.waitKey(1) & 0xFF

                                # Handle special keys
                                if key == ord('q') or key == 27:  # 'q' or ESC
                                    logger.info("Exit requested by user")
                                    break
                                elif key == ord('t'):  # Toggle window size
                                    current_size = cv2.getWindowImageRect('VR Stereo Camera Processing')
                                    if current_size[2] > 1400:  # If large, make smaller
                                        cv2.resizeWindow('VR Stereo Camera Processing', 1280, 960)
                                        logger.info("Switched to normal size")
                                    else:  # If small, make larger
                                        cv2.resizeWindow('VR Stereo Camera Processing', 1600, 1200)
                                        logger.info("Switched to large size")
                                elif key != 255:  # Key was pressed
                                    self.handle_keyboard_input(key)

                        timing = data.get('server_timing', {})
                        if timing:
                            print(timing)

                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON data")
                except Exception as e:
                    logger.error(f"Error processing received frame: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection to server closed")
        except Exception as e:
            logger.error(f"Error in frame reception: {e}")
        finally:
            cv2.destroyAllWindows()

    async def run_client(self):
        """Run the VR client application"""
        if not await self.connect_to_server():
            return

        self.is_running = True

        try:
            # Start head angle transmission task
            head_angle_task = asyncio.create_task(self.send_head_angles())

            # Start frame reception (this will block until connection closes)
            await self.receive_frames()

        except KeyboardInterrupt:
            logger.info("Client interrupted by user")
        finally:
            self.is_running = False
            if hasattr(self, 'head_angle_task'):
                head_angle_task.cancel()

            if self.websocket:
                await self.websocket.close()

            logger.info("VR Client stopped")

def main():
    # Configuration - Update this with your Raspberry Pi's IP address
    PI_SERVER_HOST = '192.168.88.252'
    PI_SERVER_PORT = 8765

    print("VR Stereo Camera Client with Advanced Processing")
    print("=" * 50)
    print(f"Connecting to: {PI_SERVER_HOST}:{PI_SERVER_PORT}")
    print("\nCamera Controls:")
    print("- C: Toggle Manual/Automatic camera control")
    print("- Arrow Keys (Manual mode):")
    print("  ↑: Pitch up (max +90°)")
    print("  ↓: Pitch down (max -75°)")
    print("  ←→: Yaw left/right (±70°)")
    print("- < >: Roll left/right (±70°)")
    print("- R: Reset all angles to center (0°)")
    print("\nProcessing Controls:")
    print("- D: Toggle Depth Map calculation")
    print("- F: Toggle Feature detection & matching")
    print("- M: Toggle Motion detection")
    print("- G: Toggle Foreground object detection")
    print("\nGeneral:")
    print("- Q/ESC: Quit application")
    print("- T: Toggle window size (Normal/Large)")
    print("=" * 50)
    print("\nFeatures:")
    print("✓ Stereo depth map calculation")
    print("✓ Feature point detection and matching")
    print("✓ Real-time motion detection")
    print("✓ Foreground object detection using depth")
    print("✓ Comprehensive real-time visualization")
    print("=" * 50)

    # Create and run client
    client = VRClient(server_host=PI_SERVER_HOST, server_port=PI_SERVER_PORT)

    try:
        asyncio.run(client.run_client())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")

if __name__ == "__main__":
    main()