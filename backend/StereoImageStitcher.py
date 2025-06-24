#!/usr/bin/env python3
"""
VR Stereo Image Stitcher
Captures dual camera stream and stitches images into panorama using feature matching
"""

import asyncio
import json

import json

import websockets
import base64
import io
import cv2
import numpy as np
import time
import logging
from PIL import Image
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StereoImageStitcher:
    def __init__(self, server_host='192.168.88.252', server_port=8765):
        self.server_host = server_host
        self.server_port = server_port
        self.websocket = None
        self.is_running = False

        # Stitching components
        self.stitcher = None
        self.detector = None
        self.matcher = None
        self.homography_cache = None
        self.calibration_frames = 0
        self.is_calibrated = False

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0

        # Settings
        self.show_keypoints = False
        self.show_matches = False
        self.auto_recalibrate = True
        self.recalibrate_interval = 100  # frames

        self.initialize_stitcher()

    def initialize_stitcher(self):
        """Initialize image stitching components"""
        try:
            # Method 1: Use OpenCV's built-in Stitcher (recommended for panoramas)
            self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

            # Method 2: Manual stitching using feature detection
            # ORB is fast and works well for real-time applications
            self.detector = cv2.ORB_create(nfeatures=1000)

            # FLANN matcher for fast feature matching
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

            logger.info("Stitcher initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize stitcher: {e}")

    def split_stereo_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split stereo image into left and right camera views"""
        height, width = image.shape[:2]
        mid_point = width // 2

        left_image = image[:, :mid_point]
        right_image = image[:, mid_point:]

        return left_image, right_image

    def stitch_with_opencv_stitcher(self, left_img: np.ndarray, right_img: np.ndarray) -> Optional[np.ndarray]:
        """Stitch images using OpenCV's built-in Stitcher class"""
        try:
            images = [left_img, right_img]
            status, stitched = self.stitcher.stitch(images)

            if status == cv2.Stitcher_OK:
                return stitched
            else:
                logger.warning(f"Stitching failed with status: {status}")
                return None

        except Exception as e:
            logger.error(f"Error in OpenCV stitcher: {e}")
            return None

    def find_homography_manual(self, left_img: np.ndarray, right_img: np.ndarray) -> Optional[np.ndarray]:
        """Find homography between images using manual feature matching"""
        try:
            # Convert to grayscale for feature detection
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and descriptors
            kp1, desc1 = self.detector.detectAndCompute(gray_left, None)
            kp2, desc2 = self.detector.detectAndCompute(gray_right, None)

            if desc1 is None or desc2 is None or len(desc1) < 10 or len(desc2) < 10:
                logger.warning("Not enough features detected")
                return None

            # Match features
            matches = self.matcher.knnMatch(desc1, desc2, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 10:
                logger.warning(f"Not enough good matches found: {len(good_matches)}")
                return None

            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography with RANSAC
            homography, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                ransacReprojThreshold=5.0,
                confidence=0.99
            )

            # Store keypoints and matches for visualization
            if self.show_matches:
                self.last_kp1, self.last_kp2 = kp1, kp2
                self.last_good_matches = good_matches
                self.last_mask = mask

            return homography

        except Exception as e:
            logger.error(f"Error in manual homography calculation: {e}")
            return None

    def stitch_with_homography(self, left_img: np.ndarray, right_img: np.ndarray,
                               homography: np.ndarray) -> Optional[np.ndarray]:
        """Stitch images using computed homography"""
        try:
            h1, w1 = left_img.shape[:2]
            h2, w2 = right_img.shape[:2]

            # Get corners of right image
            corners_right = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

            # Transform corners to left image coordinate system
            transformed_corners = cv2.perspectiveTransform(corners_right, homography)

            # Combine corners to find output canvas size
            all_corners = np.concatenate([
                np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
                transformed_corners
            ], axis=0)

            # Find bounding rectangle
            x_coords = all_corners[:, 0, 0]
            y_coords = all_corners[:, 0, 1]

            x_min, x_max = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
            y_min, y_max = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))

            # Create translation matrix to handle negative coordinates
            translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

            # Output canvas size
            output_width = x_max - x_min
            output_height = y_max - y_min

            # Warp right image
            warped_right = cv2.warpPerspective(
                right_img,
                translation @ homography,
                (output_width, output_height)
            )

            # Place left image on canvas
            result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            left_x_offset = -x_min
            left_y_offset = -y_min

            # Copy left image to result
            result[left_y_offset:left_y_offset + h1,
            left_x_offset:left_x_offset + w1] = left_img

            # Blend images (simple overlay where warped_right is non-zero)
            mask = np.any(warped_right > 0, axis=2)
            result[mask] = warped_right[mask]

            return result

        except Exception as e:
            logger.error(f"Error in homography stitching: {e}")
            return None

    def create_side_by_side_comparison(self, left_img: np.ndarray, right_img: np.ndarray,
                                       stitched: np.ndarray) -> np.ndarray:
        """Create a comparison view showing original images and stitched result"""
        try:
            # Resize images for display
            display_height = 300

            # Resize left and right images
            left_ratio = display_height / left_img.shape[0]
            right_ratio = display_height / right_img.shape[0]

            left_resized = cv2.resize(left_img,
                                      (int(left_img.shape[1] * left_ratio), display_height))
            right_resized = cv2.resize(right_img,
                                       (int(right_img.shape[1] * right_ratio), display_height))

            # Resize stitched image
            if stitched is not None:
                stitched_ratio = display_height / stitched.shape[0]
                stitched_resized = cv2.resize(stitched,
                                              (int(stitched.shape[1] * stitched_ratio), display_height))
            else:
                stitched_resized = np.zeros((display_height, 400, 3), dtype=np.uint8)
                cv2.putText(stitched_resized, "Stitching Failed", (50, display_height//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Combine images horizontally
            max_width = max(left_resized.shape[1] + right_resized.shape[1],
                            stitched_resized.shape[1])

            combined = np.zeros((display_height * 2 + 20, max_width, 3), dtype=np.uint8)

            # Place original images
            combined[:display_height, :left_resized.shape[1]] = left_resized
            combined[:display_height, left_resized.shape[1]:left_resized.shape[1] + right_resized.shape[1]] = right_resized

            # Place stitched image
            combined[display_height + 20:display_height * 2 + 20, :stitched_resized.shape[1]] = stitched_resized

            # Add labels
            cv2.putText(combined, "Left Camera", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Right Camera", (left_resized.shape[1] + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Stitched Panorama", (10, display_height + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return combined

        except Exception as e:
            logger.error(f"Error creating comparison view: {e}")
            return left_img

    def process_stereo_frame(self, stereo_image: np.ndarray) -> Optional[np.ndarray]:
        """Process stereo frame and return stitched panorama"""
        try:
            # Split stereo image
            left_img, right_img = self.split_stereo_image(stereo_image)

            # Method 1: Try OpenCV's built-in stitcher first
            # stitched = self.stitch_with_opencv_stitcher(left_img, right_img)

            # Method 2: If built-in stitcher fails, use manual method
            # if stitched is None:
            if True:
                # Use cached homography if available and not time to recalibrate
                if (self.homography_cache is not None and
                        self.is_calibrated and
                        not (self.auto_recalibrate and self.frame_count % self.recalibrate_interval == 0)):

                    stitched = self.stitch_with_homography(left_img, right_img, self.homography_cache)
                else:
                    # Calculate new homography
                    homography = self.find_homography_manual(left_img, right_img)
                    if homography is not None:
                        self.homography_cache = homography
                        self.is_calibrated = True
                        stitched = self.stitch_with_homography(left_img, right_img, homography)

            # Create comparison view for debugging
            if self.show_keypoints or self.show_matches:
                return self.create_side_by_side_comparison(left_img, right_img, stitched)

            return stitched

        except Exception as e:
            logger.error(f"Error processing stereo frame: {e}")
            return None

    def decode_image(self, base64_data: str) -> Optional[np.ndarray]:
        """Decode base64 image data to OpenCV format"""
        try:
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes))
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

    def add_overlay_info(self, frame: np.ndarray) -> np.ndarray:
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

        # Add overlay text
        overlay_texts = [
            f"FPS: {self.fps:.1f}",
            f"Calibrated: {'Yes' if self.is_calibrated else 'No'}",
            f"Homography: {'Cached' if self.homography_cache is not None else 'None'}",
            "",
            "Controls:",
            "K - Toggle keypoint display",
            "M - Toggle match display",
            "R - Reset calibration",
            "C - Force recalibration",
            "Q/ESC - Quit"
        ]

        y_offset = 30
        for i, text in enumerate(overlay_texts):
            if text:  # Skip empty strings
                cv2.putText(frame, text, (10, y_offset + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def handle_keyboard_input(self, key: int):
        """Handle keyboard input for control"""
        if key == ord('k'):
            self.show_keypoints = not self.show_keypoints
            logger.info(f"Keypoint display: {'ON' if self.show_keypoints else 'OFF'}")

        elif key == ord('m'):
            self.show_matches = not self.show_matches
            logger.info(f"Match display: {'ON' if self.show_matches else 'OFF'}")

        elif key == ord('r'):
            self.homography_cache = None
            self.is_calibrated = False
            logger.info("Calibration reset")

        elif key == ord('c'):
            self.homography_cache = None
            self.is_calibrated = False
            logger.info("Forced recalibration")

    async def connect_to_server(self) -> bool:
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

    async def run_stitcher(self):
        """Main loop for the stitcher"""
        if not await self.connect_to_server():
            return

        self.is_running = True
        cv2.namedWindow('Stereo Stitcher', cv2.WINDOW_AUTOSIZE)

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    if data.get('type') == 'camera_frame':
                        image_data = data.get('image')
                        if image_data:
                            # Decode stereo image
                            stereo_frame = self.decode_image(image_data)
                            if stereo_frame is not None:
                                # Process and stitch
                                result = self.process_stereo_frame(stereo_frame)

                                if result is not None:
                                    # Add overlay information
                                    result = self.add_overlay_info(result)

                                    # Display result
                                    cv2.imshow('Stereo Stitcher', result)

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
            logger.error(f"Error in main loop: {e}")
        finally:
            self.is_running = False
            cv2.destroyAllWindows()
            if self.websocket:
                await self.websocket.close()

def main():
    """Main function"""
    print("VR Stereo Image Stitcher")
    print("=" * 40)
    print("This program captures dual camera stream and stitches into panorama")
    print("\nFeatures:")
    print("- Automatic image stitching using OpenCV")
    print("- Fallback manual stitching with ORB features")
    print("- Real-time homography caching for performance")
    print("- Debug visualization modes")
    print("\nControls:")
    print("- K: Toggle keypoint visualization")
    print("- M: Toggle feature match visualization")
    print("- R: Reset calibration")
    print("- C: Force recalibration")
    print("- Q/ESC: Quit")
    print("=" * 40)

    # Configuration
    SERVER_HOST = '192.168.88.252'
    SERVER_PORT = 8765

    stitcher = StereoImageStitcher(server_host=SERVER_HOST, server_port=SERVER_PORT)

    try:
        asyncio.run(stitcher.run_stitcher())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")

if __name__ == "__main__":
    main()