#!/usr/bin/env python3
"""
VR Client Test Application
Connects to the Raspberry Pi camera server and displays received images
Supports both stereo and mono modes
Simulates head tilt angle transmission for testing
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VRClient:
    def __init__(self, server_host='192.168.113.209', server_port=8765, stereo_mode=True):
        self.server_host = server_host
        self.server_port = server_port
        self.stereo_mode = stereo_mode  # Switch for stereo/mono display
        self.websocket = None
        self.is_running = False
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.server_stereo_mode = None  # Track server's current mode

    async def connect_to_server(self):
        """Connect to the Raspberry Pi camera server"""
        uri = f"ws://{self.server_host}:{self.server_port}"
        logger.info(f"Connecting to camera server at {uri}")

        try:
            self.websocket = await websockets.connect(uri)
            logger.info("Connected to camera server")

            # Send initial stereo mode preference to server
            await self.send_stereo_mode_request(self.stereo_mode)

            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False

    async def send_stereo_mode_request(self, stereo_mode):
        """Send stereo mode change request to server"""
        try:
            message = {
                'type': 'set_stereo_mode',
                'stereo_mode': stereo_mode,
                'timestamp': time.time()
            }
            await self.websocket.send(json.dumps(message))
            mode_text = "STEREO" if stereo_mode else "MONO"
            logger.info(f"Requested {mode_text} mode from server")
        except Exception as e:
            logger.error(f"Error sending stereo mode request: {e}")

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

    def combine_stereo_images(self, left_image, right_image, display_mode='side_by_side'):
        """Combine left and right images for stereo display"""
        if left_image is None or right_image is None:
            return None

        try:
            if display_mode == 'side_by_side':
                # Side by side display
                combined = np.hstack((left_image, right_image))
            elif display_mode == 'anaglyph':
                # Red/Cyan anaglyph (experimental)
                # Left eye = red channel, Right eye = cyan channels
                height, width = left_image.shape[:2]
                combined = np.zeros((height, width, 3), dtype=np.uint8)

                # Red channel from left image
                combined[:,:,2] = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
                # Green and Blue channels from right image
                right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
                combined[:,:,0] = right_gray  # Blue
                combined[:,:,1] = right_gray  # Green
            else:
                # Default to side by side
                combined = np.hstack((left_image, right_image))

            return combined
        except Exception as e:
            logger.error(f"Error combining stereo images: {e}")
            return None

    def simulate_head_movement(self):
        """Simulate head movement angles for testing"""
        current_time = time.time()

        # Generate smooth sinusoidal movement
        pitch = math.sin(current_time * 0.5) * 30  # ±30 degrees
        yaw = math.cos(current_time * 0.3) * 45    # ±45 degrees
        roll = math.sin(current_time * 0.2) * 15   # ±15 degrees

        return pitch, yaw, roll

    async def send_head_angles(self):
        """Send simulated head angles to the server"""
        while self.is_running and self.websocket:
            try:
                pitch, yaw, roll = self.simulate_head_movement()

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

    def display_frame_info(self, frame, is_stereo=False):
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

        # Add overlay text
        mode_text = "STEREO" if is_stereo else "MONO"
        server_mode_text = "STEREO" if self.server_stereo_mode else "MONO" if self.server_stereo_mode is not None else "UNKNOWN"

        text = f"FPS: {self.fps:.1f} | Frame: {self.frame_count} | Mode: {mode_text} | Server: {server_mode_text}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        # Add connection status
        status_text = f"Connected to {self.server_host}:{self.server_port}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add controls info
        controls_text = "Controls: Q/ESC=Quit, S=Toggle Stereo, A=Anaglyph, Space=Side-by-side"
        cv2.putText(frame, controls_text, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return frame

    async def receive_frames(self):
        """Receive and display camera frames from the server"""
        logger.info("Starting frame reception")

        # Create OpenCV window
        window_name = 'VR Camera Feed'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        # Display mode for stereo images
        stereo_display_mode = 'side_by_side'  # 'side_by_side' or 'anaglyph'

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')

                    # Handle stereo mode confirmation
                    if message_type == 'stereo_mode_changed':
                        self.server_stereo_mode = data.get('stereo_mode')
                        mode_text = "STEREO" if self.server_stereo_mode else "MONO"
                        logger.info(f"Server confirmed {mode_text} mode")
                        continue

                    # SWITCH: Handle different frame types based on server mode
                    if message_type == 'camera_frame_stereo':
                        # Handle stereo frame
                        left_image_data = data.get('left_image')
                        right_image_data = data.get('right_image')
                        self.server_stereo_mode = True

                        if left_image_data and right_image_data:
                            left_frame = self.decode_image(left_image_data)
                            right_frame = self.decode_image(right_image_data)

                            if left_frame is not None and right_frame is not None:
                                # Combine stereo images
                                combined_frame = self.combine_stereo_images(
                                    left_frame, right_frame, stereo_display_mode
                                )

                                if combined_frame is not None:
                                    # Add information overlay
                                    combined_frame = self.display_frame_info(combined_frame, is_stereo=True)

                                    # Display the combined frame
                                    cv2.imshow(window_name, combined_frame)

                    elif message_type == 'camera_frame_mono':
                        # Handle mono frame
                        image_data = data.get('image')
                        self.server_stereo_mode = False

                        if image_data:
                            frame = self.decode_image(image_data)
                            if frame is not None:
                                # Add information overlay
                                frame = self.display_frame_info(frame, is_stereo=False)

                                # Display the frame
                                cv2.imshow(window_name, frame)

                    elif message_type == 'camera_frame':
                        # Handle legacy mono frame format
                        image_data = data.get('image')
                        self.server_stereo_mode = False

                        if image_data:
                            frame = self.decode_image(image_data)
                            if frame is not None:
                                # Add information overlay
                                frame = self.display_frame_info(frame, is_stereo=False)

                                # Display the frame
                                cv2.imshow(window_name, frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        logger.info("Exit requested by user")
                        break
                    elif key == ord('s'):  # Toggle stereo mode
                        self.stereo_mode = not self.stereo_mode
                        await self.send_stereo_mode_request(self.stereo_mode)
                        mode_text = "STEREO" if self.stereo_mode else "MONO"
                        logger.info(f"Toggled to {mode_text} mode")
                    elif key == ord('a'):  # Toggle anaglyph mode
                        stereo_display_mode = 'anaglyph' if stereo_display_mode == 'side_by_side' else 'side_by_side'
                        logger.info(f"Stereo display mode: {stereo_display_mode}")
                    elif key == ord(' '):  # Space - force side by side
                        stereo_display_mode = 'side_by_side'
                        logger.info(f"Stereo display mode: {stereo_display_mode}")

                    # Log timing info if available
                    timing = data.get('server_timing', {})
                    if timing and self.frame_count % 60 == 0:  # Log every 60 frames
                        logger.debug(f"Server timing: {timing}")

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
            # TODO (28.05.2025): odbierać informacje o kątach
            head_angle_task = asyncio.create_task(self.send_head_angles())

            # Start frame reception (this will block until connection closes)
            await self.receive_frames()

        except KeyboardInterrupt:
            logger.info("Client interrupted by user")
        finally:
            self.is_running = False
            head_angle_task.cancel()

            if self.websocket:
                await self.websocket.close()

            logger.info("VR Client stopped")

def main():
    # Configuration - Update this with your Raspberry Pi's IP address
    PI_SERVER_HOST = '192.168.1.26'  # Replace with your Pi's IP
    PI_SERVER_PORT = 8765

    # CLIENT STEREO MODE SWITCH - Set initial preference
    INITIAL_STEREO_MODE = True  # True for stereo, False for mono

    print("VR Camera Client Test Application")
    print("=" * 50)
    print(f"Connecting to: {PI_SERVER_HOST}:{PI_SERVER_PORT}")
    print(f"Initial mode: {'STEREO' if INITIAL_STEREO_MODE else 'MONO'}")
    print()
    print("Controls:")
    print("  Q or ESC    - Quit application")
    print("  S           - Toggle stereo/mono mode")
    print("  A           - Toggle anaglyph/side-by-side display")
    print("  SPACE       - Force side-by-side display")
    print("=" * 50)

    # Create and run client
    client = VRClient(
        server_host=PI_SERVER_HOST,
        server_port=PI_SERVER_PORT,
        stereo_mode=INITIAL_STEREO_MODE
    )

    try:
        asyncio.run(client.run_client())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")

if __name__ == "__main__":
    main()