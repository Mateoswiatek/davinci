#!/usr/bin/env python3
"""
VR Client Test Application
Connects to the Raspberry Pi camera server and displays received images
Simulates head tilt angle transmission for testing with keyboard control
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
    def __init__(self, server_host='192.168.113.209', server_port=8765):
        self.server_host = server_host
        self.server_port = server_port
        self.websocket = None
        self.is_running = False
        self.frame_count = 0
        self.last_fps_time = time.time()

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
        """Handle keyboard input for manual angle control"""
        angle_step = 5.0

        if key == ord('m'):  # Toggle manual control
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

        # Add overlay text
        if hasattr(self, 'fps'):
            text = f"FPS: {self.fps:.1f} | Frame: {self.frame_count}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # Add connection status
        status_text = f"Connected to {self.server_host}:{self.server_port}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add control mode and angles
        mode_text = "Manual" if self.manual_control else "Auto"
        pitch, yaw, roll = self.get_current_angles()

        cv2.putText(frame, f"Mode: {mode_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"P:{pitch:.1f}° Y:{yaw:.1f}° R:{roll:.1f}°", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Add controls help
        help_y = frame.shape[0] - 140
        help_texts = [
            "Controls:",
            "M - Toggle Manual/Auto",
            "↑ - Pitch up (+90° max)",
            "↓ - Pitch down (-75° max)",
            "← → - Yaw ±70° (140° FOV)",
            "< > - Roll ±70°",
            "R - Reset to center (0°)",
            "Q/ESC - Quit"
        ]

        for i, help_text in enumerate(help_texts):
            cv2.putText(frame, help_text, (10, help_y + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    async def receive_frames(self):
        """Receive and display camera frames from the server"""
        logger.info("Starting frame reception")

        # Create OpenCV window
        cv2.namedWindow('VR Camera Feed', cv2.WINDOW_AUTOSIZE)

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    if data.get('type') == 'camera_frame':
                        # Decode and display the image
                        image_data = data.get('image')
                        if image_data:
                            frame = self.decode_image(image_data)
                            if frame is not None:
                                # Add information overlay
                                frame = self.display_frame_info(frame)

                                # Display the frame
                                cv2.imshow('VR Camera Feed', frame)

                                # Check for keyboard input
                                key = cv2.waitKey(1) & 0xFF

                                # Handle special keys
                                if key == ord('q') or key == 27:  # 'q' or ESC
                                    logger.info("Exit requested by user")
                                    break
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
            #TODO (28.05.2025): odbierać informacje o kątach
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

    print("VR Camera Client Test Application")
    print("=" * 40)
    print(f"Connecting to: {PI_SERVER_HOST}:{PI_SERVER_PORT}")
    print("\nControls:")
    print("- M: Toggle Manual/Automatic control")
    print("- Arrow Keys:")
    print("  ↑: Pitch up (max +90°)")
    print("  ↓: Pitch down (max -75°)")
    print("  ←→: Yaw left/right (±70°, total 140° FOV)")
    print("- < >: Roll left/right (±70°)")
    print("- R: Reset all angles to center (0°)")
    print("- Q/ESC: Quit application")
    print("=" * 40)

    # Create and run client
    client = VRClient(server_host=PI_SERVER_HOST, server_port=PI_SERVER_PORT)

    try:
        asyncio.run(client.run_client())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")

if __name__ == "__main__":
    main()