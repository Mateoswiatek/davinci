#!/usr/bin/env python3
"""
Raspberry Pi Camera Server for VR Project
Captures camera images and streams them via WebSocket to VR glasses
Also receives head tilt angles for servo control
"""

import asyncio
import websockets
import json
import base64
import io
import time
from picamera2 import Picamera2
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraServer:
    def __init__(self, host='0.0.0.0', port=8765, image_quality=85, image_size=(640, 480)):
        self.host = host
        self.port = port
        self.image_quality = image_quality
        self.image_size = image_size
        self.picam2 = None
        self.connected_clients = set()
        self.is_running = False

    def initialize_camera(self):
        """Initialize the Pi Camera"""
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_still_configuration(
                main={"size": self.image_size, "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            logger.info(f"Camera initialized with resolution {self.image_size}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False

    def capture_frame(self):
        """Capture a frame from the camera and return as base64 encoded JPEG"""
        try:
            # Capture frame as numpy array
            frame = self.picam2.capture_array()

            # Convert to PIL Image
            image = Image.fromarray(frame)

            # Compress to JPEG
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.image_quality)
            buffer.seek(0)

            # Encode to base64
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return image_data
        except Exception as e:


            logger.error(f"Error capturing frame: {e}")
            return None

    async def handle_client(self, websocket):
        """Handle individual client connections"""
        client_address = websocket.remote_address
        logger.info(f"Client connected: {client_address}")

        self.connected_clients.add(websocket)

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    if data.get('type') == 'head_angles':
                        # Handle head tilt angles from VR glasses
                        pitch = data.get('pitch', 0)
                        yaw = data.get('yaw', 0)
                        roll = data.get('roll', 0)

                        logger.info(f"Received head angles - Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")

                        #TODO (28.05.2025): Integracja z Servo
                        # Here you would integrate with your servo control code
                        # self.control_servos(pitch, yaw, roll)

                    elif data.get('type') == 'request_frame':
                        # Client requesting a frame
                        await self.send_frame(websocket)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from {client_address}")
                except Exception as e:
                    logger.error(f"Error processing message from {client_address}: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_address}")
        except Exception as e:
            logger.error(f"Error with client {client_address}: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def send_frame(self, websocket):
        """Send a single frame to the specified client"""
        frame_data = self.capture_frame()
        if frame_data:
            message = {
                'type': 'camera_frame',
                'timestamp': time.time(),
                'image': frame_data
            }
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                pass

    async def broadcast_frames(self):
        """Continuously broadcast frames to all connected clients"""
        while self.is_running:
            if self.connected_clients:
                frame_data = self.capture_frame()
                if frame_data:
                    message = {
                        'type': 'camera_frame',
                        'timestamp': time.time(),
                        'image': frame_data
                    }

                    # Send to all connected clients
                    disconnected_clients = set()
                    for client in self.connected_clients.copy():
                        try:
                            await client.send(json.dumps(message))
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                        except Exception as e:
                            logger.error(f"Error sending to client: {e}")
                            disconnected_clients.add(client)

                    # Remove disconnected clients
                    self.connected_clients -= disconnected_clients

            # Control frame rate (adjust as needed)
            await asyncio.sleep(1/30)  # 30 FPS

    async def start_server(self):
        """Start the WebSocket server"""
        if not self.initialize_camera():
            logger.error("Failed to initialize camera. Exiting.")
            return

        self.is_running = True
        logger.info(f"Starting camera server on {self.host}:{self.port}")

        # Start the WebSocket server
        server = await websockets.serve(self.handle_client, self.host, self.port)

        # Start broadcasting frames
        broadcast_task = asyncio.create_task(self.broadcast_frames())

        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            self.is_running = False
            broadcast_task.cancel()
            if self.picam2:
                self.picam2.stop()
            logger.info("Server stopped")


def main():
    # Configuration
    SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
    SERVER_PORT = 8765
    IMAGE_QUALITY = 85  # JPEG quality (1-100)
    IMAGE_SIZE = (640, 480)  # Adjust based on VR glasses capability

    # Create and start server
    server = CameraServer(
        host=SERVER_HOST,
        port=SERVER_PORT,
        image_quality=IMAGE_QUALITY,
        image_size=IMAGE_SIZE
    )

    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")

if __name__ == "__main__":
    main()