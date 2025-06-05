#!/usr/bin/env python3
"""
Raspberry Pi Camera Server for VR Project
Captures camera images and streams them via WebSocket to VR glasses
Also receives head tilt angles for servo control
Supports stereo mode - splits image vertically for left/right eye
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
    def __init__(self, host='0.0.0.0', port=8765, image_quality=85, image_size=(640, 480), stereo_mode=True):
        self.host = host
        self.port = port
        self.image_quality = image_quality
        self.image_size = image_size
        self.stereo_mode = stereo_mode  # Switch for stereo/mono mode
        self.picam2 = None
        self.connected_clients = set()
        self.is_running = False
        self.frame_counter = 0
        self.latency_stats = {
            'capture_times': [],
            'split_times': [],  # New timing for image splitting
            'encode_times': [],
            'send_times': [],
            'total_times': []
        }

    def initialize_camera(self):
        """Initialize the Pi Camera"""
        try:
            init_start = time.time()
            self.picam2 = Picamera2()
            config = self.picam2.create_still_configuration(
                main={"size": self.image_size, "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            init_time = (time.time() - init_start) * 1000
            mode_text = "STEREO" if self.stereo_mode else "MONO"
            logger.info(f"Camera initialized with resolution {self.image_size} in {init_time:.2f}ms [{mode_text} MODE]")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False

    def split_image_vertically(self, image):
        """Split image vertically into left and right parts"""
        split_start = time.time()

        width, height = image.size
        mid_x = width // 2

        # Left half (0 to mid_x)
        left_image = image.crop((0, 0, mid_x, height))

        # Right half (mid_x to width)
        right_image = image.crop((mid_x, 0, width, height))

        split_time = (time.time() - split_start) * 1000

        return left_image, right_image, split_time

    def encode_image_to_base64(self, image, timing_key=None):
        """Encode single image to base64 JPEG"""
        encode_start = time.time()

        # JPEG compression
        compress_start = time.time()
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=self.image_quality)
        buffer.seek(0)
        compress_end = time.time()

        # Base64 encoding
        b64_start = time.time()
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        b64_end = time.time()

        encode_time = (time.time() - encode_start) * 1000

        timing = {
            'compression_ms': (compress_end - compress_start) * 1000,
            'base64_ms': (b64_end - b64_start) * 1000,
            'total_encode_ms': encode_time
        }

        return image_data, timing

    def capture_frame(self):
        """Capture a frame from the camera and return as base64 encoded JPEG(s) with timing info"""
        frame_start_time = time.time()
        timing = {}

        try:
            # Timestamp: Start capture
            capture_start = time.time()
            frame = self.picam2.capture_array()
            capture_end = time.time()

            # Timestamp: Start PIL conversion
            pil_start = time.time()
            image = Image.fromarray(frame)
            pil_end = time.time()

            timing['capture_ms'] = (capture_end - capture_start) * 1000
            timing['pil_conversion_ms'] = (pil_end - pil_start) * 1000

            # SWITCH: Early split (right after acquisition)
            if self.stereo_mode:
                # Split image into left and right parts
                left_image, right_image, split_time = self.split_image_vertically(image)
                timing['split_ms'] = split_time

                # Encode both images
                left_data, left_timing = self.encode_image_to_base64(left_image)
                right_data, right_timing = self.encode_image_to_base64(right_image)

                # Combine timing info
                timing['left_compression_ms'] = left_timing['compression_ms']
                timing['left_base64_ms'] = left_timing['base64_ms']
                timing['right_compression_ms'] = right_timing['compression_ms']
                timing['right_base64_ms'] = right_timing['base64_ms']
                timing['total_encode_ms'] = left_timing['total_encode_ms'] + right_timing['total_encode_ms']

                image_data = {
                    'left': left_data,
                    'right': right_data
                }

            else:
                # Mono mode - encode single image
                single_data, single_timing = self.encode_image_to_base64(image)
                timing.update(single_timing)
                timing['split_ms'] = 0  # No splitting in mono mode

                image_data = {
                    'mono': single_data
                }

            # Total processing time
            total_time = (time.time() - frame_start_time) * 1000
            timing['total_processing_ms'] = total_time

            # Update stats
            self.update_latency_stats(timing)

            # Log detailed timing every 30 frames
            self.frame_counter += 1
            if self.frame_counter % 30 == 0:
                self.log_latency_stats()

            return image_data, timing

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None, {}

    def update_latency_stats(self, timing):
        """Update latency statistics"""
        self.latency_stats['capture_times'].append(timing.get('capture_ms', 0))
        self.latency_stats['split_times'].append(timing.get('split_ms', 0))
        self.latency_stats['encode_times'].append(timing.get('total_encode_ms', 0))
        self.latency_stats['total_times'].append(timing.get('total_processing_ms', 0))

        # Keep only last 100 measurements
        for key in self.latency_stats:
            if len(self.latency_stats[key]) > 100:
                self.latency_stats[key] = self.latency_stats[key][-100:]

    def log_latency_stats(self):
        """Log average latency statistics"""
        if not self.latency_stats['total_times']:
            return

        avg_capture = sum(self.latency_stats['capture_times']) / len(self.latency_stats['capture_times'])
        avg_split = sum(self.latency_stats['split_times']) / len(self.latency_stats['split_times'])
        avg_encode = sum(self.latency_stats['encode_times']) / len(self.latency_stats['encode_times'])
        avg_total = sum(self.latency_stats['total_times']) / len(self.latency_stats['total_times'])

        mode_text = "STEREO" if self.stereo_mode else "MONO"
        logger.info(f"Latency Stats [{mode_text}] (Frame #{self.frame_counter}) - "
                    f"Capture: {avg_capture:.2f}ms, "
                    f"Split: {avg_split:.2f}ms, "
                    f"Encode: {avg_encode:.2f}ms, "
                    f"Total: {avg_total:.2f}ms")

    async def handle_client(self, websocket):
        """Handle individual client connections"""
        client_address = websocket.remote_address
        connect_time = time.time()
        logger.info(f"Client connected: {client_address} at {connect_time}")

        self.connected_clients.add(websocket)

        try:
            async for message in websocket:
                message_received_time = time.time()
                try:
                    parse_start = time.time()
                    data = json.loads(message)
                    parse_time = (time.time() - parse_start) * 1000

                    if data.get('type') == 'head_angles':
                        # Handle head tilt angles from VR glasses
                        pitch = data.get('pitch', 0)
                        yaw = data.get('yaw', 0)
                        roll = data.get('roll', 0)

                        # Check if client sent timestamp
                        client_timestamp = data.get('timestamp')
                        if client_timestamp:
                            network_latency = (message_received_time - client_timestamp) * 1000
                            logger.info(f"Head angles - Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f} "
                                        f"(Network latency: {network_latency:.2f}ms, Parse: {parse_time:.2f}ms)")
                        else:
                            logger.info(f"Head angles - Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")

                        #TODO (28.05.2025): Integracja z Servo
                        # Here you would integrate with your servo control code
                        # self.control_servos(pitch, yaw, roll)

                    elif data.get('type') == 'request_frame':
                        # Client requesting a frame
                        request_start = time.time()
                        await self.send_frame(websocket)
                        request_time = (time.time() - request_start) * 1000
                        logger.info(f"Frame request processed in {request_time:.2f}ms")

                    elif data.get('type') == 'set_stereo_mode':
                        # Client requesting stereo mode change
                        new_mode = data.get('stereo_mode', self.stereo_mode)
                        if new_mode != self.stereo_mode:
                            self.stereo_mode = new_mode
                            mode_text = "STEREO" if self.stereo_mode else "MONO"
                            logger.info(f"Switched to {mode_text} mode")

                            # Send confirmation to client
                            response = {
                                'type': 'stereo_mode_changed',
                                'stereo_mode': self.stereo_mode,
                                'timestamp': time.time()
                            }
                            await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from {client_address}")
                except Exception as e:
                    logger.error(f"Error processing message from {client_address}: {e}")

        except websockets.exceptions.ConnectionClosed:
            disconnect_time = time.time()
            session_duration = disconnect_time - connect_time
            logger.info(f"Client disconnected: {client_address} (session: {session_duration:.2f}s)")
        except Exception as e:
            logger.error(f"Error with client {client_address}: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def send_frame(self, websocket):
        """Send a single frame to the specified client"""
        send_start_time = time.time()

        frame_result = self.capture_frame()
        if frame_result[0]:  # frame_data exists
            frame_data, timing = frame_result

            message_build_start = time.time()

            # SWITCH: Late decision (right before sending)
            if self.stereo_mode:
                message = {
                    'type': 'camera_frame_stereo',
                    'timestamp': time.time(),
                    'frame_id': self.frame_counter,
                    'left_image': frame_data['left'],
                    'right_image': frame_data['right'],
                    'stereo_mode': True,
                    'server_timing': timing
                }
            else:
                message = {
                    'type': 'camera_frame_mono',
                    'timestamp': time.time(),
                    'frame_id': self.frame_counter,
                    'image': frame_data['mono'],
                    'stereo_mode': False,
                    'server_timing': timing
                }

            message_build_time = (time.time() - message_build_start) * 1000

            try:
                websocket_send_start = time.time()
                await websocket.send(json.dumps(message))
                websocket_send_time = (time.time() - websocket_send_start) * 1000

                total_send_time = (time.time() - send_start_time) * 1000

                mode_text = "STEREO" if self.stereo_mode else "MONO"
                logger.debug(f"Frame #{self.frame_counter} sent [{mode_text}] - "
                             f"Build: {message_build_time:.2f}ms, "
                             f"WebSocket: {websocket_send_time:.2f}ms, "
                             f"Total: {total_send_time:.2f}ms")

            except websockets.exceptions.ConnectionClosed:
                pass

    async def broadcast_frames(self):
        """Continuously broadcast frames to all connected clients"""
        loop_counter = 0
        while self.is_running:
            loop_start = time.time()

            if self.connected_clients:
                frame_result = self.capture_frame()
                if frame_result[0]:  # frame_data exists
                    frame_data, timing = frame_result

                    message_start = time.time()

                    # SWITCH: Late decision (right before broadcasting)
                    if self.stereo_mode:
                        message = {
                            'type': 'camera_frame_stereo',
                            'timestamp': time.time(),
                            'frame_id': self.frame_counter,
                            'left_image': frame_data['left'],
                            'right_image': frame_data['right'],
                            'stereo_mode': True,
                            'server_timing': timing
                        }
                    else:
                        message = {
                            'type': 'camera_frame_mono',
                            'timestamp': time.time(),
                            'frame_id': self.frame_counter,
                            'image': frame_data['mono'],
                            'stereo_mode': False,
                            'server_timing': timing
                        }

                    message_build_time = (time.time() - message_start) * 1000

                    # Send to all connected clients
                    send_start = time.time()
                    disconnected_clients = set()
                    sent_count = 0

                    for client in self.connected_clients.copy():
                        try:
                            await client.send(json.dumps(message))
                            sent_count += 1
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                        except Exception as e:
                            logger.error(f"Error sending to client: {e}")
                            disconnected_clients.add(client)

                    send_time = (time.time() - send_start) * 1000

                    # Remove disconnected clients
                    self.connected_clients -= disconnected_clients

                    # Log broadcast stats every 60 frames
                    loop_counter += 1
                    if loop_counter % 60 == 0:
                        total_loop_time = (time.time() - loop_start) * 1000
                        mode_text = "STEREO" if self.stereo_mode else "MONO"
                        logger.info(f"Broadcast #{loop_counter} [{mode_text}] - "
                                    f"Clients: {sent_count}, "
                                    f"Message build: {message_build_time:.2f}ms, "
                                    f"Send: {send_time:.2f}ms, "
                                    f"Total loop: {total_loop_time:.2f}ms")

            await asyncio.sleep(1/60)  # 60 FPS

    async def start_server(self):
        """Start the WebSocket server"""
        if not self.initialize_camera():
            logger.error("Failed to initialize camera. Exiting.")
            return

        self.is_running = True
        server_start_time = time.time()
        mode_text = "STEREO" if self.stereo_mode else "MONO"
        logger.info(f"Starting camera server [{mode_text}] on {self.host}:{self.port} at {server_start_time}")

        # Start the WebSocket server
        server = await websockets.serve(self.handle_client, self.host, self.port)

        # Start broadcasting frames
        broadcast_task = asyncio.create_task(self.broadcast_frames())

        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            shutdown_start = time.time()
            self.is_running = False
            broadcast_task.cancel()
            if self.picam2:
                self.picam2.stop()
            shutdown_time = (time.time() - shutdown_start) * 1000
            total_runtime = time.time() - server_start_time
            logger.info(f"Server stopped (Runtime: {total_runtime:.2f}s, Shutdown: {shutdown_time:.2f}ms)")


def main():
    # Configuration
    SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
    SERVER_PORT = 8765
    IMAGE_QUALITY = 85  # JPEG quality (1-100)
    IMAGE_SIZE = (640, 480)  # Adjust based on VR glasses capability
    # IMAGE_SIZE = (1280, 960)  # Higher resolution option

    # STEREO MODE SWITCH - Set to False for mono mode
    STEREO_MODE = True

    logger.info("Starting Raspberry Pi Camera Server with stereo support...")
    startup_time = time.time()

    # Create and start server
    server = CameraServer(
        host=SERVER_HOST,
        port=SERVER_PORT,
        image_quality=IMAGE_QUALITY,
        image_size=IMAGE_SIZE,
        stereo_mode=STEREO_MODE
    )

    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    finally:
        total_runtime = time.time() - startup_time
        logger.info(f"Total application runtime: {total_runtime:.2f}s")

if __name__ == "__main__":
    main()