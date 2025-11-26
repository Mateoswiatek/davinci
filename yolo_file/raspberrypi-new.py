#!/usr/bin/env python3
"""
Raspberry Pi Camera Server for VR Project with OpenCV YOLO
Captures camera images and streams them via WebSocket to VR glasses
Also receives head tilt angles for servo control

LIGHTWEIGHT VERSION: Uses OpenCV DNN instead of PyTorch/Ultralytics
"""

import asyncio
import websockets
import json
import base64
import io
import time
import sys
import os
from gpiozero import AngularServo
from picamera2 import Picamera2
from PIL import Image
import logging

# Add the backend directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import OpenCV-based YOLO detector (reliable and working!)
try:
    from cv_ai.OpenCVSimpleYOLO import SimpleYOLODetector
    YOLO_AVAILABLE = True
    print("✅ OpenCV YOLO detector imported successfully")
except ImportError as e:
    logging.warning(f"❌ YOLO detector not available: {e}")
    YOLO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServoControl:
    def __init__(self, pan_pin=14, tilt_pin=23, roll_pin=18):
        """Initialize servo control with VR client angle ranges"""
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        self.roll_pin = roll_pin
        
        # VR Client angle ranges (centered at 0°)
        self.max_pitch = 75.0   # ±75° pitch
        self.max_yaw = 70.0     # ±70° yaw
        self.max_roll = 70.0    # ±70° roll
        
        # Servo physical limits (degrees)
        self.servo_pitch_min = 15
        self.servo_pitch_max = 165
        self.servo_yaw_min = 20
        self.servo_yaw_max = 160
        self.servo_roll_min = 20
        self.servo_roll_max = 160
        
        # Current servo positions (start at center)
        self.servo_center = 90
        self.current_pan = self.servo_center
        self.current_tilt = self.servo_center
        self.current_roll = self.servo_center
        
        min_pulse_width = 1.0/1000
        max_pulse_width = 2.0/1000
        
        # Initialize servos
        self.servos_initialized = False
        
        try:
            self.pan_servo = AngularServo(
                self.pan_pin,
                min_angle=self.servo_yaw_min,
                max_angle=self.servo_yaw_max,
                min_pulse_width=min_pulse_width,
                max_pulse_width=max_pulse_width,
                initial_angle=self.servo_center
            )
            
            self.tilt_servo = AngularServo(
                self.tilt_pin,
                min_angle=self.servo_pitch_min,
                max_angle=self.servo_pitch_max,
                min_pulse_width=min_pulse_width,
                max_pulse_width=max_pulse_width,
                initial_angle=self.servo_center
            )
            
            self.roll_servo = AngularServo(
                self.roll_pin,
                min_angle=self.servo_roll_min,
                max_angle=self.servo_roll_max,
                min_pulse_width=min_pulse_width,
                max_pulse_width=max_pulse_width,
                initial_angle=self.servo_center
            )
            
            # Move to center position
            self.move_to_center()
            self.servos_initialized = True
            logger.info("✅ Servos initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize servos: {e}")
            self.servos_initialized = False

    def vr_to_servo_angle(self, vr_angle, axis):
        """Convert VR client angles to servo angles with proper clamping"""
        if axis == 'pitch':
            # Clamp VR angle to valid range
            vr_angle = max(-self.max_pitch, min(self.max_pitch, vr_angle))
            # Map VR -75° to +75° -> Servo 15° to 165°
            servo_range = self.servo_pitch_max - self.servo_pitch_min
            normalized = (vr_angle + self.max_pitch) / (2 * self.max_pitch)
            servo_angle = self.servo_pitch_min + (normalized * servo_range)
            
        elif axis == 'yaw':
            # Clamp VR angle to valid range
            vr_angle = max(-self.max_yaw, min(self.max_yaw, vr_angle))
            # Map VR -70° to +70° -> Servo 20° to 160°
            servo_range = self.servo_yaw_max - self.servo_yaw_min
            normalized = (vr_angle + self.max_yaw) / (2 * self.max_yaw)
            servo_angle = self.servo_yaw_min + (normalized * servo_range)
            
        elif axis == 'roll':
            # Clamp VR angle to valid range
            vr_angle = max(-self.max_roll, min(self.max_roll, vr_angle))
            # Map VR -70° to +70° -> Servo 20° to 160°
            servo_range = self.servo_roll_max - self.servo_roll_min
            normalized = (vr_angle + self.max_roll) / (2 * self.max_roll)
            servo_angle = self.servo_roll_min + (normalized * servo_range)
            
        else:
            servo_angle = self.servo_center
            
        return servo_angle

    def move_to_center(self):
        """Move all servos to center position"""
        if not self.servos_initialized:
            return False
            
        try:
            # Calculate center positions based on ranges
            pitch_center = (self.servo_pitch_min + self.servo_pitch_max) / 2
            yaw_center = (self.servo_yaw_min + self.servo_yaw_max) / 2
            roll_center = (self.servo_roll_min + self.servo_roll_max) / 2
            
            self.current_pan = yaw_center
            self.current_tilt = pitch_center
            self.current_roll = roll_center
            
            self.pan_servo.angle = self.current_pan
            self.tilt_servo.angle = self.current_tilt
            self.roll_servo.angle = self.current_roll
            
            logger.info(f"Servos moved to center - Pan: {yaw_center}°, Tilt: {pitch_center}°, Roll: {roll_center}°")
            return True
            
        except Exception as e:
            logger.error(f"Error moving servos to center: {e}")
            return False

    def update_servo_positions(self, pitch, yaw, roll):
        """Update servo positions based on VR client angles"""
        if not self.servos_initialized:
            logger.warning("Servos not initialized, cannot update positions")
            return False
            
        try:
            # Convert VR angles to servo angles (includes clamping)
            pan_angle = self.vr_to_servo_angle(yaw, 'yaw')
            tilt_angle = self.vr_to_servo_angle(pitch, 'pitch')
            roll_angle = self.vr_to_servo_angle(roll, 'roll')
            
            # Update servo positions
            self.current_pan = pan_angle
            self.current_tilt = tilt_angle
            self.current_roll = roll_angle
            
            self.pan_servo.angle = pan_angle
            self.tilt_servo.angle = tilt_angle
            self.roll_servo.angle = roll_angle
            
            logger.info(f"Servos updated - VR(P:{pitch:.1f}°, Y:{yaw:.1f}°, R:{roll:.1f}°) "
                        f"-> Servo(P:{tilt_angle:.1f}°, Y:{pan_angle:.1f}°, R:{roll_angle:.1f}°)")
            return True
            
        except Exception as e:
            logger.error(f"Error updating servo positions: {e}")
            return False

    def get_current_positions(self):
        """Get current servo positions"""
        return {
            'pan': self.current_pan,
            'tilt': self.current_tilt,
            'roll': self.current_roll,
            'initialized': self.servos_initialized
        }

    def cleanup(self):
        """Clean up servo resources"""
        if self.servos_initialized:
            try:
                self.move_to_center()
                time.sleep(0.5)
                self.pan_servo.close()
                self.tilt_servo.close()
                self.roll_servo.close()
                logger.info("✅ Servos cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during servo cleanup: {e}")

class CameraServer:
    def __init__(self, host='0.0.0.0', port=8765, image_quality=85, image_size=(640, 480), enable_yolo=True):
        self.host = host
        self.port = port
        self.image_quality = image_quality
        self.image_size = image_size
        self.picam2 = None
        self.connected_clients = set()
        self.is_running = False
        self.frame_counter = 0
        
        self.latency_stats = {
            'capture_times': [],
            'encode_times': [],
            'send_times': [],
            'total_times': []
        }
        
        # Frame skipping for YOLO optimization
        self.yolo_frame_skip = 10  # Run YOLO every 10th frame (balanced performance)
        self.yolo_frame_counter = 0
        self.cached_detections = []  # Cache last YOLO results
        
        logger.info(f"🎯 Frame skipping initialized: Every {self.yolo_frame_skip} frames")
        
        # Initialize servo control
        self.servo_control = ServoControl()
        
        # Initialize OpenCV YOLO detector
        self.enable_yolo = enable_yolo and YOLO_AVAILABLE
        self.yolo_detector = None
        self.yolo_stats = {
            'detections_count': 0,
            'humans_detected': 0
        }
        
        if self.enable_yolo:
            self._initialize_yolo()

    def _initialize_yolo(self):
        """Initialize OpenCV YOLO detector (reliable version)"""
        try:
            logger.info("🔍 Initializing OpenCV YOLO detector for human detection...")
            
            # Initialize with speed-optimized settings
            self.yolo_detector = SimpleYOLODetector(
                config_path="../yolo/yolov4-tiny.cfg",
                weights_path="../yolo/yolov4-tiny.weights",
                names_path="../yolo/coco.names",
                confidence_threshold=0.3,  # Lower for better detection
                nms_threshold=0.4,
                input_size=320  # Balanced size for speed vs accuracy
            )
            
            if self.yolo_detector.is_available():
                logger.info("✅ OpenCV YOLO detector initialized successfully")
                stats = self.yolo_detector.get_performance_stats()
                logger.info(f"   Model loaded: {stats.get('model_loaded')}")
                logger.info(f"   Total classes: {stats.get('total_classes')}")
                logger.info(f"   Input size: 320x320 (speed optimized)")
                logger.info(f"   Frame skipping: Every {self.yolo_frame_skip} frames ({self.yolo_frame_skip}x speed boost)")
                logger.info(f"   Confidence threshold: 0.3 (more sensitive)")
                logger.info(f"   Camera resolution: {self.image_size} (lower for performance)")
            else:
                logger.warning("⚠️ OpenCV YOLO detector not available")
                self.enable_yolo = False
                self.yolo_detector = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenCV YOLO detector: {e}")
            self.enable_yolo = False
            self.yolo_detector = None

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
            logger.info(f"✅ Camera initialized with resolution {self.image_size} in {init_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize camera: {e}")
            return False

    def capture_frame(self):
        """Capture a frame from the camera and return as base64 encoded JPEG with timing info"""
        frame_start_time = time.time()
        timing = {}
        detections = []
        run_yolo_this_frame = False
        
        try:
            # Timestamp: Start capture
            capture_start = time.time()
            frame = self.picam2.capture_array()
            capture_end = time.time()
            
            # Timestamp: Start PIL conversion
            pil_start = time.time()
            image = Image.fromarray(frame)
            image = image.rotate(180)
            pil_end = time.time()
            
            # OpenCV YOLO Human Detection with Frame Skipping (if enabled)
            yolo_start = time.time()
            
            if self.enable_yolo and self.yolo_detector is not None:
                # Increment YOLO frame counter
                self.yolo_frame_counter += 1
                
                # Run YOLO only every N-th frame
                if self.yolo_frame_counter >= self.yolo_frame_skip:
                    run_yolo_this_frame = True
                    self.yolo_frame_counter = 0  # Reset counter
                    logger.debug(f"🔍 Running YOLO on frame {self.frame_counter + 1}")
                    
                    try:
                        # Convert PIL to OpenCV format for YOLO
                        import cv2
                        import numpy as np
                        opencv_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        
                        # Run YOLO detection
                        detections = self.yolo_detector.detect_humans(opencv_frame)
                        
                        # Cache the results for next frames
                        self.cached_detections = detections
                        
                        # Debug log
                        logger.debug(f"🎯 YOLO detected {len(detections)} humans, cached for next {self.yolo_frame_skip-1} frames")
                        
                        # Update YOLO stats
                        self.yolo_stats['detections_count'] += len(detections)
                        if detections:
                            self.yolo_stats['humans_detected'] += 1
                        
                        # Draw detections on the frame if humans are found
                        if detections:
                            opencv_frame_with_detections = self.yolo_detector.draw_detections(opencv_frame, detections)
                            # Convert back to PIL
                            image = Image.fromarray(cv2.cvtColor(opencv_frame_with_detections, cv2.COLOR_BGR2RGB))
                            
                    except Exception as yolo_error:
                        logger.warning(f"⚠️ YOLO detection error: {yolo_error}")
                        detections = []
                        self.cached_detections = []
                else:
                    # Use cached detections from previous YOLO run
                    detections = self.cached_detections
                    logger.debug(f"📋 Using cached detections: {len(detections)} humans (frame {self.yolo_frame_counter}/{self.yolo_frame_skip})")
                    
                    # Draw cached detections if any
                    if detections:
                        try:
                            import cv2
                            import numpy as np
                            opencv_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            opencv_frame_with_detections = self.yolo_detector.draw_detections(opencv_frame, detections)
                            image = Image.fromarray(cv2.cvtColor(opencv_frame_with_detections, cv2.COLOR_BGR2RGB))
                        except Exception as draw_error:
                            logger.warning(f"⚠️ Drawing cached detections error: {draw_error}")
            else:
                # YOLO disabled
                detections = []
                logger.debug(f"❌ YOLO disabled - no detections")
            
            yolo_end = time.time()
            
            # Timestamp: Start JPEG compression
            compress_start = time.time()
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.image_quality)
            buffer.seek(0)
            compress_end = time.time()
            
            # Timestamp: Start base64 encoding
            encode_start = time.time()
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            encode_end = time.time()
            
            # Total processing time
            total_time = (time.time() - frame_start_time) * 1000
            
            timing['capture_ms'] = (capture_end - capture_start) * 1000
            timing['pil_conversion_ms'] = (pil_end - pil_start) * 1000
            timing['yolo_detection_ms'] = (yolo_end - yolo_start) * 1000
            timing['yolo_ran_this_frame'] = run_yolo_this_frame
            timing['yolo_frame_skip'] = self.yolo_frame_skip
            timing['compression_ms'] = (compress_end - compress_start) * 1000
            timing['base64_ms'] = (encode_end - encode_start) * 1000
            timing['total_processing_ms'] = total_time
            timing['humans_detected'] = len(detections)
            
            # Update stats
            self.update_latency_stats(timing)
            
            # Log detailed timing every 30 frames
            self.frame_counter += 1
            if self.frame_counter % 30 == 0:
                self.log_latency_stats()
                if self.enable_yolo:
                    yolo_efficiency = (self.yolo_stats['detections_count'] / max(1, self.frame_counter // self.yolo_frame_skip)) * 100
                    logger.info(f"🎯 YOLO Efficiency - Running every {self.yolo_frame_skip} frames, Detection rate: {yolo_efficiency:.1f}%")
                
            return image_data, timing, detections
            
        except Exception as e:
            logger.error(f"❌ Error capturing frame: {e}")
            return None, {}, []

    def update_latency_stats(self, timing):
        """Update latency statistics"""
        self.latency_stats['capture_times'].append(timing.get('capture_ms', 0))
        self.latency_stats['encode_times'].append(timing.get('base64_ms', 0))
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
        avg_encode = sum(self.latency_stats['encode_times']) / len(self.latency_stats['encode_times'])
        avg_total = sum(self.latency_stats['total_times']) / len(self.latency_stats['total_times'])
        
        yolo_info = ""
        if self.enable_yolo and self.yolo_detector:
            stats = self.yolo_detector.get_performance_stats()
            if stats:
                yolo_info = f", YOLO: {stats['avg_inference_ms']:.1f}ms ({stats['fps']:.1f} FPS)"
        
        logger.info(f"📊 Frame #{self.frame_counter} - "
                    f"Capture: {avg_capture:.1f}ms, "
                    f"Encode: {avg_encode:.1f}ms, "
                    f"Total: {avg_total:.1f}ms{yolo_info}")

    async def handle_client(self, websocket):
        """Handle individual client connections"""
        client_address = websocket.remote_address
        connect_time = time.time()
        logger.info(f"🔗 Client connected: {client_address} at {connect_time}")
        
        self.connected_clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'head_angles':
                        # Handle head angles from VR glasses
                        pitch = data.get('pitch', 0)
                        yaw = data.get('yaw', 0)
                        roll = data.get('roll', 0)
                        
                        logger.info(f"🎯 Head angles - Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°, Roll: {roll:.2f}°")
                        
                        # Update servo positions
                        success = self.servo_control.update_servo_positions(pitch, yaw, roll)
                        if not success:
                            logger.warning("⚠️ Failed to update servo positions")
                            
                    elif data.get('type') == 'request_frame':
                        # Client requesting a frame
                        request_start = time.time()
                        await self.send_frame(websocket)
                        request_time = (time.time() - request_start) * 1000
                        logger.info(f"📸 Frame request processed in {request_time:.2f}ms")
                        
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON received from {client_address}")
                except Exception as e:
                    logger.error(f"❌ Error processing message from {client_address}: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            disconnect_time = time.time()
            session_duration = disconnect_time - connect_time
            logger.info(f"🔌 Client disconnected: {client_address} (session: {session_duration:.2f}s)")
        except Exception as e:
            logger.error(f"❌ Error with client {client_address}: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def send_frame(self, websocket):
        """Send a single frame to the specified client"""
        send_start_time = time.time()
        frame_result = self.capture_frame()
        
        if frame_result[0]:  # frame_data exists
            frame_data, timing, detections = frame_result
            
            message_build_start = time.time()
            message = {
                'type': 'camera_frame',
                'timestamp': time.time(),
                'frame_id': self.frame_counter,
                'image': frame_data,
                'server_timing': {
                    'capture_ms': timing.get('capture_ms', 0),
                    'yolo_detection_ms': timing.get('yolo_detection_ms', 0),
                    'compression_ms': timing.get('compression_ms', 0),
                    'total_processing_ms': timing.get('total_processing_ms', 0),
                    'humans_detected': timing.get('humans_detected', 0)
                },
                'detections': detections if self.enable_yolo else []
            }
            message_build_time = (time.time() - message_build_start) * 1000
            
            try:
                websocket_send_start = time.time()
                await websocket.send(json.dumps(message))
                websocket_send_time = (time.time() - websocket_send_start) * 1000
                
                total_send_time = (time.time() - send_start_time) * 1000
                
                logger.debug(f"📤 Frame #{self.frame_counter} sent - "
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
                    frame_data, timing, detections = frame_result
                    
                    message_start = time.time()
                    message = {
                        'type': 'camera_frame',
                        'timestamp': time.time(),
                        'frame_id': self.frame_counter,
                        'image': frame_data,
                        'server_timing': timing,
                        'detections': detections if self.enable_yolo else []
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
                            logger.error(f"❌ Error sending to client: {e}")
                            disconnected_clients.add(client)
                            
                    send_time = (time.time() - send_start) * 1000
                    
                    # Remove disconnected clients
                    self.connected_clients -= disconnected_clients
                    
                    # Log broadcast stats every 60 frames
                    loop_counter += 1
                    if loop_counter % 60 == 0:
                        total_loop_time = (time.time() - loop_start) * 1000
                        logger.info(f"📡 Broadcast #{loop_counter} - "
                                    f"Clients: {sent_count}, "
                                    f"Message build: {message_build_time:.2f}ms, "
                                    f"Send: {send_time:.2f}ms, "
                                    f"Total loop: {total_loop_time:.2f}ms")
                                    
            await asyncio.sleep(1/30)  # 30 FPS

    async def start_server(self):
        """Start the WebSocket server"""
        if not self.initialize_camera():
            logger.error("❌ Failed to initialize camera. Exiting.")
            return
            
        self.is_running = True
        server_start_time = time.time()
        
        logger.info(f"🚀 Starting camera server on {self.host}:{self.port} at {server_start_time}")
        
        # Start the WebSocket server
        server = await websockets.serve(self.handle_client, self.host, self.port)
        
        # Start broadcasting frames
        broadcast_task = asyncio.create_task(self.broadcast_frames())
        
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("🛑 Server shutdown requested")
        finally:
            shutdown_start = time.time()
            self.is_running = False
            broadcast_task.cancel()
            
            # Cleanup YOLO detector
            if self.yolo_detector:
                logger.info("🧹 OpenCV YOLO detector cleanup completed")
                
            if self.picam2:
                self.picam2.stop()
                
            # Cleanup servo control
            self.servo_control.cleanup()
            
            shutdown_time = (time.time() - shutdown_start) * 1000
            total_runtime = time.time() - server_start_time
            
            # Log final YOLO stats
            if self.enable_yolo:
                logger.info(f"📊 YOLO Stats - Total detections: {self.yolo_stats['detections_count']}, "
                           f"Frames with humans: {self.yolo_stats['humans_detected']}")
                           
            logger.info(f"✅ Server stopped (Runtime: {total_runtime:.2f}s, Shutdown: {shutdown_time:.2f}ms)")

def main():
    # Configuration
    SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
    SERVER_PORT = 8765
    IMAGE_QUALITY = 50  # JPEG quality (1-100) - lower for better performance
    IMAGE_SIZE = (640, 480)  # Lower resolution for better performance
    # IMAGE_SIZE = (800, 270)  # Adjust based on VR glasses capability
    # IMAGE_SIZE = (1600, 540)  # High resolution (slower)
    # IMAGE_SIZE = (3200, 1080)  # Adjust based on VR glasses capability
    
    # YOLO Configuration
    ENABLE_YOLO = True  # Set to False to disable human detection
    
    logger.info("🚀 Starting Raspberry Pi Camera Server with OpenCV YOLO Human Detection...")
    
    if ENABLE_YOLO and YOLO_AVAILABLE:
        logger.info("✅ OpenCV YOLO human detection enabled (reliable version with frame skipping)")
    elif ENABLE_YOLO and not YOLO_AVAILABLE:
        logger.warning("⚠️ YOLO requested but not available - running without detection")
    else:
        logger.info("ℹ️ YOLO human detection disabled")
        
    startup_time = time.time()
    
    # Create and start server
    server = CameraServer(
        host=SERVER_HOST,
        port=SERVER_PORT,
        image_quality=IMAGE_QUALITY,
        image_size=IMAGE_SIZE,
        enable_yolo=ENABLE_YOLO
    )
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("🛑 Application interrupted by user")
    finally:
        total_runtime = time.time() - startup_time
        logger.info(f"⏱️ Total application runtime: {total_runtime:.2f}s")

if __name__ == "__main__":
    main()
