#!/usr/bin/env python3
"""
Raspberry Pi Camera Server for VR Project (OpenCV version)
Captures camera images from any V4L2-compatible camera (e.g., Arducam PiVariety)
Streams them via WebSocket to VR glasses and handles servo control.
"""

import asyncio
import websockets
import json
import base64
import io
import time
import logging

import cv2
from gpiozero import AngularServo
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServoControl:
    def __init__(self, pan_pin=14, tilt_pin=23, roll_pin=18):
        """Initialize servo control with VR client angle ranges"""
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        self.roll_pin = roll_pin

        self.max_pitch = 75.0
        self.max_yaw = 70.0
        self.max_roll = 70.0

        self.servo_pitch_min = 15
        self.servo_pitch_max = 165
        self.servo_yaw_min = 20
        self.servo_yaw_max = 160
        self.servo_roll_min = 20
        self.servo_roll_max = 160

        self.servo_center = 90
        self.current_pan = self.servo_center
        self.current_tilt = self.servo_center
        self.current_roll = self.servo_center

        min_pulse_width = 1.0/1000
        max_pulse_width = 2.0/1000

        # Initialize servos
        self.pan_servo = AngularServo(
            self.pan_pin, min_angle=self.servo_yaw_min, max_angle=self.servo_yaw_max,
            min_pulse_width=min_pulse_width, max_pulse_width=max_pulse_width,
            initial_angle=self.servo_center
        )
        self.tilt_servo = AngularServo(
            self.tilt_pin, min_angle=self.servo_pitch_min, max_angle=self.servo_pitch_max,
            min_pulse_width=min_pulse_width, max_pulse_width=max_pulse_width,
            initial_angle=self.servo_center
        )
        self.roll_servo = AngularServo(
            self.roll_pin, min_angle=self.servo_roll_min, max_angle=self.servo_roll_max,
            min_pulse_width=min_pulse_width, max_pulse_width=max_pulse_width,
            initial_angle=self.servo_center
        )

        self.move_to_center()
        logger.info("Servos initialized successfully")

    def vr_to_servo_angle(self, vr_angle, axis):
        if axis == 'pitch':
            vr_angle = max(-self.max_pitch, min(self.max_pitch, vr_angle))
            servo_range = self.servo_pitch_max - self.servo_pitch_min
            normalized = (vr_angle + self.max_pitch) / (2 * self.max_pitch)
            return self.servo_pitch_min + normalized * servo_range
        elif axis == 'yaw':
            vr_angle = max(-self.max_yaw, min(self.max_yaw, vr_angle))
            servo_range = self.servo_yaw_max - self.servo_yaw_min
            normalized = (vr_angle + self.max_yaw) / (2 * self.max_yaw)
            return self.servo_yaw_min + normalized * servo_range
        elif axis == 'roll':
            vr_angle = max(-self.max_roll, min(self.max_roll, vr_angle))
            servo_range = self.servo_roll_max - self.servo_roll_min
            normalized = (vr_angle + self.max_roll) / (2 * self.max_roll)
            return self.servo_roll_min + normalized * servo_range
        else:
            return self.servo_center

    def move_to_center(self):
        pitch_center = (self.servo_pitch_min + self.servo_pitch_max) / 2
        yaw_center = (self.servo_yaw_min + self.servo_yaw_max) / 2
        roll_center = (self.servo_roll_min + self.servo_roll_max) / 2

        self.current_pan = yaw_center
        self.current_tilt = pitch_center
        self.current_roll = roll_center

        self.pan_servo.angle = self.current_pan
        self.tilt_servo.angle = self.current_tilt
        self.roll_servo.angle = self.current_roll

        logger.info(f"Servos moved to center - Pan: {yaw_center}, Tilt: {pitch_center}, Roll: {roll_center}")

    def update_servo_positions(self, pitch, yaw, roll):
        self.current_pan = self.vr_to_servo_angle(yaw, 'yaw')
        self.current_tilt = self.vr_to_servo_angle(pitch, 'pitch')
        self.current_roll = self.vr_to_servo_angle(roll, 'roll')

        self.pan_servo.angle = self.current_pan
        self.tilt_servo.angle = self.current_tilt
        self.roll_servo.angle = self.current_roll

        logger.info(f"Servos updated - VR(P:{pitch},Y:{yaw},R:{roll}) -> Servo(P:{self.current_tilt},Y:{self.current_pan},R:{self.current_roll})")

class CameraServer:
    def __init__(self, host='0.0.0.0', port=8765, image_quality=85, image_size=(640,480), camera_index=0):
        self.host = host
        self.port = port
        self.image_quality = image_quality
        self.image_size = image_size
        self.camera_index = camera_index
        self.cap = None
        self.connected_clients = set()
        self.is_running = False
        self.frame_counter = 0

        self.servo_control = ServoControl()

    def initialize_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size[1])
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return False
        logger.info(f"Camera initialized with resolution {self.image_size}")
        return True

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Rotate 180° if needed
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Convert to JPEG
        pil_img = Image.fromarray(frame)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=self.image_quality)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.frame_counter += 1
        return img_base64

    async def handle_client(self, websocket):
        self.connected_clients.add(websocket)
        client_address = websocket.remote_address
        logger.info(f"Client connected: {client_address}")

        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get('type') == 'head_angles':
                    pitch = data.get('pitch',0)
                    yaw = data.get('yaw',0)
                    roll = data.get('roll',0)
                    self.servo_control.update_servo_positions(pitch, yaw, roll)
                elif data.get('type') == 'request_frame':
                    await self.send_frame(websocket)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"Client disconnected: {client_address}")

    async def send_frame(self, websocket):
        img_base64 = self.capture_frame()
        if img_base64:
            message = {
                'type':'camera_frame',
                'timestamp':time.time(),
                'frame_id': self.frame_counter,
                'image': img_base64
            }
            await websocket.send(json.dumps(message))

    async def broadcast_frames(self):
        while self.is_running:
            if self.connected_clients:
                img_base64 = self.capture_frame()
                if img_base64:
                    message = {
                        'type':'camera_frame',
                        'timestamp': time.time(),
                        'frame_id': self.frame_counter,
                        'image': img_base64
                    }
                    disconnected = set()
                    for client in self.connected_clients.copy():
                        try:
                            await client.send(json.dumps(message))
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)
                    self.connected_clients -= disconnected
            await asyncio.sleep(1/30)

    async def start_server(self):
        if not self.initialize_camera():
            return
        self.is_running = True
        server = await websockets.serve(self.handle_client, self.host, self.port)
        broadcast_task = asyncio.create_task(self.broadcast_frames())
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            self.is_running = False
            broadcast_task.cancel()
            self.cap.release()
            logger.info("Camera released, server stopped")

def main():
    server = CameraServer(
        host='0.0.0.0',
        port=8765,
        image_quality=60,
        image_size=(1600,540),
        camera_index=0
    )
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Application interrupted")

if __name__ == "__main__":
    main()
