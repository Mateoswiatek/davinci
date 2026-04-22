#!/usr/bin/env python3
"""
VR Angles Sender - Standalone Application
Sends head tilt angles to Raspberry Pi server without requiring camera feed
Keyboard control for testing servo movements
"""

import asyncio
import websockets
import json
import time
import logging
import threading
from pynput import keyboard
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VRAnglesSender:
    def __init__(self, server_host='192.168.113.209', server_port=8765):
        self.server_host = server_host
        self.server_port = server_port
        self.websocket = None
        self.is_running = False
        self.connection_active = False

        # Manual control angles
        self.manual_pitch = 0.0
        self.manual_yaw = 0.0
        self.manual_roll = 0.0

        # Angle limits (from center position)
        self.max_pitch_up = 90.0    # Max up
        self.max_pitch_down = -75.0 # Max down
        self.max_yaw = 70.0         # ±70° (140° field of view)
        self.max_roll = 70.0        # ±70° roll

        # Angle step for keyboard control
        self.angle_step = 5.0

        # Keyboard listener
        self.keyboard_listener = None

    def print_status(self):
        """Print current status and angles"""
        print(f"\r📍 Pitch: {self.manual_pitch:+6.1f}° | Yaw: {self.manual_yaw:+6.1f}° | Roll: {self.manual_roll:+6.1f}° | Connection: {'✅' if self.connection_active else '❌'}", end='', flush=True)

    def print_controls(self):
        """Print available controls"""
        print("\n🎮 VR Angles Sender - Standalone")
        print("=" * 50)
        print(f"📡 Server: {self.server_host}:{self.server_port}")
        print("\n🎯 Controls:")
        print("  ↑/W     - Pitch UP   (max +90°)")
        print("  ↓/S     - Pitch DOWN (max -75°)")
        print("  ←/A     - Yaw LEFT   (±70°)")
        print("  →/D     - Yaw RIGHT  (±70°)")
        print("  Q       - Roll LEFT  (±70°)")
        print("  E       - Roll RIGHT (±70°)")
        print("  R       - Reset all angles to 0°")
        print("  ESC/Ctrl+C - Quit")
        print("\n📊 Angle Limits:")
        print(f"  Pitch: {self.max_pitch_down}° to +{self.max_pitch_up}°")
        print(f"  Yaw:   ±{self.max_yaw}° (Total FOV: {self.max_yaw*2}°)")
        print(f"  Roll:  ±{self.max_roll}°")
        print(f"  Step:  {self.angle_step}° per key press")
        print("=" * 50)

    async def connect_to_server(self):
        """Connect to the Raspberry Pi server"""
        uri = f"ws://{self.server_host}:{self.server_port}"

        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Connecting to server... (attempt {attempt + 1}/{max_retries})")
                self.websocket = await websockets.connect(uri)
                self.connection_active = True
                logger.info("✅ Connected to server successfully")
                return True

            except Exception as e:
                logger.warning(f"❌ Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff

        logger.error("❌ Failed to connect to server after all attempts")
        return False

    def on_key_press(self, key):
        """Handle keyboard input"""
        try:
            # Handle special keys
            if key == keyboard.Key.up:
                self.manual_pitch = min(self.max_pitch_up, self.manual_pitch + self.angle_step)
            elif key == keyboard.Key.down:
                self.manual_pitch = max(self.max_pitch_down, self.manual_pitch - self.angle_step)
            elif key == keyboard.Key.left:
                self.manual_yaw = max(-self.max_yaw, self.manual_yaw - self.angle_step)
            elif key == keyboard.Key.right:
                self.manual_yaw = min(self.max_yaw, self.manual_yaw + self.angle_step)
            elif key == keyboard.Key.esc:
                logger.info("\n👋 Exit requested by user (ESC)")
                self.stop()
                return False

            # Handle character keys
            elif hasattr(key, 'char') and key.char:
                char = key.char.lower()

                if char == 'w':  # Pitch up
                    self.manual_pitch = min(self.max_pitch_up, self.manual_pitch + self.angle_step)
                elif char == 's':  # Pitch down
                    self.manual_pitch = max(self.max_pitch_down, self.manual_pitch - self.angle_step)
                elif char == 'a':  # Yaw left
                    self.manual_yaw = max(-self.max_yaw, self.manual_yaw - self.angle_step)
                elif char == 'd':  # Yaw right
                    self.manual_yaw = min(self.max_yaw, self.manual_yaw + self.angle_step)
                elif char == 'q':  # Roll left
                    self.manual_roll = max(-self.max_roll, self.manual_roll - self.angle_step)
                elif char == 'e':  # Roll right
                    self.manual_roll = min(self.max_roll, self.manual_roll + self.angle_step)
                elif char == 'r':  # Reset
                    self.manual_pitch = 0.0
                    self.manual_yaw = 0.0
                    self.manual_roll = 0.0
                    logger.info("\n🎯 Angles reset to 0°")

            self.print_status()

        except Exception as e:
            logger.error(f"Error handling key press: {e}")

    def start_keyboard_listener(self):
        """Start keyboard listener in a separate thread"""
        try:
            self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
            self.keyboard_listener.start()
            logger.info("⌨️  Keyboard listener started")
        except Exception as e:
            logger.error(f"Failed to start keyboard listener: {e}")

    def stop_keyboard_listener(self):
        """Stop keyboard listener"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            logger.info("⌨️  Keyboard listener stopped")

    async def send_angles_loop(self):
        """Main loop for sending angles to server"""
        last_angles = (None, None, None)
        send_interval = 0.1  # Send every 100ms

        while self.is_running:
            try:
                if not self.websocket:
                    await asyncio.sleep(1)
                    continue

                current_angles = (
                    round(self.manual_pitch, 2),
                    round(self.manual_yaw, 2),
                    round(self.manual_roll, 2)
                )

                # Only send if angles changed (to reduce traffic)
                if current_angles != last_angles:
                    message = {
                        'type': 'head_angles',
                        'pitch': current_angles[0],
                        'yaw': current_angles[1],
                        'roll': current_angles[2],
                        'timestamp': time.time()
                    }

                    await self.websocket.send(json.dumps(message))
                    last_angles = current_angles

                await asyncio.sleep(send_interval)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("🔌 Connection lost")
                self.connection_active = False
                self.print_status()

                # Try to reconnect
                if await self.connect_to_server():
                    continue
                else:
                    break

            except Exception as e:
                logger.error(f"Error sending angles: {e}")
                await asyncio.sleep(1)

    async def monitor_connection(self):
        """Monitor connection and handle reconnection"""
        while self.is_running:
            if self.websocket and not self.connection_active:
                logger.info("🔄 Attempting to reconnect...")
                if not await self.connect_to_server():
                    await asyncio.sleep(5)  # Wait before next reconnect attempt
            await asyncio.sleep(1)

    def stop(self):
        """Stop the application"""
        self.is_running = False

    async def run(self):
        """Run the angles sender application"""
        self.print_controls()

        # Connect to server
        if not await self.connect_to_server():
            logger.error("❌ Could not establish initial connection")
            return

        self.is_running = True

        # Start keyboard listener
        self.start_keyboard_listener()

        try:
            # Print initial status
            self.print_status()

            # Create tasks
            send_task = asyncio.create_task(self.send_angles_loop())
            monitor_task = asyncio.create_task(self.monitor_connection())

            # Wait for tasks to complete
            await asyncio.gather(send_task, monitor_task, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.is_running = False
            self.stop_keyboard_listener()

            if self.websocket:
                await self.websocket.close()

            print("\n👋 VR Angles Sender stopped")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Interrupt received, shutting down...")
    sys.exit(0)

def main():
    # Configuration - Update this with your Raspberry Pi's IP address
    PI_SERVER_HOST = '192.168.113.209'
    PI_SERVER_PORT = 8765

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Create and run sender
    sender = VRAnglesSender(server_host=PI_SERVER_HOST, server_port=PI_SERVER_PORT)

    try:
        asyncio.run(sender.run())
    except KeyboardInterrupt:
        logger.info("\n👋 Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()