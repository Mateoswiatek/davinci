# Jak to zsetupować?

## RASPBERRY

1. Podłącz raspberry do internetu
2. Odpal ten plik za pomocą komendy `/home/pi/venv/davinci/bin/python /home/pi/Desktop/project/davinci/raspberrypi.py`

```py
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

from gpiozero import AngularServo
from picamera2 import Picamera2
from PIL import Image
import logging

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
        # try:
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
        logger.info("Servos initialized successfully")

        # except Exception as e:
        #     logger.error(f"Failed to initialize servos: {e}")
        #     self.servos_initialized = False

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

                logger.info("Servos cleaned up successfully")

            except Exception as e:
                logger.error(f"Error during servo cleanup: {e}")


class CameraServer:
    def __init__(self, host='0.0.0.0', port=8765, image_quality=85, image_size=(640, 480)):
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

        # Initialize servo control
        self.servo_control = ServoControl()

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
            logger.info(f"Camera initialized with resolution {self.image_size} in {init_time:.2f}ms")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False

    def capture_frame(self):
        """Capture a frame from the camera and return as base64 encoded JPEG with timing info"""
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
            timing['compression_ms'] = (compress_end - compress_start) * 1000
            timing['base64_ms'] = (encode_end - encode_start) * 1000
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

        logger.info(f"Latency Stats (Frame #{self.frame_counter}) - "
                    f"Capture: {avg_capture:.2f}ms, "
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
                        # Handle head angles from VR glasses
                        pitch = data.get('pitch', 0)
                        yaw = data.get('yaw', 0)
                        roll = data.get('roll', 0)

                        logger.info(f"Head angles - Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°, Roll: {roll:.2f}°")

                        # Update servo positions
                        success = self.servo_control.update_servo_positions(pitch, yaw, roll)
                        if not success:
                            logger.warning("Failed to update servo positions")

                        #TODO (28.05.2025): Integracja z Servo
                        # Here you would integrate with your servo control code
                        # self.control_servos(pitch, yaw, roll)

                    elif data.get('type') == 'request_frame':
                        # Client requesting a frame
                        request_start = time.time()
                        await self.send_frame(websocket)
                        request_time = (time.time() - request_start) * 1000
                        logger.info(f"Frame request processed in {request_time:.2f}ms")

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
            message = {
                'type': 'camera_frame',
                'timestamp': time.time(),
                'frame_id': self.frame_counter,
                'image': frame_data,
                'server_timing': {
                    'capture_ms': timing.get('capture_ms', 0),
                    'compression_ms': timing.get('compression_ms', 0),
                    'total_processing_ms': timing.get('total_processing_ms', 0)
                }
            }
            message_build_time = (time.time() - message_build_start) * 1000

            try:
                websocket_send_start = time.time()
                await websocket.send(json.dumps(message))
                websocket_send_time = (time.time() - websocket_send_start) * 1000

                total_send_time = (time.time() - send_start_time) * 1000

                logger.debug(f"Frame #{self.frame_counter} sent - "
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
                    message = {
                        'type': 'camera_frame',
                        'timestamp': time.time(),
                        'frame_id': self.frame_counter,
                        'image': frame_data,
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
                        logger.info(f"Broadcast #{loop_counter} - "
                                    f"Clients: {sent_count}, "
                                    f"Message build: {message_build_time:.2f}ms, "
                                    f"Send: {send_time:.2f}ms, "
                                    f"Total loop: {total_loop_time:.2f}ms")

            await asyncio.sleep(1/30)  # 30 FPS

    async def start_server(self):
        """Start the WebSocket server"""
        if not self.initialize_camera():
            logger.error("Failed to initialize camera. Exiting.")
            return

        self.is_running = True
        server_start_time = time.time()
        logger.info(f"Starting camera server on {self.host}:{self.port} at {server_start_time}")

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
    # IMAGE_SIZE = (3200, 1080)  # Adjust based on VR glasses capability

    logger.info("Starting Raspberry Pi Camera Server with detailed timing...")
    startup_time = time.time()

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
    finally:
        total_runtime = time.time() - startup_time
        logger.info(f"Total application runtime: {total_runtime:.2f}s")

if __name__ == "__main__":
    main()
```
3. Uruchom komendę `ngrok http http://localhost:8765`

Dostaniesz taki response:
```
ngrok                                                                                                                                     (Ctrl+C to quit)
                                                                                                                                                          
🤖 Want to hang with ngrokkers on our new Discord? http://ngrok.com/discord                                                                               
                                                                                                                                                          
Session Status                online                                                                                                                      
Account                       radoslaw.mysliwiec@ivyconsultants.com (Plan: Free)                                                                          
Version                       3.23.1                                                                                                                      
Region                        Europe (eu)                                                                                                                 
Latency                       34ms                                                                                                                        
Web Interface                 http://127.0.0.1:4040                                                                                                       
Forwarding                    https://4b63-46-205-197-171.ngrok-free.app -> http://localhost:8765                                                         
                                                                                                                                                          
Connections                   ttl     opn     rt1     rt5     p50     p90                                                                                 
                              2       0       0.00    0.00    31.52   48.60                                                                               
                                                                                                                                                          
HTTP Requests
```

4. Skopiuj do schowka fragment forwarding url - w tym przypadku to jest `4b63-46-205-197-171.ngrok-free.app`

## TWÓJ KOMPUTER

1. Podłącz Twój komputer do internetu
2. Skopiuj ten kod do pliku `vr.html`

```
<!DOCTYPE html>
<html lang="pl">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1.0" />
    <title>Stereoskopia – Meta Quest 3 (stream + head-tracking)</title>
    <style>
      body { margin: 0; background: #000; color: #fff; font-family: Arial; }
      #vrButton {
        position: absolute; bottom: 24px; left: 50%;
        transform: translateX(-50%);
        padding: 15px 32px; font-size: 18px; border: none;
        border-radius: 6px; background: #007bff; color: #fff;
        cursor: pointer; z-index: 10;
      }
      #vrButton:disabled { background: #666; cursor: not-allowed; }
      canvas { display: block; }
    </style>
  </head>

  <body>
    <button id="vrButton">Enter VR</button>
    <canvas id="canvas"></canvas>

    <script type="module">
      /* ---------------- konfiguracja ---------------- */
      const WS_URL           = "wss://4b63-46-205-197-171.ngrok-free.app";  // <-- RPi IP:port
      const SEND_INTERVAL_MS = 100;                        // co ile wysyłać kąty
      const Y_OFFSET = -0.25;
      const Z_DIST   = -2.0;

      /* ---------------- importy ---------------- */
      import * as THREE from
        "https://cdn.jsdelivr.net/npm/three@0.164.0/build/three.module.js";

      /* ---------------- zmienne globalne ---------------- */
      let scene, camera, renderer, headLocked;
      let leftMesh, rightMesh, leftTex, rightTex;
      let xrSession = null;
      let lastSent  = 0;
      const quat    = new THREE.Quaternion();
      const euler   = new THREE.Euler();

      /* ---------------- WebSocket ---------------- */
      const ws = new WebSocket(WS_URL);

      ws.onopen    = () => console.log("WS connected");
      ws.onclose   = e => console.log("WS closed", e.reason);
      ws.onerror   = e => console.error("WS error", e);

      ws.onmessage = ({ data }) => {
        try {
          const msg = JSON.parse(data);
          if (msg.type !== "camera_frame" || !msg.image) return;

          const img = new Image();
          img.onload = () => {
            const half = img.width / 2, h = img.height;

            const lC = Object.assign(document.createElement("canvas"),
                                      { width: half, height: h });
            const rC = Object.assign(document.createElement("canvas"),
                                      { width: half, height: h });

            lC.getContext("2d")
               .drawImage(img, 0, 0, half, h, 0, 0, half, h);
            rC.getContext("2d")
               .drawImage(img, half, 0, half, h, 0, 0, half, h);

            leftTex.image  = lC;
            rightTex.image = rC;
            leftTex.needsUpdate  = true;
            rightTex.needsUpdate = true;
          };
          img.src = `data:image/jpeg;base64,${msg.image}`;
        } catch (err) { console.error(err); }
      };

      /* ---------------- inicjalizacja sceny ---------------- */
      init();
      function init() {
        scene            = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        camera = new THREE.PerspectiveCamera(
          75, innerWidth / innerHeight, 0.1, 1000
        );
        camera.position.set(0, 1.6, 0);
        scene.add(camera);

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.xr.enabled = true;
        renderer.setPixelRatio(devicePixelRatio);
        renderer.setSize(innerWidth, innerHeight);
        renderer.setClearColor(0x000000, 1);

        headLocked = new THREE.Group();
        camera.add(headLocked);

        const geom = new THREE.PlaneGeometry(3, 2.25);
        const phMat = new THREE.MeshBasicMaterial({
          color: 0x444444, side: THREE.DoubleSide
        });

        leftTex  = new THREE.Texture();
        rightTex = new THREE.Texture();
        leftMesh  = new THREE.Mesh(geom, phMat.clone());
        rightMesh = new THREE.Mesh(geom, phMat.clone());
        leftMesh.layers.set(1);
        rightMesh.layers.set(2);
        headLocked.add(leftMesh, rightMesh);

        setupVRView();

        if ("xr" in navigator) {
          navigator.xr
            .isSessionSupported("immersive-vr")
            .then(ok => vrButton.disabled = !ok);
        } else vrButton.disabled = true;

        vrButton.onclick = startStopXR;
        window.onresize  = () => {
          camera.aspect = innerWidth / innerHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(innerWidth, innerHeight);
        };

        renderer.setAnimationLoop(renderLoop);
      }

      function setupVRView() {
        leftMesh.position.set(0, Y_OFFSET, Z_DIST);
        rightMesh.position.copy(leftMesh.position);
      }

      function setEyeLayers() {
        const rig = renderer.xr.getCamera();
        if (rig.cameras?.length >= 2) {
          rig.cameras[0].layers.set(1);   // lewe
          rig.cameras[1].layers.set(2);   // prawe
        }
      }

      async function startStopXR() {
        if (renderer.xr.isPresenting) { await xrSession.end(); return; }

        xrSession = await navigator.xr.requestSession("immersive-vr", {
          optionalFeatures: ["local-floor", "bounded-floor"]
        });
        await renderer.xr.setSession(xrSession);
        vrButton.textContent = "Exit VR";
        xrSession.addEventListener("end", () => {
          vrButton.textContent = "Enter VR";
        });
      }

      /* ---------------- pętla renderująca ---------------- */
      function renderLoop(_, frame) {
        if (frame && renderer.xr.isPresenting) setEyeLayers();

        sendHeadAngles();     // <- nasz dodatkowy krok
        renderer.render(scene, camera);
      }

      /* ---------------- wysyłanie kątów ---------------- */
      function sendHeadAngles() {
        const now = performance.now();
        if (now - lastSent < SEND_INTERVAL_MS) return;
        if (ws.readyState !== WebSocket.OPEN)      return;

        camera.getWorldQuaternion(quat);
        euler.setFromQuaternion(quat, "YXZ");  // yaw-pitch-roll

        const pitch = THREE.MathUtils.radToDeg(euler.x);
        const yaw   = THREE.MathUtils.radToDeg(euler.y);
        const roll  = THREE.MathUtils.radToDeg(euler.z);

        ws.send(
          JSON.stringify({
            type:  "head_angles",
            pitch: pitch,
            yaw:   yaw,
            roll:  roll,
            ts:    Date.now()
          })
        );
        lastSent = now;
      }
    </script>
  </body>
</html>
```

3. W miejsce WS_URL wklej odpowiednią część URL skopiowaną do schowka wcześniej
4. Uruchom serwer http w directory, w którym siedzi plik vr.html za pomocą komendy `python3 -m http.server 8080`
5. Uruchom komendę `ngrok http http://localhost:8080` (UWAGA: ngrok darmowy pozwala na tylko jeden serwer na raz, dlatego polecam założyć dwa konta po prostu)

Dostaniesz taki response:
```
ngrok                                                                                                                                     (Ctrl+C to quit)
                                                                                                                                                          
Take our ngrok in production survey! https://forms.gle/aXiBFWzEA36DudFn6                                                                                  
                                                                                                                                                          
Session Status                online                                                                                                                      
Account                       radek.m2001@gmail.com (Plan: Free)                                                                                          
Version                       3.23.1                                                                                                                      
Region                        Europe (eu)                                                                                                                 
Latency                       33ms                                                                                                                        
Web Interface                 http://127.0.0.1:4040                                                                                                       
Forwarding                    https://68f8-46-205-197-171.ngrok-free.app -> http://localhost:8080                                                         
                                                                                                                                                          
Connections                   ttl     opn     rt1     rt5     p50     p90                                                                                 
                              94      0       0.00    0.00    0.00    0.01                                                                                
                                                                                                                                                          
HTTP Requests                                                                                                                                             
-------------                                                   
```

6. Zwróć uwagę na Forwarding URL (pełne) oraz ścieżkę do pliku vr.html. W tym przypadku będzie to `https://68f8-46-205-197-171.ngrok-free.app/vr.html`.

## META QUEST 3S

1. Wejdź w przeglądarkę domyślną meta questa i po prostu wpisz w wyszukiwarkę ten pełny Forwarding URL. Działa!
