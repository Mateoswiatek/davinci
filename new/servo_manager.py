"""
Servo Manager — pan/tilt/roll control for DaVinci VR remote head.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class ServoConfig:
    pan_pin: int = 14      # GPIO BCM: pan = yaw
    tilt_pin: int = 15     # GPIO BCM: tilt = pitch
    roll_pin: int = 18     # GPIO BCM: roll

    min_angle: float = -90.0
    max_angle: float = 90.0
    step: float = 0.25     # Quantization step in degrees

    min_pulse_width: float = 0.0005   # 0.5 ms
    max_pulse_width: float = 0.0025   # 2.5 ms
    deadband: float = 0.0             # min change (°) to actually move servo (SG90 ma własny dead zone)

    # Initial position on startup (degrees, within min_angle..max_angle)
    initial_pan: float = 0.0
    initial_tilt: float = 0.0
    initial_roll: float = 0.0


class ServoManager:
    """
    Controls three gpiozero AngularServos.
    Falls back to simulation when gpiozero is unavailable (dev machine).
    """

    def __init__(self, config: ServoConfig):
        self.config = config
        self._pan_angle = 0.0
        self._tilt_angle = 0.0
        self._roll_angle = 0.0
        self._pan = None
        self._tilt = None
        self._roll = None
        self._simulated = False
        self._last_log: float = 0.0

    def initialize(self) -> bool:
        try:
            from gpiozero import AngularServo, Device

            # RPi 5 uses RP1 GPIO chip — lgpio gives stable PWM (pigpio won't work)
            # RPi 4 and earlier can use pigpio for best results
            self._setup_pin_factory()

            kw = dict(
                min_angle=self.config.min_angle,
                max_angle=self.config.max_angle,
                min_pulse_width=self.config.min_pulse_width,
                max_pulse_width=self.config.max_pulse_width,
            )
            self._pan = AngularServo(self.config.pan_pin, **kw)
            self._tilt = AngularServo(self.config.tilt_pin, **kw)
            self._roll = AngularServo(self.config.roll_pin, **kw)
            logger.info(
                f"Servos on GPIO pan={self.config.pan_pin}, "
                f"tilt={self.config.tilt_pin}, roll={self.config.roll_pin}"
            )
            # Ustaw pozycję startową na wszystkich trzech od razu — wszystkie trzymają PWM
            self._pan_angle  = self._q(self._c(self.config.initial_pan))
            self._tilt_angle = self._q(self._c(self.config.initial_tilt))
            self._roll_angle = self._q(self._c(self.config.initial_roll))
            self._pan.angle  = self._pan_angle
            self._tilt.angle = self._tilt_angle
            self._roll.angle = self._roll_angle
            logger.info(
                f"Initial position: pan={self._pan_angle}°, "
                f"tilt={self._tilt_angle}°, roll={self._roll_angle}°"
            )
        except Exception as e:
            import traceback
            logger.warning(f"gpiozero unavailable — simulation mode")
            logger.warning(f"  Error: {e}")
            logger.warning(f"  Traceback:\n{traceback.format_exc()}")
            self._simulated = True
        return True

    @staticmethod
    def _setup_pin_factory():
        """Set the best available pin factory. lgpio preferred on RPi 5."""
        from gpiozero import Device
        try:
            from gpiozero.pins.lgpio import LGPIOFactory
            Device.pin_factory = LGPIOFactory()
            logger.info("Pin factory: lgpio (stable PWM)")
            return
        except Exception as e:
            logger.debug(f"lgpio not available: {e}")
        try:
            from gpiozero.pins.pigpio import PiGPIOFactory
            Device.pin_factory = PiGPIOFactory()
            logger.info("Pin factory: pigpio (stable PWM)")
            return
        except Exception as e:
            logger.debug(f"pigpio not available: {e}")
        logger.warning("Pin factory: software PWM fallback — servo may jitter")

    def move(self, pitch: Optional[float], yaw: Optional[float], roll: Optional[float]):
        """Move servos to absolute target angles (relative to initial/neutral position). None = hold current position."""
        if pitch is not None:
            new = self._q(self._c(pitch + self.config.initial_tilt))
            if abs(new - self._tilt_angle) > self.config.deadband:
                self._tilt_angle = new
                if self._tilt:
                    try:
                        self._tilt.angle = self._tilt_angle
                    except Exception as e:
                        logger.error(f"gpiozero tilt error: {e}")

        if yaw is not None:
            new = self._q(self._c(yaw + self.config.initial_pan))
            if abs(new - self._pan_angle) > self.config.deadband:
                self._pan_angle = new
                if self._pan:
                    try:
                        self._pan.angle = self._pan_angle
                    except Exception as e:
                        logger.error(f"gpiozero pan error: {e}")

        if roll is not None:
            new = self._q(self._c(roll + self.config.initial_roll))
            if abs(new - self._roll_angle) > self.config.deadband:
                self._roll_angle = new
                if self._roll:
                    try:
                        self._roll.angle = self._roll_angle
                    except Exception as e:
                        logger.error(f"gpiozero roll error: {e}")

        now = time.monotonic()
        if now - self._last_log >= 1.0:
            prefix = "[SIM]" if self._simulated else "[HW]"
            logger.info(
                f"{prefix} pan={self._pan_angle:.2f}° "
                f"tilt={self._tilt_angle:.2f}° "
                f"roll={self._roll_angle:.2f}°"
            )
            self._last_log = now

    def get_angles(self) -> Dict[str, float]:
        return {
            'pan': self._pan_angle,
            'tilt': self._tilt_angle,
            'roll': self._roll_angle,
        }

    def center(self):
        self.move(0.0, 0.0, 0.0)

    def close(self):
        for servo in (self._pan, self._tilt, self._roll):
            if servo:
                try:
                    servo.close()
                except Exception:
                    pass

    def _q(self, angle: float) -> float:
        s = self.config.step
        return round(angle / s) * s if s > 0 else angle

    def _c(self, angle: float) -> float:
        return max(self.config.min_angle, min(self.config.max_angle, angle))


class ConnectionManager:
    """
    WebSocket client roles: first connected = controller, rest = observers.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._controller_id: Optional[str] = None
        self._clients: Dict[str, str] = {}

    async def on_connect(self, ws_id: str) -> str:
        """Returns assigned role: 'controller' or 'observer'."""
        async with self._lock:
            if self._controller_id is None:
                self._controller_id = ws_id
                role = 'controller'
            else:
                role = 'observer'
            self._clients[ws_id] = role
            logger.info(f"Client connected as {role} (id={ws_id[:8]})")
            return role

    async def on_disconnect(self, ws_id: str) -> bool:
        """Returns True if the controller slot is now open."""
        async with self._lock:
            self._clients.pop(ws_id, None)
            was_controller = (ws_id == self._controller_id)
            if was_controller:
                self._controller_id = None
                logger.info("Controller disconnected — slot open")
            return was_controller

    async def request_control(self, ws_id: str) -> bool:
        """Claim controller role if slot is free. Returns success."""
        async with self._lock:
            if ws_id not in self._clients:
                return False
            if self._controller_id in (None, ws_id):
                self._controller_id = ws_id
                self._clients[ws_id] = 'controller'
                logger.info(f"Client {ws_id[:8]} claimed controller")
                return True
            return False

    async def release_control(self, ws_id: str):
        """Voluntarily give up controller role."""
        async with self._lock:
            if self._controller_id == ws_id:
                self._controller_id = None
                self._clients[ws_id] = 'observer'
                logger.info(f"Client {ws_id[:8]} released controller")

    def is_controller(self, ws_id: str) -> bool:
        return self._controller_id == ws_id

    def controller_available(self) -> bool:
        return self._controller_id is None

    def get_role(self, ws_id: str) -> str:
        return self._clients.get(ws_id, 'unknown')
