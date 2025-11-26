#!/usr/bin/env python3
"""
Real-Time Scheduling dla ultra-niskiej latencji
"""

import os
import ctypes
import threading
from dataclasses import dataclass
from typing import Optional

# Import Linux scheduling constants
SCHED_OTHER = 0
SCHED_FIFO = 1
SCHED_RR = 2
SCHED_BATCH = 3
SCHED_IDLE = 5
SCHED_DEADLINE = 6


@dataclass
class SchedParam:
    """Linux sched_param structure"""
    sched_priority: int


class RealtimeScheduler:
    """Zarządza real-time scheduling dla procesów i wątków"""

    # Priorytety dla różnych komponentów (1-99, wyższy = ważniejszy)
    PRIORITY_CAMERA_CAPTURE = 90      # Najwyższy - camera nie może dropować frames
    PRIORITY_NETWORK_SEND = 85        # Wysoki - natychmiastowe wysyłanie
    PRIORITY_ENCODING = 80            # Wysoko-średni
    PRIORITY_YOLO = 70                # Średni - może czekać
    PRIORITY_STATS = 50               # Niski

    def __init__(self):
        self.libc = ctypes.CDLL('libc.so.6')

    def set_realtime_priority(self, priority: int,
                             policy: int = SCHED_FIFO,
                             pid: int = 0):
        """
        Ustaw real-time priority dla procesu/wątku

        Args:
            priority: 1-99 (wyższy = ważniejszy)
            policy: SCHED_FIFO lub SCHED_RR
            pid: 0 = current thread, >0 = specific PID
        """

        class sched_param(ctypes.Structure):
            _fields_ = [('sched_priority', ctypes.c_int)]

        param = sched_param(priority)

        result = self.libc.sched_setscheduler(
            pid,
            policy,
            ctypes.byref(param)
        )

        if result != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"sched_setscheduler failed: {os.strerror(errno)}")

        policy_name = {
            SCHED_FIFO: "SCHED_FIFO",
            SCHED_RR: "SCHED_RR"
        }.get(policy, f"POLICY_{policy}")

        print(f"Set {policy_name} priority {priority} for PID/TID {pid or 'current'}")

    def set_thread_realtime(self, priority: int, policy: int = SCHED_FIFO):
        """Ustaw RT priority dla bieżącego wątku"""
        tid = threading.get_native_id()
        self.set_realtime_priority(priority, policy, tid)

    def get_priority(self, pid: int = 0) -> tuple:
        """Pobierz obecny priority i policy"""
        policy = self.libc.sched_getscheduler(pid)

        class sched_param(ctypes.Structure):
            _fields_ = [('sched_priority', ctypes.c_int)]

        param = sched_param()
        self.libc.sched_getparam(pid, ctypes.byref(param))

        return policy, param.sched_priority

    def set_nice(self, nice_value: int, pid: int = 0):
        """
        Ustaw nice value (-20 do 19, niższy = wyższy priorytet)
        Dla procesów non-realtime
        """
        try:
            os.setpriority(os.PRIO_PROCESS, pid, nice_value)
            print(f"Set nice value {nice_value} for PID {pid or 'current'}")
        except PermissionError:
            print("Need sudo for negative nice values")

    def check_realtime_limits(self):
        """Sprawdź limity RT dla użytkownika"""
        try:
            with open('/etc/security/limits.conf', 'r') as f:
                content = f.read()
                print("=== RT Limits Config ===")
                for line in content.split('\n'):
                    if 'rtprio' in line or 'nice' in line:
                        print(line)
        except Exception as e:
            print(f"Cannot read limits.conf: {e}")

        # Sprawdź obecne limity
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_RTPRIO)
        print(f"\nCurrent RT priority limits: soft={soft}, hard={hard}")


class RealtimeThread(threading.Thread):
    """Thread z automatycznym ustawieniem RT priority"""

    def __init__(self, target, priority: int,
                 policy: int = SCHED_FIFO,
                 cpu_affinity: Optional[list] = None,
                 *args, **kwargs):
        super().__init__(target=self._wrapped_target, *args, **kwargs)
        self._user_target = target
        self._priority = priority
        self._policy = policy
        self._cpu_affinity = cpu_affinity
        self.scheduler = RealtimeScheduler()

    def _wrapped_target(self):
        """Wrapper który ustawia RT przed uruchomieniem target"""
        try:
            # Ustaw CPU affinity jeśli określono
            if self._cpu_affinity:
                import psutil
                p = psutil.Process()
                p.cpu_affinity(self._cpu_affinity)

            # Ustaw RT priority
            self.scheduler.set_thread_realtime(self._priority, self._policy)

            print(f"RT Thread {threading.get_native_id()} started: "
                  f"priority={self._priority}, "
                  f"CPUs={self._cpu_affinity}")

            # Uruchom właściwą funkcję
            self._user_target()

        except Exception as e:
            print(f"RT Thread error: {e}")
            raise


def configure_rt_kernel_params():
    """Konfiguruje parametry kernela dla RT"""

    print("=== Kernel RT Configuration ===\n")

    params = {
        # Zwiększ budżet czasu dla RT tasks (95% CPU)
        "/proc/sys/kernel/sched_rt_runtime_us": "950000",

        # Okres dla RT (1 sekunda)
        "/proc/sys/kernel/sched_rt_period_us": "1000000",

        # Wyłącz CPU frequency scaling
        "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor": "performance",

        # Minimalna latencja wakeup
        "/proc/sys/kernel/sched_min_granularity_ns": "1000000",  # 1ms
        "/proc/sys/kernel/sched_wakeup_granularity_ns": "500000",  # 0.5ms
    }

    print("Execute these commands:")
    for param, value in params.items():
        if '*' in param:
            print(f"  for cpu in /sys/devices/system/cpu/cpu[0-9]*; do")
            print(f"    echo {value} | sudo tee $cpu/cpufreq/scaling_governor")
            print(f"  done")
        else:
            print(f"  echo {value} | sudo tee {param}")

    print("\n=== RT Preempt Kernel ===")
    print("Sprawdź czy masz RT kernel:")
    print("  uname -a | grep PREEMPT")
    print("\nJeśli nie, zainstaluj:")
    print("  sudo apt-get install linux-image-rt-arm64")


def setup_rt_limits():
    """Konfiguruje /etc/security/limits.conf dla RT"""

    config = """
# Real-time priorities dla VR streaming
@realtime    -    rtprio    99
@realtime    -    nice      -20
@realtime    -    memlock   unlimited

# Lub dla konkretnego użytkownika:
pi           -    rtprio    99
pi           -    nice      -20
pi           -    memlock   unlimited
"""

    print("=== Add to /etc/security/limits.conf ===")
    print(config)
    print("\nPo edycji wyloguj się i zaloguj ponownie!")


# Przykład użycia
def camera_capture_loop():
    """Symulacja pętli camera capture"""
    import time
    print(f"Camera thread {threading.get_native_id()} running...")
    for i in range(5):
        time.sleep(0.1)
        print(f"  Captured frame {i}")


def network_send_loop():
    """Symulacja wysyłania sieciowego"""
    import time
    print(f"Network thread {threading.get_native_id()} running...")
    for i in range(5):
        time.sleep(0.15)
        print(f"  Sent packet {i}")


if __name__ == "__main__":
    scheduler = RealtimeScheduler()

    print("=== Real-Time Scheduling Setup ===\n")

    # Sprawdź limity
    scheduler.check_realtime_limits()

    print("\n=== Example: RT Threads ===\n")

    try:
        # Utwórz RT threads dla różnych komponentów
        camera_thread = RealtimeThread(
            target=camera_capture_loop,
            priority=RealtimeScheduler.PRIORITY_CAMERA_CAPTURE,
            policy=SCHED_FIFO,
            cpu_affinity=[2],  # CPU 2
            name="CameraCapture"
        )

        network_thread = RealtimeThread(
            target=network_send_loop,
            priority=RealtimeScheduler.PRIORITY_NETWORK_SEND,
            policy=SCHED_FIFO,
            cpu_affinity=[1],  # CPU 1
            name="NetworkSend"
        )

        camera_thread.start()
        network_thread.start()

        camera_thread.join()
        network_thread.join()

        print("\nRT Threads completed successfully")

    except PermissionError:
        print("\n⚠️  Need root privileges for RT scheduling!")
        print("Run with: sudo python3 realtime_scheduler.py")
        print("Or configure limits.conf as shown below:\n")
        setup_rt_limits()

    print("\n" + "="*50)
    configure_rt_kernel_params()
    print("\n" + "="*50)
    setup_rt_limits()