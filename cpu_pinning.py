#!/usr/bin/env python3
"""
CPU Pinning i Process Isolation dla minimalnej latencji
"""

import os
import psutil
import subprocess
from typing import List


class CPUPinner:
    """Zarządza przypisaniem procesów do konkretnych rdzeni CPU"""

    def __init__(self):
        self.total_cpus = psutil.cpu_count()

    def pin_process(self, pid: int, cpus: List[int]):
        """Przypina proces do wybranych rdzeni"""
        cpu_mask = ','.join(map(str, cpus))
        try:
            subprocess.run(['taskset', '-cp', cpu_mask, str(pid)],
                         check=True, capture_output=True)
            print(f"Process {pid} pinned to CPUs: {cpu_mask}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to pin process: {e.stderr.decode()}")

    def pin_current_process(self, cpus: List[int]):
        """Przypina bieżący proces"""
        pid = os.getpid()
        self.pin_process(pid, cpus)

    def pin_thread(self, thread_id: int, cpus: List[int]):
        """Przypina konkretny wątek"""
        self.pin_process(thread_id, cpus)

    def set_cpu_affinity(self, cpus: List[int]):
        """Alternatywna metoda używająca psutil"""
        p = psutil.Process()
        p.cpu_affinity(cpus)
        print(f"Process {p.pid} affinity set to: {cpus}")

    def get_cpu_stats(self):
        """Wyświetla statystyki użycia CPU"""
        return psutil.cpu_percent(interval=0.1, percpu=True)


class CGroupManager:
    """Zarządza cgroups dla izolacji zasobów"""

    def __init__(self, group_name: str = "vr_streaming"):
        self.group_name = group_name
        self.cgroup_path = f"/sys/fs/cgroup/cpu/{group_name}"

    def create_cgroup(self, cpu_shares: int = 1024,
                      cpuset: str = "2,3"):
        """Tworzy cgroup z określonymi limitami CPU"""

        commands = [
            # Utwórz cgroup dla CPU
            f"sudo cgcreate -g cpu,cpuset:/{self.group_name}",

            # Ustaw CPU shares (priorytet)
            f"sudo cgset -r cpu.shares={cpu_shares} {self.group_name}",

            # Ustaw dostępne CPU
            f"sudo cgset -r cpuset.cpus={cpuset} {self.group_name}",

            # Ustaw memory nodes
            f"sudo cgset -r cpuset.mems=0 {self.group_name}",
        ]

        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=True)
                print(f"Executed: {cmd}")
            except subprocess.CalledProcessError as e:
                print(f"Failed: {cmd}, Error: {e}")

    def add_process_to_cgroup(self, pid: int):
        """Dodaje proces do cgroup"""
        cmd = f"sudo cgclassify -g cpu,cpuset:/{self.group_name} {pid}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"Process {pid} added to cgroup {self.group_name}")

    def run_in_cgroup(self, command: str):
        """Uruchamia komendę w cgroup"""
        cmd = f"sudo cgexec -g cpu,cpuset:/{self.group_name} {command}"
        subprocess.run(cmd, shell=True)


def setup_low_latency_cpu_config():
    """
    Kompletna konfiguracja CPU dla niskiej latencji

    Alokacja rdzeni:
    - CPU 0: System ogólny + network IRQs
    - CPU 1: Networking + encoding
    - CPU 2: Camera capture (izolowany)
    - CPU 3: YOLO detection (izolowany)
    """

    pinner = CPUPinner()

    print("=== CPU Configuration dla VR Streaming ===")
    print(f"Total CPUs: {pinner.total_cpus}")

    # Przykład: Pin głównego procesu do rdzeni 2-3
    print("\n1. Pinning main process to isolated CPUs (2-3)")
    pinner.set_cpu_affinity([2, 3])

    # Wyświetl obecne obciążenie
    print("\n2. Current CPU usage:")
    stats = pinner.get_cpu_stats()
    for i, usage in enumerate(stats):
        print(f"   CPU {i}: {usage}%")

    return pinner


def setup_cgroups_isolation():
    """Konfiguruje cgroups dla różnych komponentów"""

    # Camera capture - CPU 2, wysoki priorytet
    camera_cgroup = CGroupManager("vr_camera")
    camera_cgroup.create_cgroup(cpu_shares=2048, cpuset="2")

    # YOLO detection - CPU 3, średni priorytet
    yolo_cgroup = CGroupManager("vr_yolo")
    yolo_cgroup.create_cgroup(cpu_shares=1024, cpuset="3")

    # Network streaming - CPU 1
    network_cgroup = CGroupManager("vr_network")
    network_cgroup.create_cgroup(cpu_shares=1536, cpuset="1")

    print("\nCGroups created successfully")
    return camera_cgroup, yolo_cgroup, network_cgroup


if __name__ == "__main__":
    # Ustaw CPU affinity dla bieżącego procesu
    pinner = setup_low_latency_cpu_config()

    # Opcjonalnie: konfiguruj cgroups (wymaga sudo)
    # setup_cgroups_isolation()

    print("\n=== Zalecane komendy systemowe ===")
    print("1. Sprawdź isolated CPUs:")
    print("   cat /sys/devices/system/cpu/isolated")

    print("\n2. Wyłącz CPU frequency scaling na izolowanych rdzeniach:")
    print("   sudo cpupower frequency-set -g performance -r 2-3")

    print("\n3. Wyłącz IRQ balancing na izolowanych CPU:")
    print("   sudo systemctl stop irqbalance")

    print("\n4. Sprawdź IRQ affinity:")
    print("   grep . /proc/irq/*/smp_affinity_list")