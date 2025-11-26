#!/usr/bin/env python3
"""
Real-time monitoring dashboard dla VR streaming system
Wyświetla latencję, CPU usage, network stats w czasie rzeczywistym
"""

import os
import sys
import time
import curses
from collections import deque
from typing import Dict, List, Optional
import psutil
import subprocess


class MonitorDashboard:
    """Real-time monitoring dashboard using curses"""

    def __init__(self, target_latency_ms: float = 30.0):
        self.target_latency_ms = target_latency_ms
        self.latency_history = deque(maxlen=100)
        self.fps_history = deque(maxlen=100)
        self.running = True

    def get_process_stats(self, process_name: str) -> Optional[Dict]:
        """Pobierz statystyki procesu"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent',
                                            'memory_percent', 'num_threads']):
                if process_name.lower() in proc.info['name'].lower():
                    return {
                        'pid': proc.info['pid'],
                        'cpu': proc.info['cpu_percent'],
                        'memory': proc.info['memory_percent'],
                        'threads': proc.info['num_threads'],
                        'cpu_affinity': proc.cpu_affinity() if hasattr(proc, 'cpu_affinity') else []
                    }
        except:
            pass
        return None

    def get_network_stats(self, interface: str = 'wlan0') -> Dict:
        """Pobierz statystyki sieciowe"""
        try:
            with open(f'/sys/class/net/{interface}/statistics/rx_packets') as f:
                rx_packets = int(f.read().strip())
            with open(f'/sys/class/net/{interface}/statistics/tx_packets') as f:
                tx_packets = int(f.read().strip())
            with open(f'/sys/class/net/{interface}/statistics/rx_bytes') as f:
                rx_bytes = int(f.read().strip())
            with open(f'/sys/class/net/{interface}/statistics/tx_bytes') as f:
                tx_bytes = int(f.read().strip())
            with open(f'/sys/class/net/{interface}/statistics/tx_dropped') as f:
                tx_dropped = int(f.read().strip())

            return {
                'rx_packets': rx_packets,
                'tx_packets': tx_packets,
                'rx_mbytes': rx_bytes / 1024 / 1024,
                'tx_mbytes': tx_bytes / 1024 / 1024,
                'tx_dropped': tx_dropped
            }
        except:
            return {}

    def get_cpu_stats(self) -> Dict:
        """Pobierz statystyki CPU"""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)

        # CPU temperature (Raspberry Pi specific)
        try:
            result = subprocess.run(['vcgencmd', 'measure_temp'],
                                  capture_output=True, text=True)
            temp_str = result.stdout.strip()
            temp = float(temp_str.split('=')[1].split("'")[0])
        except:
            temp = 0.0

        return {
            'per_core': cpu_percent,
            'total': sum(cpu_percent) / len(cpu_percent),
            'temperature': temp
        }

    def draw_header(self, stdscr, y: int):
        """Rysuj nagłówek"""
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(y, 0, "="*80)
        stdscr.addstr(y+1, 25, "VR STREAMING - REAL-TIME MONITOR")
        stdscr.addstr(y+2, 0, "="*80)
        stdscr.attroff(curses.color_pair(1))
        return y + 3

    def draw_system_stats(self, stdscr, y: int):
        """Rysuj statystyki systemowe"""
        cpu_stats = self.get_cpu_stats()

        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(y, 0, "SYSTEM:")
        stdscr.attroff(curses.color_pair(2))

        y += 1

        # CPU per-core
        stdscr.addstr(y, 2, "CPU:")
        for i, usage in enumerate(cpu_stats['per_core']):
            color = 1 if usage < 70 else (3 if usage < 90 else 4)
            stdscr.attron(curses.color_pair(color))
            stdscr.addstr(y, 10 + i*10, f"CPU{i}: {usage:5.1f}%")
            stdscr.attroff(curses.color_pair(color))

        y += 1

        # Temperature
        temp = cpu_stats['temperature']
        temp_color = 1 if temp < 70 else (3 if temp < 80 else 4)
        stdscr.attron(curses.color_pair(temp_color))
        stdscr.addstr(y, 2, f"Temperature: {temp:.1f}°C")
        stdscr.attroff(curses.color_pair(temp_color))

        y += 1

        # Memory
        mem = psutil.virtual_memory()
        mem_color = 1 if mem.percent < 70 else (3 if mem.percent < 90 else 4)
        stdscr.attron(curses.color_pair(mem_color))
        stdscr.addstr(y, 2, f"Memory: {mem.used/1024/1024:.0f}MB / "
                           f"{mem.total/1024/1024:.0f}MB ({mem.percent:.1f}%)")
        stdscr.attroff(curses.color_pair(mem_color))

        return y + 2

    def draw_process_stats(self, stdscr, y: int):
        """Rysuj statystyki procesów"""
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(y, 0, "PROCESSES:")
        stdscr.attroff(curses.color_pair(2))

        y += 1

        processes = {
            'Camera': 'CameraProcess',
            'YOLO': 'YOLOProcess',
            'Network': 'NetworkProcess'
        }

        for name, proc_name in processes.items():
            stats = self.get_process_stats(proc_name)

            if stats:
                cpu_color = 1 if stats['cpu'] < 70 else (3 if stats['cpu'] < 90 else 4)

                stdscr.addstr(y, 2, f"{name:8s}:")
                stdscr.attron(curses.color_pair(cpu_color))
                stdscr.addstr(y, 12, f"PID {stats['pid']:5d} | "
                                    f"CPU {stats['cpu']:5.1f}% | "
                                    f"MEM {stats['memory']:4.1f}% | "
                                    f"Threads {stats['threads']:2d}")
                stdscr.attroff(curses.color_pair(cpu_color))

                if stats['cpu_affinity']:
                    stdscr.addstr(y, 60, f"CPUs: {stats['cpu_affinity']}")
            else:
                stdscr.attron(curses.color_pair(4))
                stdscr.addstr(y, 2, f"{name:8s}: NOT RUNNING")
                stdscr.attroff(curses.color_pair(4))

            y += 1

        return y + 1

    def draw_network_stats(self, stdscr, y: int):
        """Rysuj statystyki sieciowe"""
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(y, 0, "NETWORK:")
        stdscr.attroff(curses.color_pair(2))

        y += 1

        # Sprawdź dostępne interfejsy
        interfaces = ['eth0', 'wlan0']
        found = False

        for iface in interfaces:
            stats = self.get_network_stats(iface)
            if stats:
                found = True
                stdscr.addstr(y, 2, f"{iface}:")
                stdscr.addstr(y, 12, f"RX: {stats['rx_packets']:8d} pkts "
                                    f"({stats['rx_mbytes']:8.1f} MB)")
                y += 1
                stdscr.addstr(y, 12, f"TX: {stats['tx_packets']:8d} pkts "
                                    f"({stats['tx_mbytes']:8.1f} MB)")

                if stats['tx_dropped'] > 0:
                    stdscr.attron(curses.color_pair(4))
                    stdscr.addstr(y, 60, f"Dropped: {stats['tx_dropped']}")
                    stdscr.attroff(curses.color_pair(4))

                y += 1

        if not found:
            stdscr.addstr(y, 2, "No network interface found")
            y += 1

        return y + 1

    def draw_latency_stats(self, stdscr, y: int):
        """Rysuj statystyki latencji"""
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(y, 0, "LATENCY:")
        stdscr.attroff(curses.color_pair(2))

        y += 1

        if self.latency_history:
            avg_latency = sum(self.latency_history) / len(self.latency_history)
            min_latency = min(self.latency_history)
            max_latency = max(self.latency_history)

            # Color based on target
            lat_color = 1 if avg_latency < self.target_latency_ms else 4

            stdscr.attron(curses.color_pair(lat_color))
            stdscr.addstr(y, 2, f"Current: {avg_latency:6.2f} ms (target: <{self.target_latency_ms:.0f} ms)")
            stdscr.attroff(curses.color_pair(lat_color))

            y += 1
            stdscr.addstr(y, 2, f"Min: {min_latency:6.2f} ms  |  Max: {max_latency:6.2f} ms")

            # ASCII graph
            y += 2
            stdscr.addstr(y, 2, "Latency Graph (last 50 frames):")
            y += 1

            graph_data = list(self.latency_history)[-50:]
            if graph_data:
                max_val = max(max(graph_data), self.target_latency_ms)
                graph_height = 10

                for row in range(graph_height):
                    threshold = max_val * (1 - row / graph_height)
                    line = f"{threshold:5.1f} ms |"

                    for val in graph_data:
                        if val >= threshold:
                            char = '█'
                            color = 1 if val < self.target_latency_ms else 4
                        else:
                            char = ' '
                            color = 1

                        line += char

                    stdscr.attron(curses.color_pair(color))
                    stdscr.addstr(y + row, 2, line)
                    stdscr.attroff(curses.color_pair(color))

                y += graph_height
        else:
            stdscr.addstr(y, 2, "No latency data yet...")
            y += 1

        return y + 1

    def draw_footer(self, stdscr, y: int):
        """Rysuj stopkę"""
        max_y, max_x = stdscr.getmaxyx()
        footer_y = max_y - 2

        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(footer_y, 0, "="*80)
        stdscr.addstr(footer_y + 1, 2, "Press 'q' to quit  |  'r' to reset stats  |  Refresh: 1s")
        stdscr.attroff(curses.color_pair(1))

    def update_latency(self, latency_ms: float):
        """Update latency data"""
        self.latency_history.append(latency_ms)

    def run(self, stdscr):
        """Main display loop"""

        # Setup colors
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)

        # Non-blocking input
        stdscr.nodelay(True)
        curses.curs_set(0)  # Hide cursor

        last_update = time.time()

        while self.running:
            try:
                stdscr.clear()

                y = 0

                # Draw sections
                y = self.draw_header(stdscr, y)
                y = self.draw_system_stats(stdscr, y)
                y = self.draw_process_stats(stdscr, y)
                y = self.draw_network_stats(stdscr, y)
                y = self.draw_latency_stats(stdscr, y)
                self.draw_footer(stdscr, y)

                stdscr.refresh()

                # Check for input
                try:
                    key = stdscr.getch()
                    if key == ord('q'):
                        self.running = False
                    elif key == ord('r'):
                        self.latency_history.clear()
                        self.fps_history.clear()
                except:
                    pass

                # Simulate latency data (w prawdziwym kodzie pobierz z profiler)
                # TODO: Read from latency_profiler shared memory or file
                if time.time() - last_update > 1.0:
                    # Fake data dla demo
                    import random
                    fake_latency = random.uniform(15.0, 35.0)
                    self.update_latency(fake_latency)
                    last_update = time.time()

                time.sleep(0.1)  # 10Hz refresh

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                # Error handling (w przypadku resize terminala itp.)
                pass


def main():
    """Main entry point"""

    import argparse

    parser = argparse.ArgumentParser(description='VR Streaming Monitor Dashboard')
    parser.add_argument('--target', type=float, default=30.0,
                       help='Target latency in ms (default: 30.0)')

    args = parser.parse_args()

    dashboard = MonitorDashboard(target_latency_ms=args.target)

    try:
        curses.wrapper(dashboard.run)
    except KeyboardInterrupt:
        pass

    print("\nMonitor stopped.")


if __name__ == "__main__":
    main()