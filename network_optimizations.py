#!/usr/bin/env python3
"""
Network optimizations dla ultra-niskiej latencji streaming
- UDP vs TCP configuration
- Socket buffer tuning
- TCP_NODELAY, TCP_CORK
- Interrupt coalescing
- Network namespaces
"""

import socket
import struct
import os
import subprocess
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class NetworkStats:
    """Statystyki sieciowe"""
    interface: str
    rx_packets: int
    tx_packets: int
    rx_bytes: int
    tx_bytes: int
    rx_dropped: int
    tx_dropped: int


class LowLatencySocket:
    """Socket zoptymalizowany pod ultra-niską latencję"""

    def __init__(self, use_udp: bool = True, port: int = 8554):
        """
        Args:
            use_udp: True = UDP (niższa latencja), False = TCP
            port: Port do nasłuchiwania/połączenia
        """
        self.use_udp = use_udp
        self.port = port

        if use_udp:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self._configure_socket()

    def _configure_socket(self):
        """Konfiguruj socket dla minimalnej latencji"""

        print(f"\n=== Configuring {'UDP' if self.use_udp else 'TCP'} Socket ===")

        # 1. Socket buffer sizes
        # Większe bufory = mniej syscalls, ale więcej latencji
        # Dla VR: małe bufory, częste wysyłanie
        SO_RCVBUF = socket.SO_RCVBUF
        SO_SNDBUF = socket.SO_SNDBUF

        # Dla ultra-low latency: małe bufory (256KB)
        # Dla wysokiego throughput: duże bufory (4MB+)
        send_buf_size = 256 * 1024  # 256 KB
        recv_buf_size = 256 * 1024

        self.sock.setsockopt(socket.SOL_SOCKET, SO_SNDBUF, send_buf_size)
        self.sock.setsockopt(socket.SOL_SOCKET, SO_RCVBUF, recv_buf_size)
        print(f"✓ Buffer sizes: send={send_buf_size/1024:.0f}KB, recv={recv_buf_size/1024:.0f}KB")

        # 2. Reuse address (szybszy restart)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print(f"✓ SO_REUSEADDR enabled")

        # 3. TCP-specific optimizations
        if not self.use_udp:
            # TCP_NODELAY - wyłącz Nagle's algorithm (natychmiast wysyłaj)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"✓ TCP_NODELAY enabled (no buffering)")

            # TCP_QUICKACK - wyłącz delayed ACK
            try:
                TCP_QUICKACK = 12  # Linux constant
                self.sock.setsockopt(socket.IPPROTO_TCP, TCP_QUICKACK, 1)
                print(f"✓ TCP_QUICKACK enabled")
            except (OSError, AttributeError):
                print(f"⚠ TCP_QUICKACK not available")

            # TCP_CORK - przeciwieństwo TCP_NODELAY, użyj tylko dla batch sends
            # self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CORK, 0)

        # 4. UDP-specific optimizations
        else:
            # Ustaw TTL
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TTL, 64)

            # Dla multicast (jeśli używane)
            # self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

        # 5. Set non-blocking mode (use with select/epoll)
        # self.sock.setblocking(False)

        # 6. Socket priority (Linux QoS)
        try:
            # Priority 0-7 (7 = highest)
            SO_PRIORITY = 12
            self.sock.setsockopt(socket.SOL_SOCKET, SO_PRIORITY, 6)
            print(f"✓ Socket priority set to 6")
        except (OSError, AttributeError):
            print(f"⚠ SO_PRIORITY not available")

        # 7. TOS (Type of Service) / DSCP dla QoS
        try:
            # DSCP EF (Expedited Forwarding) = 46 << 2 = 184
            # Dla real-time traffic
            dscp_ef = 46 << 2
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, dscp_ef)
            print(f"✓ IP_TOS set to {dscp_ef} (DSCP EF)")
        except (OSError, AttributeError):
            print(f"⚠ IP_TOS not available")

    def bind(self, host: str = '0.0.0.0'):
        """Bind socket"""
        self.sock.bind((host, self.port))
        print(f"✓ Bound to {host}:{self.port}")

    def connect(self, host: str, port: Optional[int] = None):
        """Connect to remote (TCP)"""
        if port is None:
            port = self.port
        self.sock.connect((host, port))
        print(f"✓ Connected to {host}:{port}")

    def send(self, data: bytes, addr: Optional[Tuple[str, int]] = None):
        """Send data"""
        if self.use_udp and addr:
            return self.sock.sendto(data, addr)
        else:
            return self.sock.send(data)

    def recv(self, bufsize: int = 65536):
        """Receive data"""
        if self.use_udp:
            return self.sock.recvfrom(bufsize)
        else:
            return self.sock.recv(bufsize), None

    def close(self):
        """Close socket"""
        self.sock.close()


class NetworkTuner:
    """Tuning parametrów sieciowych na poziomie systemu"""

    @staticmethod
    def get_network_stats(interface: str = 'eth0') -> NetworkStats:
        """Pobierz statystyki interfejsu"""
        try:
            with open(f'/sys/class/net/{interface}/statistics/rx_packets') as f:
                rx_packets = int(f.read().strip())
            with open(f'/sys/class/net/{interface}/statistics/tx_packets') as f:
                tx_packets = int(f.read().strip())
            with open(f'/sys/class/net/{interface}/statistics/rx_bytes') as f:
                rx_bytes = int(f.read().strip())
            with open(f'/sys/class/net/{interface}/statistics/tx_bytes') as f:
                tx_bytes = int(f.read().strip())
            with open(f'/sys/class/net/{interface}/statistics/rx_dropped') as f:
                rx_dropped = int(f.read().strip())
            with open(f'/sys/class/net/{interface}/statistics/tx_dropped') as f:
                tx_dropped = int(f.read().strip())

            return NetworkStats(
                interface=interface,
                rx_packets=rx_packets,
                tx_packets=tx_packets,
                rx_bytes=rx_bytes,
                tx_bytes=tx_bytes,
                rx_dropped=rx_dropped,
                tx_dropped=tx_dropped
            )
        except FileNotFoundError:
            print(f"Interface {interface} not found")
            return None

    @staticmethod
    def configure_kernel_network_params():
        """Konfiguruj parametry kernela dla niskiej latencji"""

        print("\n=== Network Kernel Parameters ===\n")

        params = {
            # TCP tuning
            "net.ipv4.tcp_low_latency": "1",
            "net.ipv4.tcp_sack": "1",
            "net.ipv4.tcp_timestamps": "1",
            "net.ipv4.tcp_window_scaling": "1",

            # Disable slow start after idle
            "net.ipv4.tcp_slow_start_after_idle": "0",

            # TCP buffer sizes (min, default, max)
            "net.ipv4.tcp_rmem": "4096 87380 16777216",  # 16MB max
            "net.ipv4.tcp_wmem": "4096 65536 16777216",

            # UDP buffer sizes
            "net.core.rmem_max": "16777216",  # 16MB
            "net.core.wmem_max": "16777216",
            "net.core.rmem_default": "262144",  # 256KB
            "net.core.wmem_default": "262144",

            # Network device queue
            "net.core.netdev_max_backlog": "5000",

            # Optymalizacja dla WiFi
            "net.core.netdev_budget": "600",
            "net.core.netdev_budget_usecs": "8000",

            # TCP congestion control (BBR dla lepszej latencji)
            "net.ipv4.tcp_congestion_control": "bbr",
            "net.core.default_qdisc": "fq",  # Fair Queue dla BBR

            # Fast socket opening
            "net.ipv4.tcp_fastopen": "3",

            # Reduce FIN timeout
            "net.ipv4.tcp_fin_timeout": "15",

            # Keepalive
            "net.ipv4.tcp_keepalive_time": "300",
            "net.ipv4.tcp_keepalive_probes": "5",
            "net.ipv4.tcp_keepalive_intvl": "15",
        }

        print("Add to /etc/sysctl.conf:\n")
        for param, value in params.items():
            print(f"  {param} = {value}")

        print("\nApply immediately:")
        print("  sudo sysctl -p")

        print("\n=== Check BBR availability ===")
        print("  cat /proc/sys/net/ipv4/tcp_available_congestion_control")
        print("  # Should include 'bbr'")

    @staticmethod
    def configure_network_interface(interface: str = 'eth0'):
        """Konfiguruj interface dla niskiej latencji"""

        print(f"\n=== Interface {interface} Configuration ===\n")

        commands = [
            # Wyłącz interrupt coalescing (natychmiastowa obsługa pakietów)
            f"sudo ethtool -C {interface} rx-usecs 0 tx-usecs 0",

            # Wyłącz offloading (zwiększa CPU ale zmniejsza latencję)
            f"sudo ethtool -K {interface} gso off",
            f"sudo ethtool -K {interface} tso off",
            f"sudo ethtool -K {interface} gro off",

            # Ring buffer sizes (większy = więcej buffering = większa latencja)
            # Dla low-latency: małe ring buffers
            f"sudo ethtool -G {interface} rx 256 tx 256",

            # Ustaw queue discipline (fq_codel dla low latency)
            f"sudo tc qdisc replace dev {interface} root fq_codel",

            # Wyłącz power management
            f"sudo ethtool -s {interface} wol d",
        ]

        print("Execute these commands:\n")
        for cmd in commands:
            print(f"  {cmd}")

        print("\n=== IRQ Affinity (pin network IRQ to specific CPU) ===")
        print(f"  # Find network IRQ number:")
        print(f"  grep {interface} /proc/interrupts")
        print(f"  # Set affinity to CPU 1:")
        print(f"  echo 2 | sudo tee /proc/irq/[IRQ_NUM]/smp_affinity")

    @staticmethod
    def setup_wifi_low_latency(interface: str = 'wlan0'):
        """Specjalna konfiguracja dla WiFi"""

        print(f"\n=== WiFi Low Latency Config ({interface}) ===\n")

        commands = [
            # Wyłącz power save
            f"sudo iw dev {interface} set power_save off",

            # Użyj 5GHz jeśli dostępne (mniej interference)
            # Ustaw channel width na 80MHz dla większego bandwidth
            # (to musi być zgodne z AP)

            # QoS dla WiFi (WMM)
            # Ustawiane przez network manager lub wpa_supplicant
        ]

        print("Execute:\n")
        for cmd in commands:
            print(f"  {cmd}")

        print("\nAdd to /etc/NetworkManager/conf.d/wifi-powersave.conf:")
        print("  [connection]")
        print("  wifi.powersave = 2")  # 2 = disable

        print("\nRestart NetworkManager:")
        print("  sudo systemctl restart NetworkManager")


class WebRTCOptimizer:
    """Optymalizacje dla WebRTC streaming"""

    @staticmethod
    def get_recommended_config():
        """Zwraca zalecaną konfigurację WebRTC dla low-latency"""

        config = {
            'video': {
                'codec': 'H264',  # lub VP8/VP9
                'profile': 'baseline',  # Niższa latencja niż main/high
                'bitrate': {
                    'min': 500_000,     # 500 kbps
                    'start': 2_000_000,  # 2 Mbps
                    'max': 10_000_000,   # 10 Mbps (dla 5GHz WiFi)
                },
                'framerate': 60,  # Dla VR minimum 60fps
                'keyframe_interval': 2000,  # Co 2s
            },
            'network': {
                'ice_servers': [
                    {'urls': 'stun:stun.l.google.com:19302'}
                ],
                # Preferuj UDP
                'ice_transport_policy': 'all',  # lub 'relay' dla NAT
                'bundle_policy': 'max-bundle',
                'rtcp_mux_policy': 'require',
            },
            'buffers': {
                'jitter_buffer': 0,  # Minimal buffering dla VR
                'playout_delay': {'min': 0, 'max': 100},  # Max 100ms
            }
        }

        return config


def benchmark_network_latency(host: str, port: int = 8554,
                              use_udp: bool = True,
                              num_packets: int = 100):
    """
    Benchmark latencji sieciowej
    """
    import time

    print(f"\n=== Network Latency Benchmark ===")
    print(f"Protocol: {'UDP' if use_udp else 'TCP'}")
    print(f"Target: {host}:{port}")
    print(f"Packets: {num_packets}\n")

    sock = LowLatencySocket(use_udp=use_udp, port=port)

    # Dane testowe (1KB)
    test_data = b'X' * 1024
    latencies = []

    try:
        if use_udp:
            # UDP - bezpośrednie wysyłanie
            for i in range(num_packets):
                start = time.perf_counter()
                sock.send(test_data, (host, port))
                # W prawdziwym benchmarku czekałbyś na odpowiedź
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
        else:
            # TCP - połącz się najpierw
            sock.connect(host, port)
            for i in range(num_packets):
                start = time.perf_counter()
                sock.send(test_data)
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

        # Statystyki
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            print(f"Results:")
            print(f"  Average: {avg_latency:.2f} ms")
            print(f"  Min: {min_latency:.2f} ms")
            print(f"  Max: {max_latency:.2f} ms")

    finally:
        sock.close()


if __name__ == "__main__":
    print("=== Network Optimization Setup ===")

    # Statystyki
    stats = NetworkTuner.get_network_stats('eth0')
    if stats:
        print(f"\n{stats.interface} stats:")
        print(f"  RX: {stats.rx_packets:,} packets ({stats.rx_bytes / 1024 / 1024:.1f} MB)")
        print(f"  TX: {stats.tx_packets:,} packets ({stats.tx_bytes / 1024 / 1024:.1f} MB)")
        print(f"  Dropped: RX={stats.rx_dropped}, TX={stats.tx_dropped}")

    # Konfiguracja
    print("\n" + "="*70)
    NetworkTuner.configure_kernel_network_params()

    print("\n" + "="*70)
    NetworkTuner.configure_network_interface('eth0')

    print("\n" + "="*70)
    NetworkTuner.setup_wifi_low_latency('wlan0')

    # WebRTC config
    print("\n" + "="*70)
    print("=== WebRTC Configuration ===\n")
    config = WebRTCOptimizer.get_recommended_config()
    import json
    print(json.dumps(config, indent=2))

    # Socket demo
    print("\n" + "="*70)
    print("=== Socket Configuration Demo ===")
    udp_sock = LowLatencySocket(use_udp=True, port=8554)
    tcp_sock = LowLatencySocket(use_udp=False, port=8555)

    print("\n=== Summary ===")
    print("For VR streaming:")
    print("  • Use UDP for lowest latency (no retransmits)")
    print("  • Enable TCP_NODELAY if using TCP")
    print("  • Set small socket buffers (256KB)")
    print("  • Disable interrupt coalescing on NIC")
    print("  • Use BBR congestion control")
    print("  • Pin network IRQs to dedicated CPU")
    print("  • Disable WiFi power save")
    print("  • Use 5GHz WiFi for lower latency")