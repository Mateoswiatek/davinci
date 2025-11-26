#!/bin/bash
#
# Setup script dla ultra-low-latency VR streaming na Raspberry Pi 5
# Konfiguruje system dla <30ms latencji
#
# Usage: sudo ./setup_low_latency_system.sh
#

set -e

COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_NC='\033[0m' # No Color

echo -e "${COLOR_GREEN}================================${COLOR_NC}"
echo -e "${COLOR_GREEN}Ultra-Low-Latency System Setup${COLOR_NC}"
echo -e "${COLOR_GREEN}Raspberry Pi 5 VR Streaming${COLOR_NC}"
echo -e "${COLOR_GREEN}================================${COLOR_NC}"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${COLOR_RED}Error: This script must be run as root${COLOR_NC}"
   echo "Usage: sudo $0"
   exit 1
fi

# ============================================================================
# 1. KERNEL PARAMETERS
# ============================================================================

echo -e "\n${COLOR_YELLOW}[1/10] Configuring kernel parameters...${COLOR_NC}"

# Backup cmdline.txt
if [ ! -f /boot/firmware/cmdline.txt.backup ]; then
    cp /boot/firmware/cmdline.txt /boot/firmware/cmdline.txt.backup
    echo "✓ Backed up cmdline.txt"
fi

# Add isolcpus and RT parameters
CMDLINE=$(cat /boot/firmware/cmdline.txt)

# Remove existing isolcpus if present
CMDLINE=$(echo "$CMDLINE" | sed 's/isolcpus=[^ ]*//g')
CMDLINE=$(echo "$CMDLINE" | sed 's/nohz_full=[^ ]*//g')
CMDLINE=$(echo "$CMDLINE" | sed 's/rcu_nocbs=[^ ]*//g')

# Add new parameters
CMDLINE="$CMDLINE isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3"

echo "$CMDLINE" > /boot/firmware/cmdline.txt
echo "✓ Added isolcpus=2,3 to cmdline.txt"

# ============================================================================
# 2. GPU MEMORY
# ============================================================================

echo -e "\n${COLOR_YELLOW}[2/10] Configuring GPU memory...${COLOR_NC}"

# Backup config.txt
if [ ! -f /boot/firmware/config.txt.backup ]; then
    cp /boot/firmware/config.txt /boot/firmware/config.txt.backup
    echo "✓ Backed up config.txt"
fi

# Set gpu_mem
if grep -q "^gpu_mem=" /boot/firmware/config.txt; then
    sed -i 's/^gpu_mem=.*/gpu_mem=256/' /boot/firmware/config.txt
else
    echo "gpu_mem=256" >> /boot/firmware/config.txt
fi

echo "✓ Set gpu_mem=256 in config.txt"

# ============================================================================
# 3. SYSCTL (VM, NETWORK)
# ============================================================================

echo -e "\n${COLOR_YELLOW}[3/10] Configuring sysctl parameters...${COLOR_NC}"

cat > /etc/sysctl.d/99-vr-lowlatency.conf <<EOF
# VR Streaming Low-Latency Configuration

# ===== MEMORY =====
# Disable swap
vm.swappiness = 0

# Dirty page writeback (fast)
vm.dirty_expire_centisecs = 100
vm.dirty_writeback_centisecs = 100

# Cache pressure
vm.vfs_cache_pressure = 50

# Overcommit
vm.overcommit_memory = 1

# Huge pages
vm.nr_hugepages = 128

# ===== NETWORK =====
# TCP optimization
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_slow_start_after_idle = 0

# TCP buffers
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# UDP buffers
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 262144
net.core.wmem_default = 262144

# Network device queue
net.core.netdev_max_backlog = 5000
net.core.netdev_budget = 600
net.core.netdev_budget_usecs = 8000

# BBR congestion control
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq

# Fast open
net.ipv4.tcp_fastopen = 3

# Timeouts
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 15

# ===== SCHEDULER =====
# RT runtime (95% CPU for RT tasks)
kernel.sched_rt_runtime_us = 950000
kernel.sched_rt_period_us = 1000000

# Scheduler granularity (low latency)
kernel.sched_min_granularity_ns = 1000000
kernel.sched_wakeup_granularity_ns = 500000
EOF

# Apply sysctl
sysctl -p /etc/sysctl.d/99-vr-lowlatency.conf
echo "✓ Applied sysctl parameters"

# ============================================================================
# 4. SECURITY LIMITS (RT PRIORITY)
# ============================================================================

echo -e "\n${COLOR_YELLOW}[4/10] Configuring RT priority limits...${COLOR_NC}"

cat >> /etc/security/limits.conf <<EOF

# VR Streaming - Real-time priorities
pi              -       rtprio          99
pi              -       nice            -20
pi              -       memlock         unlimited

@realtime       -       rtprio          99
@realtime       -       nice            -20
@realtime       -       memlock         unlimited
EOF

echo "✓ Added RT limits to limits.conf"

# ============================================================================
# 5. DISABLE SWAP
# ============================================================================

echo -e "\n${COLOR_YELLOW}[5/10] Disabling swap...${COLOR_NC}"

# Disable swap
swapoff -a

# Comment out swap in fstab
if [ -f /etc/fstab ]; then
    sed -i '/swap/s/^/#/' /etc/fstab
fi

echo "✓ Swap disabled"

# ============================================================================
# 6. DISABLE SERVICES
# ============================================================================

echo -e "\n${COLOR_YELLOW}[6/10] Disabling unnecessary services...${COLOR_NC}"

# Disable IRQ balancing (manual IRQ pinning)
systemctl stop irqbalance 2>/dev/null || true
systemctl disable irqbalance 2>/dev/null || true
echo "✓ Disabled irqbalance"

# Disable Bluetooth (if not needed)
systemctl stop bluetooth 2>/dev/null || true
systemctl disable bluetooth 2>/dev/null || true
echo "✓ Disabled bluetooth"

# ============================================================================
# 7. CPU FREQUENCY SCALING
# ============================================================================

echo -e "\n${COLOR_YELLOW}[7/10] Configuring CPU frequency...${COLOR_NC}"

# Install cpufrequtils
apt-get install -y cpufrequtils > /dev/null 2>&1

# Set performance governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > "$cpu" 2>/dev/null || true
done

# Make permanent
cat > /etc/default/cpufrequtils <<EOF
GOVERNOR="performance"
EOF

echo "✓ Set CPU governor to 'performance'"

# ============================================================================
# 8. NETWORK INTERFACE
# ============================================================================

echo -e "\n${COLOR_YELLOW}[8/10] Configuring network interface...${COLOR_NC}"

# Detect network interface
ETH_IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^eth|^enp' | head -1)
WLAN_IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^wlan|^wlp' | head -1)

if [ -n "$ETH_IFACE" ]; then
    echo "Found ethernet: $ETH_IFACE"

    # Disable offloading (lower latency, higher CPU)
    ethtool -K "$ETH_IFACE" gso off tso off gro off 2>/dev/null || true

    # Set small ring buffers
    ethtool -G "$ETH_IFACE" rx 256 tx 256 2>/dev/null || true

    # Disable interrupt coalescing
    ethtool -C "$ETH_IFACE" rx-usecs 0 tx-usecs 0 2>/dev/null || true

    echo "✓ Configured $ETH_IFACE"
fi

if [ -n "$WLAN_IFACE" ]; then
    echo "Found WiFi: $WLAN_IFACE"

    # Disable power save
    iw dev "$WLAN_IFACE" set power_save off 2>/dev/null || true

    # NetworkManager config
    mkdir -p /etc/NetworkManager/conf.d
    cat > /etc/NetworkManager/conf.d/wifi-powersave.conf <<EOF
[connection]
wifi.powersave = 2
EOF

    echo "✓ Configured $WLAN_IFACE"
fi

# ============================================================================
# 9. TRANSPARENT HUGE PAGES
# ============================================================================

echo -e "\n${COLOR_YELLOW}[9/10] Configuring transparent huge pages...${COLOR_NC}"

# Set to madvise (app-controlled)
echo "madvise" > /sys/kernel/mm/transparent_hugepage/enabled
echo "✓ Set THP to 'madvise'"

# Make permanent via rc.local
cat > /etc/rc.local <<'EOF'
#!/bin/bash
echo "madvise" > /sys/kernel/mm/transparent_hugepage/enabled
exit 0
EOF

chmod +x /etc/rc.local
systemctl enable rc-local 2>/dev/null || true

# ============================================================================
# 10. PYTHON DEPENDENCIES
# ============================================================================

echo -e "\n${COLOR_YELLOW}[10/10] Installing Python dependencies...${COLOR_NC}"

# Install system packages
apt-get update > /dev/null 2>&1
apt-get install -y \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-psutil \
    libcamera-apps \
    linux-tools-generic \
    > /dev/null 2>&1

# Install pip packages
pip3 install --break-system-packages \
    py-spy \
    ultralytics \
    > /dev/null 2>&1 || true

echo "✓ Installed dependencies"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo -e "${COLOR_GREEN}================================${COLOR_NC}"
echo -e "${COLOR_GREEN}Configuration Complete!${COLOR_NC}"
echo -e "${COLOR_GREEN}================================${COLOR_NC}"
echo ""
echo "Changes made:"
echo "  ✓ Isolated CPUs 2-3 (cmdline.txt)"
echo "  ✓ GPU memory = 256 MB"
echo "  ✓ Sysctl optimizations (network, memory, scheduler)"
echo "  ✓ RT priority limits configured"
echo "  ✓ Swap disabled"
echo "  ✓ CPU governor = performance"
echo "  ✓ Network interface optimized"
echo "  ✓ Transparent huge pages = madvise"
echo "  ✓ Python dependencies installed"
echo ""
echo -e "${COLOR_YELLOW}IMPORTANT: Reboot required!${COLOR_NC}"
echo ""
echo "After reboot, verify:"
echo "  1. Isolated CPUs: cat /sys/devices/system/cpu/isolated"
echo "  2. RT limits: ulimit -r"
echo "  3. BBR enabled: sysctl net.ipv4.tcp_congestion_control"
echo "  4. No swap: free -h"
echo ""
echo "To start VR streaming:"
echo "  sudo python3 vr_streaming_optimized.py --ip <OCULUS_IP>"
echo ""

read -p "Reboot now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Rebooting..."
    reboot
fi