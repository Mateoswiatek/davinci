ssh pi@raspberrypi.local
ip a
ssh pi@192.168.1.26

nmap -sn 192.168.113.209/24




# Komenda odpalająca program na raspberrypi
/home/pi/venv/davinci/bin/python /home/pi/Desktop/project/davinci/raspberrypi.py

# Przewód RJ45
cat /sys/class/net/eth0/carrier
cat /sys/class/net/eth0/speed

# Konfiguracja, czy kamera jest włączona
cat /boot/firmware/config.txt | grep camera

# Logi systemowe dotyczące kamery
dmesg | grep -i camera

# Komenda do wyświetlenia konfiguracji kamery
v4l2-ctl --all -d /dev/video0

# Dostępne kamery
libcamera-hello --list-cameras

# Które /dev/video odpowiada kamerze
for i in {0..7}; do
  echo "=== /dev/video$i ==="
  v4l2-ctl --device=/dev/video$i --list-formats-ext 2>/dev/null | head -10
  echo ""
done

vcgencmd measure_temp
vcgencmd measure_volts



# Spróbuj z mniejszym timeout i prostszymi parametrami
libcamera-still -o test.jpg --timeout 500 --width 640 --height 200
# Lub z natywną rozdzielczością
libcamera-still -o test.jpg --timeout 500 --width 1280 --height 400
# Spróbuj trybu raw
libcamera-still -o test.jpg --timeout 500 --raw


# Test każdego trybu osobno
libcamera-hello --list-cameras
libcamera-still -o test_640x200.jpg --timeout 500 --width 640 --height 200
libcamera-still -o test_1280x400.jpg --timeout 500 --width 1280 --height 400
libcamera-still -o test_2560x720.jpg --timeout 500 --width 2560 --height 720

find /usr/share/libcamera -name "*.json" | grep -i arducam
find /usr/share/libcamera -name "*.json" | grep -i pivariety
ls -la /usr/share/libcamera/ipa/rpi/pisp/

