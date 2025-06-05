ssh pi@raspberrypi.local
ip a
ssh pi@192.168.1.26

nmap -sn 192.168.113.209/24




# Komenda odpalająca program na raspberrypi
/home/pi/venv/davinci/bin/python /home/pi/Desktop/project/davinci/raspberrypi.py

# Przewód RJ45
cat /sys/class/net/eth0/carrier
cat /sys/class/net/eth0/speed

# Komenda do wyświetlenia konfiguracji kamery
v4l2-ctl --all -d /dev/video0

