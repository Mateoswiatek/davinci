#!/usr/bin/env python3

from gpiozero import AngularServo
from time import sleep

pan_servo = AngularServo(14, min_angle=0, max_angle=180, 
                        min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

tilt_servo = AngularServo(15, min_angle=0, max_angle=180, 
                         min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

roll_servo = AngularServo(18, min_angle=0, max_angle=180, 
                         min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

current_pan = 90
current_tilt = 90
current_roll = 90

def move_servos(pan_angle, tilt_angle, roll_angle):
    global current_pan, current_tilt, current_roll
    
    current_pan = pan_angle
    current_tilt = tilt_angle
    current_roll = roll_angle
    
    print(f"Moving to: Pan={pan_angle}°, Tilt={tilt_angle}°, Roll={roll_angle}°")
    
    pan_servo.angle = pan_angle
    tilt_servo.angle = tilt_angle
    roll_servo.angle = roll_angle
    
    sleep(0.8)

positions = {
    "center": (90, 90, 90),
    "left": (0, 90, 90),
    "right": (180, 90, 90),
    "up": (90, 180, 90),
    "down": (90, 0, 90),
    "roll_left": (90, 90, 0),
    "roll_right": (90, 90, 180),
    "look_upper_left": (0, 180, 90),
    "look_upper_right": (180, 180, 90)
}

try:
    print("Three Servo Control (Pan/Tilt/Roll) - Press CTRL+C to exit")
    
    print("\nMoving to center position")
    move_servos(90, 90, 90)
    
    while True:
        print("\nCurrent position: Pan =", current_pan, "°, Tilt =", current_tilt, "°, Roll =", current_roll, "°")
        print("\nCommands:")
        print("  p X    - Set pan angle to X degrees")
        print("  t X    - Set tilt angle to X degrees")
        print("  r X    - Set roll angle to X degrees")
        print("  m X Y Z- Set all servos (pan, tilt, roll)")
        print("  pos    - Show available preset positions") 
        print("  goto X - Go to preset position X")
        print("  scan   - Run a simple scan pattern")
        print("  center - Center all servos")
        print("  q      - Quit")
        
        cmd = input("\nEnter command: ").strip().lower()
        
        if cmd == 'q':
            break
            
        elif cmd == 'center':
            move_servos(90, 90, 90)
            
        elif cmd.startswith('p '):
            try:
                angle = int(cmd.split()[1])
                if 0 <= angle <= 180:
                    move_servos(angle, current_tilt, current_roll)
                else:
                    print("Angle must be between 0 and 180")
            except (ValueError, IndexError):
                print("Invalid format. Use: p X where X is 0-180")
                
        elif cmd.startswith('t '):
            try:
                angle = int(cmd.split()[1])
                if 0 <= angle <= 180:
                    move_servos(current_pan, angle, current_roll)
                else:
                    print("Angle must be between 0 and 180")
            except (ValueError, IndexError):
                print("Invalid format. Use: t X where X is 0-180")
                
        elif cmd.startswith('r '):
            try:
                angle = int(cmd.split()[1])
                if 0 <= angle <= 180:
                    move_servos(current_pan, current_tilt, angle)
                else:
                    print("Angle must be between 0 and 180")
            except (ValueError, IndexError):
                print("Invalid format. Use: r X where X is 0-180")
                
        elif cmd.startswith('m '):
            try:
                parts = cmd.split()
                if len(parts) == 4:
                    pan = int(parts[1])
                    tilt = int(parts[2])
                    roll = int(parts[3])
                    if 0 <= pan <= 180 and 0 <= tilt <= 180 and 0 <= roll <= 180:
                        move_servos(pan, tilt, roll)
                    else:
                        print("Angles must be between 0 and 180")
                else:
                    print("Invalid format. Use: m X Y Z where X, Y, Z are 0-180")
            except ValueError:
                print("Invalid numbers. Use integers between 0 and 180")
                
        elif cmd == 'pos':
            print("\nAvailable positions:")
            for name, (p, t, r) in positions.items():
                print(f"  {name}: Pan={p}°, Tilt={t}°, Roll={r}°")
                
        elif cmd.startswith('goto '):
            pos_name = cmd.split()[1]
            if pos_name in positions:
                pan, tilt, roll = positions[pos_name]
                print(f"Going to position: {pos_name}")
                move_servos(pan, tilt, roll)
            else:
                print(f"Unknown position. Use 'pos' to see available positions.")
                
        elif cmd == 'scan':
            print("Running scan pattern...")
            
            original_pan = current_pan
            original_tilt = current_tilt
            original_roll = current_roll
            
            print("Scanning horizontally...")
            for angle in [0, 45, 90, 135, 180]:
                move_servos(angle, current_tilt, current_roll)
                sleep(0.3)
                
            print("Scanning vertically...")
            for angle in [0, 45, 90, 135, 180]:
                move_servos(current_pan, angle, current_roll)
                sleep(0.3)

            print("Scanning roll...")
            for angle in [0, 45, 90, 135, 180]:
                move_servos(current_pan, current_tilt, angle)
                sleep(0.3)
                
            print("Returning to original position...")
            move_servos(original_pan, original_tilt, original_roll)
            
        else:
            print("Unknown command")

except KeyboardInterrupt:
    print("\nProgram stopped by user")
finally:
    print("\nCentering servos and cleaning up...")
    move_servos(90, 90, 90)
    
    pan_servo.close()
    tilt_servo.close()
    roll_servo.close()
    print("Program terminated") 