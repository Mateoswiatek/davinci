#!/usr/bin/env python3
"""
Test servo - ciągły obrót od jednego ograniczenia do drugiego
Servo podłączone do pinu 14
"""

import time
from gpiozero import AngularServo
import signal
import sys

class ServoTester:
    def __init__(self, pin=14):
        self.pin = pin
        self.servo = None
        self.running = True

        # Ustawienia servo - fizyczne ograniczenia
        self.min_angle = 0      # Minimalna pozycja
        self.max_angle = 180    # Maksymalna pozycja
        self.step = 2           # Krok co ile stopni
        self.delay = 0.05       # Opóźnienie między krokami (sekundy)

        # Kierunek ruchu
        self.direction = 1  # 1 = w górę, -1 = w dół
        self.current_angle = self.min_angle

    def initialize_servo(self):
        """Inicjalizacja servo"""
        try:
            print(f"🔧 Inicjalizacja servo na pinie {self.pin}")

            # Ustawienia impulsu dla standardowego servo
            min_pulse_width = 1.0/1000  # 1ms
            max_pulse_width = 2.0/1000  # 2ms

            self.servo = AngularServo(
                self.pin,
                min_angle=self.min_angle,
                max_angle=self.max_angle,
                min_pulse_width=min_pulse_width,
                max_pulse_width=max_pulse_width,
                initial_angle=self.min_angle
            )

            print(f"✅ Servo zainicjalizowane pomyślnie")
            print(f"📊 Zakres: {self.min_angle}° - {self.max_angle}°")
            print(f"⚙️  Krok: {self.step}°, Opóźnienie: {self.delay}s")
            return True

        except Exception as e:
            print(f"❌ Błąd inicjalizacji servo: {e}")
            return False

    def move_to_angle(self, angle):
        """Przesuń servo do określonej pozycji"""
        try:
            self.servo.angle = angle
            self.current_angle = angle
            return True
        except Exception as e:
            print(f"❌ Błąd przesuwania servo do {angle}°: {e}")
            return False

    def run_continuous_test(self):
        """Uruchom ciągły test ruchu servo"""
        if not self.initialize_servo():
            return

        print("\n🚀 Rozpoczynanie ciągłego testu servo")
        print("🛑 Aby zatrzymać, naciśnij Ctrl+C")
        print("=" * 50)

        cycle_count = 0
        start_time = time.time()

        try:
            while self.running:
                # Oblicz następną pozycję
                next_angle = self.current_angle + (self.step * self.direction)

                # Sprawdź ograniczenia i zmień kierunek jeśli potrzeba
                if next_angle >= self.max_angle:
                    next_angle = self.max_angle
                    self.direction = -1  # Zmień kierunek na w dół
                    cycle_count += 0.5
                    print(f"🔄 Osiągnięto maksimum ({self.max_angle}°) - zmiana kierunku")

                elif next_angle <= self.min_angle:
                    next_angle = self.min_angle
                    self.direction = 1   # Zmień kierunek na w górę
                    cycle_count += 0.5
                    elapsed_time = time.time() - start_time
                    print(f"🔄 Osiągnięto minimum ({self.min_angle}°) - zmiana kierunku")
                    print(f"📈 Cykl #{int(cycle_count)} zakończony (czas: {elapsed_time:.1f}s)")

                # Przesuń servo
                if self.move_to_angle(next_angle):
                    # Pokaż postęp co 10 stopni
                    if next_angle % 10 == 0:
                        direction_arrow = "↗️" if self.direction == 1 else "↘️"
                        print(f"{direction_arrow} Pozycja: {next_angle:3.0f}° | Kierunek: {'w górę' if self.direction == 1 else 'w dół'}")

                # Czekaj przed następnym krokiem
                time.sleep(self.delay)

        except KeyboardInterrupt:
            print(f"\n🛑 Test zatrzymany przez użytkownika")

        finally:
            self.cleanup()

    def cleanup(self):
        """Oczyszczenie zasobów"""
        if self.servo:
            try:
                print(f"\n🔧 Powrót do pozycji centralnej...")
                center_angle = (self.min_angle + self.max_angle) / 2
                self.servo.angle = center_angle
                time.sleep(1)

                print(f"🧹 Zamykanie servo...")
                self.servo.close()
                print(f"✅ Servo zamknięte pomyślnie")

            except Exception as e:
                print(f"❌ Błąd podczas zamykania servo: {e}")

    def run_single_sweep(self):
        """Uruchom pojedynczy przejazd od min do max i z powrotem"""
        if not self.initialize_servo():
            return

        print("\n🔄 Pojedynczy przejazd servo")
        print("=" * 30)

        try:
            # Od min do max
            print(f"➡️  Ruch od {self.min_angle}° do {self.max_angle}°")
            for angle in range(self.min_angle, self.max_angle + 1, self.step):
                self.move_to_angle(angle)
                print(f"📍 Pozycja: {angle}°")
                time.sleep(self.delay)

            print(f"✅ Osiągnięto maksimum: {self.max_angle}°")
            time.sleep(0.5)

            # Od max do min
            print(f"⬅️  Ruch od {self.max_angle}° do {self.min_angle}°")
            for angle in range(self.max_angle, self.min_angle - 1, -self.step):
                self.move_to_angle(angle)
                print(f"📍 Pozycja: {angle}°")
                time.sleep(self.delay)

            print(f"✅ Osiągnięto minimum: {self.min_angle}°")

        except KeyboardInterrupt:
            print(f"\n🛑 Test zatrzymany przez użytkownika")
        finally:
            self.cleanup()

def signal_handler(signum, frame):
    """Obsługa sygnału przerwania (Ctrl+C)"""
    print(f"\n🛑 Otrzymano sygnał przerwania, zatrzymywanie...")
    sys.exit(0)

def main():
    # Konfiguracja
    SERVO_PIN = 14

    print("🤖 Test servo - ciągły obrót")
    print("=" * 40)
    print(f"📌 Pin servo: {SERVO_PIN}")
    print("📋 Opcje:")
    print("  1 - Ciągły test (w pętli nieskończonej)")
    print("  2 - Pojedynczy przejazd (min→max→min)")
    print("  3 - Konfiguracja zaawansowana")

    # Obsługa Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Tworzenie testera servo
    tester = ServoTester(pin=SERVO_PIN)

    try:
        choice = input("\n🎯 Wybierz opcję (1-3): ").strip()

        if choice == "1":
            tester.run_continuous_test()

        elif choice == "2":
            tester.run_single_sweep()

        elif choice == "3":
            print("\n⚙️  Konfiguracja zaawansowana:")

            try:
                new_min = int(input(f"Minimalny kąt (aktualnie {tester.min_angle}°): ") or tester.min_angle)
                new_max = int(input(f"Maksymalny kąt (aktualnie {tester.max_angle}°): ") or tester.max_angle)
                new_step = int(input(f"Krok (aktualnie {tester.step}°): ") or tester.step)
                new_delay = float(input(f"Opóźnienie (aktualnie {tester.delay}s): ") or tester.delay)

                tester.min_angle = new_min
                tester.max_angle = new_max
                tester.step = new_step
                tester.delay = new_delay
                tester.current_angle = new_min

                print(f"\n✅ Nowa konfiguracja:")
                print(f"   Zakres: {new_min}° - {new_max}°")
                print(f"   Krok: {new_step}°, Opóźnienie: {new_delay}s")

                subchoice = input("\nUruchomić test? (1=ciągły, 2=pojedynczy): ").strip()
                if subchoice == "1":
                    tester.run_continuous_test()
                elif subchoice == "2":
                    tester.run_single_sweep()

            except ValueError:
                print("❌ Nieprawidłowa wartość, używam domyślnych ustawień")
                tester.run_continuous_test()
        else:
            print("❌ Nieprawidłowy wybór, uruchamiam test ciągły")
            tester.run_continuous_test()

    except KeyboardInterrupt:
        print(f"\n👋 Program zakończony przez użytkownika")
    except Exception as e:
        print(f"❌ Błąd programu: {e}")

if __name__ == "__main__":
    main()