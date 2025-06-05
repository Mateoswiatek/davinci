import cv2
import subprocess
import time

# Ustaw nazwę pliku tymczasowego
filename = "/tmp/capture.jpg"

# Wykonaj libcamera-still w tle (zapis 1 klatki)
subprocess.run(["libcamera-still", "-o", filename, "-t", "1000", "--width", "640", "--height", "480", "--nopreview"])

# Poczekaj na zakończenie zapisu
time.sleep(1)

# Wczytaj obraz do OpenCV
image = cv2.imread(filename)
if image is not None:
    print(image)
    # cv2.imshow("Obraz z kamery", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nie udało się wczytać obrazu")