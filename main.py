from picamera2 import Picamera2
from PIL import Image

# Inicjalizacja kamery
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (2560, 800)})
picam2.configure(config)
picam2.start()

# Zrób zdjęcie
picam2.capture_file("stereo_full.jpg")
picam2.stop()

print("Zapisano: stereo_full.jpg")