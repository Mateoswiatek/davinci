`/home/pi/venv/davinci/bin/python3 /home/pi/Desktop/project/davinci/new/run_server.py`

# Na RPi — generuj self-signed cert Serwer musi wystartować z SSL (aiohttp to obsługuje natywnie)
mkdir -p ~/certs                                                                                                                                                                                          
openssl req -x509 -newkey rsa:2048 -keyout ~/certs/key.pem -out ~/certs/cert.pem \                                                                                                                        
-days 365 -nodes -subj "/CN=raspberrypi.local"

Później ustawiamy w run_server.py 
```
ssl_certfile="/home/pi/certs/cert.pem",
ssl_keyfile="/home/pi/certs/key.pem",
```

Potem w przeglądarce Oculus wchodzisz raz na https://IP_RPI:8000, akceptujesz ostrzeżenie o certyfikacie — i od tej pory działa. To jednorazowe.


Na Occulusie pobieramy Aurora Store, przez file(s) wchodzimy, instalujemy i później instalujemy tailscale
Po zalogowaniu się (przypisaniu Occulusa do konta w tailscale, tego samego do którego przypisana jest raspberka)
możemy zobaczyć wizję z kamer.
