# TODO
- Przedłużacz elektryczny
- Router WiFi - tak aby można było łatwiej pracować + testy VR (Potencjalnie również przydatne do testów = jedna konfiguracja, swój sprzęt)
- inna sala - bardzo wolny internet w gniazdkach

Lokalnie - środowisko z zaw-a.

# Konfigurowanie lapka
Laptop połączony przewodowo do internetu, wifi do rasbperryPi
`rfkill list` - blokady - wyłączenie WiFi gdy przewód jest podłączony
`sudo rfkill unblock wifi` - odblokowanie wifi
`ip a` - podłączenia wifi
`sudo ip link set wlp2s0 up` - podniesienie wifi
`nmcli radio wifi on` - wyłączenie automatycznego wyłączania wifi XD
`nmcli device wifi list` - lista wifi
`nmcli device wifi connect "NAZWA_SIECI" password "HASLO"` - podłączenie się
`sudo ip route add 192.168.0.156 dev wlp2s0 metric 50` - dodanie route do raspberki
Po tych powinien zadziałać ping w dwie strony

# Konfiguracja płytki
Powinno działać wszędzie:
`ssh pi@raspberrypi.local`

W labie (ściana):
ssh pi@192.168.0.171
Dlink:
ssh pi@192.168.0.156

/davinci/raspberrypi.py - główny plik

## Dostęp do RaspbeeryPi z zewnątrz - zdalne programowanie
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh

Dostajemy link, w którym musimy się zalogować i przypisać urządzenia "do nas".

admin.tailscale.com

Następnie na drugim urządzeniu (np u nas na laptopie) również musimy zainstalować tailscale
curl -fsSL https://tailscale.com/install.sh | sh

Na kompie dajemy:
sudo tailscale up - zalogowanie kompa do sieci
Na stronie:
https://login.tailscale.com/admin/machines

na stronie dostępne są nasze urządzenia, wraz z adresami ip
