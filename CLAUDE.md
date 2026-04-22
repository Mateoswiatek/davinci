Hejka, w tym folderze są różnego rodzaju pliki powiązane z projektem DaVinciV2.
Ogólnie chodzi o to, że kod jest uruchamiany na RaspberryPi, które strumieniuje obraz na okulary Occulus.
W drugą stronę odbywa się wysyłanie kątów głowy.
RaspberryPi jest podłączona do trzech serv, które sterują kamerami. Ogólnie chodzi o stworzenie takiej "zdalnej głowy".
Operator obraca się o 15% w lewo, ta inforamcja jest przesyłana do raspberryPi, obraca servami kamerę również o 15%. tak aby pozycja kamer zgadzała 
się z pozycją operatora. Cały czas RaspberryPi wysyła obraz do operatora.

Ogónie, w /computing jest nowsze podejście, bardziej ogarnięte i je głównie chciałbym rozwijać.
Chciałbym uprościć, powywalać niepotrzebne pliki i foldery, tak aby zostało tylko to co działa i jest używane.
AKtualnie zależy mi tylko na streamingu obrazu, bez rzeczy związanych z YOLO (Bo i tak się wysypuje / w przyszłości będą zmiany).
Ewentualnie możesz utworzyć nowy folder w którym będą sprawdzone rzeczy, które faktycznie działają dobrze.

Brakuje mi opisu i ogólnego pojęcia co tu dokładnie jest, jakie są funkcjonalności i co działa prawidłowo a co nie, co można na spokojnie wyjebać a co powinno zostać.
Na pewno chciałbym, aby został plik /mnt/adata-disk/projects/agh/davinci/davinci/raspberrypi.py - bo to jest wersja, która jako tako działa (obraz, serva nie były w 100% sprawdzane), ręcznie pisana
Oraz odpowiadający mu /mnt/adata-disk/projects/agh/davinci/davinci/backend/main.py - odbieranie testowe.
Jeśli chodzi o kierunek rozwoju, to raczej w stronę tego co jest w computing.
Przejrzyjmy razem co tu wgl jest w tym repozytorium kurwa mać, posprzątajmy, zróbmy jedno konkretne Readme, ale bez jakiś pixel artów, bez zbędnego pierdolenia i opisywania,
tylko faktycznie to co jest, instrukcje krok po kroku (instrukcje i obszary mogą być w innych plikach, ale powinny być linkowane w głównym readme).
Jeśli chodzi o testowanie to będziemy razem sprawdzać na bieżaco co działa / nie działa. bo mam dostęp do raspberryPi fizycznie i automatycznie mam skonfigurowane w Pycharmie
przesyłanie zmian.