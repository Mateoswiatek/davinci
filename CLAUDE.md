Hejka, w tym folderze są różnego rodzaju pliki powiązane z projektem DaVinciV2.
Ogólnie chodzi o to, że kod jest uruchamiany na RaspberryPi, które strumieniuje obraz na okulary Occulus.
W drugą stronę odbywa się wysyłanie kątów głowy.
RaspberryPi jest podłączona do trzech serv, które sterują kamerami. Ogólnie chodzi o stworzenie takiej "zdalnej głowy".
Operator obraca się o 15% w lewo, ta inforamcja jest przesyłana do raspberryPi, obraca servami kamerę również o 15%. tak aby pozycja kamer zgadzała 
się z pozycją operatora. Cały czas RaspberryPi wysyła obraz do operatora.