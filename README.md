# Jak to zsetupować?

## RASPBERRY

1. Podłącz raspberry do internetu
2. Odpal serwer strumieniujący obraz za pomocą komendy `/home/pi/venv/davinci/bin/python /home/pi/Desktop/project/davinci/raspberrypi.py`
3. Uruchom komendę `ngrok http http://localhost:8765`

Dostaniesz taki response:
```
ngrok                                                                                                                                     (Ctrl+C to quit)
                                                                                                                                                          
🤖 Want to hang with ngrokkers on our new Discord? http://ngrok.com/discord                                                                               
                                                                                                                                                          
Session Status                online                                                                                                                      
Account                       radoslaw.mysliwiec@ivyconsultants.com (Plan: Free)                                                                          
Version                       3.23.1                                                                                                                      
Region                        Europe (eu)                                                                                                                 
Latency                       34ms                                                                                                                        
Web Interface                 http://127.0.0.1:4040                                                                                                       
Forwarding                    https://4b63-46-205-197-171.ngrok-free.app -> http://localhost:8765                                                         
                                                                                                                                                          
Connections                   ttl     opn     rt1     rt5     p50     p90                                                                                 
                              2       0       0.00    0.00    31.52   48.60                                                                               
                                                                                                                                                          
HTTP Requests
```

4. Skopiuj do schowka fragment forwarding url - w tym przypadku to jest `4b63-46-205-197-171.ngrok-free.app`

## TWÓJ KOMPUTER

1. Podłącz Twój komputer do internetu
2. Skopiuj ten kod do pliku `vr.html`

```
<!DOCTYPE html>
<html lang="pl">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1.0" />
    <title>Stereoskopia – Meta Quest 3 (stream + head-tracking)</title>
    <style>
      body { margin: 0; background: #000; color: #fff; font-family: Arial; }
      #vrButton {
        position: absolute; bottom: 24px; left: 50%;
        transform: translateX(-50%);
        padding: 15px 32px; font-size: 18px; border: none;
        border-radius: 6px; background: #007bff; color: #fff;
        cursor: pointer; z-index: 10;
      }
      #vrButton:disabled { background: #666; cursor: not-allowed; }
      canvas { display: block; }
    </style>
  </head>

  <body>
    <button id="vrButton">Enter VR</button>
    <canvas id="canvas"></canvas>

    <script type="module">
      /* ---------------- konfiguracja ---------------- */
      const WS_URL           = "wss://4b63-46-205-197-171.ngrok-free.app";  // <-- RPi IP:port
      const SEND_INTERVAL_MS = 100;                        // co ile wysyłać kąty
      const Y_OFFSET = -0.25;
      const Z_DIST   = -2.0;

      /* ---------------- importy ---------------- */
      import * as THREE from
        "https://cdn.jsdelivr.net/npm/three@0.164.0/build/three.module.js";

      /* ---------------- zmienne globalne ---------------- */
      let scene, camera, renderer, headLocked;
      let leftMesh, rightMesh, leftTex, rightTex;
      let xrSession = null;
      let lastSent  = 0;
      const quat    = new THREE.Quaternion();
      const euler   = new THREE.Euler();

      /* ---------------- WebSocket ---------------- */
      const ws = new WebSocket(WS_URL);

      ws.onopen    = () => console.log("WS connected");
      ws.onclose   = e => console.log("WS closed", e.reason);
      ws.onerror   = e => console.error("WS error", e);

      ws.onmessage = ({ data }) => {
        try {
          const msg = JSON.parse(data);
          if (msg.type !== "camera_frame" || !msg.image) return;

          const img = new Image();
          img.onload = () => {
            const half = img.width / 2, h = img.height;

            const lC = Object.assign(document.createElement("canvas"),
                                      { width: half, height: h });
            const rC = Object.assign(document.createElement("canvas"),
                                      { width: half, height: h });

            lC.getContext("2d")
               .drawImage(img, 0, 0, half, h, 0, 0, half, h);
            rC.getContext("2d")
               .drawImage(img, half, 0, half, h, 0, 0, half, h);

            leftTex.image  = lC;
            rightTex.image = rC;
            leftTex.needsUpdate  = true;
            rightTex.needsUpdate = true;
          };
          img.src = `data:image/jpeg;base64,${msg.image}`;
        } catch (err) { console.error(err); }
      };

      /* ---------------- inicjalizacja sceny ---------------- */
      init();
      function init() {
        scene            = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        camera = new THREE.PerspectiveCamera(
          75, innerWidth / innerHeight, 0.1, 1000
        );
        camera.position.set(0, 1.6, 0);
        scene.add(camera);

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.xr.enabled = true;
        renderer.setPixelRatio(devicePixelRatio);
        renderer.setSize(innerWidth, innerHeight);
        renderer.setClearColor(0x000000, 1);

        headLocked = new THREE.Group();
        camera.add(headLocked);

        const geom = new THREE.PlaneGeometry(3, 2.25);
        const phMat = new THREE.MeshBasicMaterial({
          color: 0x444444, side: THREE.DoubleSide
        });

        leftTex  = new THREE.Texture();
        rightTex = new THREE.Texture();
        leftMesh  = new THREE.Mesh(geom, phMat.clone());
        rightMesh = new THREE.Mesh(geom, phMat.clone());
        leftMesh.layers.set(1);
        rightMesh.layers.set(2);
        headLocked.add(leftMesh, rightMesh);

        setupVRView();

        if ("xr" in navigator) {
          navigator.xr
            .isSessionSupported("immersive-vr")
            .then(ok => vrButton.disabled = !ok);
        } else vrButton.disabled = true;

        vrButton.onclick = startStopXR;
        window.onresize  = () => {
          camera.aspect = innerWidth / innerHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(innerWidth, innerHeight);
        };

        renderer.setAnimationLoop(renderLoop);
      }

      function setupVRView() {
        leftMesh.position.set(0, Y_OFFSET, Z_DIST);
        rightMesh.position.copy(leftMesh.position);
      }

      function setEyeLayers() {
        const rig = renderer.xr.getCamera();
        if (rig.cameras?.length >= 2) {
          rig.cameras[0].layers.set(1);   // lewe
          rig.cameras[1].layers.set(2);   // prawe
        }
      }

      async function startStopXR() {
        if (renderer.xr.isPresenting) { await xrSession.end(); return; }

        xrSession = await navigator.xr.requestSession("immersive-vr", {
          optionalFeatures: ["local-floor", "bounded-floor"]
        });
        await renderer.xr.setSession(xrSession);
        vrButton.textContent = "Exit VR";
        xrSession.addEventListener("end", () => {
          vrButton.textContent = "Enter VR";
        });
      }

      /* ---------------- pętla renderująca ---------------- */
      function renderLoop(_, frame) {
        if (frame && renderer.xr.isPresenting) setEyeLayers();

        sendHeadAngles();     // <- nasz dodatkowy krok
        renderer.render(scene, camera);
      }

      /* ---------------- wysyłanie kątów ---------------- */
      function sendHeadAngles() {
        const now = performance.now();
        if (now - lastSent < SEND_INTERVAL_MS) return;
        if (ws.readyState !== WebSocket.OPEN)      return;

        camera.getWorldQuaternion(quat);
        euler.setFromQuaternion(quat, "YXZ");  // yaw-pitch-roll

        const pitch = THREE.MathUtils.radToDeg(euler.x);
        const yaw   = THREE.MathUtils.radToDeg(euler.y);
        const roll  = THREE.MathUtils.radToDeg(euler.z);

        ws.send(
          JSON.stringify({
            type:  "head_angles",
            pitch: pitch,
            yaw:   yaw,
            roll:  roll,
            ts:    Date.now()
          })
        );
        lastSent = now;
      }
    </script>
  </body>
</html>
```

3. W miejsce WS_URL wklej odpowiednią część URL skopiowaną do schowka wcześniej
4. Zaloguj się i ustaw swój token z ngrok (https://dashboard.ngrok.com/get-started/your-authtoken), użyj komendy `ngrok config add-authtoken $YOUR_AUTHTOKEN`
5. Uruchom serwer http w directory, w którym siedzi plik vr.html za pomocą komendy `python3 -m http.server 8080`
5. Uruchom komendę `ngrok http http://localhost:8080` (UWAGA: ngrok darmowy pozwala na tylko jeden serwer na raz, dlatego polecam założyć dwa konta po prostu)



Dostaniesz taki response:
```
ngrok                                                                                                                                     (Ctrl+C to quit)
                                                                                                                                                          
Take our ngrok in production survey! https://forms.gle/aXiBFWzEA36DudFn6                                                                                  
                                                                                                                                                          
Session Status                online                                                                                                                      
Account                       radek.m2001@gmail.com (Plan: Free)                                                                                          
Version                       3.23.1                                                                                                                      
Region                        Europe (eu)                                                                                                                 
Latency                       33ms                                                                                                                        
Web Interface                 http://127.0.0.1:4040                                                                                                       
Forwarding                    https://68f8-46-205-197-171.ngrok-free.app -> http://localhost:8080                                                         
                                                                                                                                                          
Connections                   ttl     opn     rt1     rt5     p50     p90                                                                                 
                              94      0       0.00    0.00    0.00    0.01                                                                                
                                                                                                                                                          
HTTP Requests                                                                                                                                             
-------------                                                   
```

6. Zwróć uwagę na Forwarding URL (pełne) oraz ścieżkę do pliku vr.html. W tym przypadku będzie to `https://68f8-46-205-197-171.ngrok-free.app/vr.html`.

## META QUEST 3S

1. Wejdź w przeglądarkę domyślną meta questa i po prostu wpisz w wyszukiwarkę ten pełny Forwarding URL. Działa!
