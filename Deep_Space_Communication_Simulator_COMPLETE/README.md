# Deep Space Communication Simulator — Complete Build

Python + Dash academic simulator with a NASA-style dark UI.

## Included
- Login, sign-up (first successful sign-up opens the dashboard), logout and password reset.
- PBKDF2 password hashing; legacy SHA-256 password hashes remain verifiable for migration compatibility.
- CelesTrak GP/TLE catalog with local cache and SGP4 propagation (satellite positions are not hard-coded).
- 3D Earth/orbit and ground-station view, 3D planetary view using JPL DE440s via Skyfield when available.
- Link-budget estimates: range, light-time delay, free-space path loss, received power, SNR, BER, packet-loss estimate and throughput.
- Satellite-specific three-step analysis and approximate upcoming station visibility windows (10-degree elevation mask, 6-hour horizon).
- Packet transmission simulation, DTN store-and-forward queue, mission assistant and persistent mission history.
- RandomForest regression trained on analytical link-model samples with an explicit 80/20 train/test split and evaluation metrics.
- Refresh interval set to 5 seconds.

## Important scientific limitations
This is an educational simulator, not flight-certified software. Link budget inputs are configurable example assumptions. The ML dataset is generated from the analytical model rather than real mission telemetry. Pass predictions use a simplified TEME-to-Earth-fixed rotation and coarse 120-second sampling; validate results with a mission-grade tool before operational use. JPL DE440s may need to download on first run; if unavailable, the planetary panel reports its status instead of inventing positions. CelesTrak or network access can be temporarily unavailable; cached records may be used where available.

## Windows / VS Code setup
Open the extracted `Deep_Space_Communication_Simulator_COMPLETE` folder in VS Code. In the terminal run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8051/`. Keep the terminal open while using the app. Press `Ctrl+C` to stop it.

## Data files
JSON caches and history are created in `data/` on first run. The current authentication file is `data/users.json`. Do not use this simple local JSON-backed application for production accounts or expose it publicly without adding secure server-side sessions, CSRF protection, rate limits, HTTPS and a production WSGI server.

## Tests
Run `python -m pytest` after installing requirements. Tests cover pure communication calculations, ML split shape, packet/DTN behavior, password hash verification and SGP4-related helper contracts without needing to start the Dash server.
