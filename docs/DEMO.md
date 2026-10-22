# Demo guide (for README / LinkedIn)

## Quick run (no drone)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/smoke_test.py
python launcher.py
```

In the launcher, use **simulation** or run `python main.py --mode gesture` with a webcam.

## Real Tello

1. Connect PC Wi‑Fi to `Tello-XXXXXX` only (no VPN/Ethernet stealing the route).
2. `pip install djitellopy av`
3. `python scripts/verify_tello.py` — must print `OK`.
4. `python main.py --mode gesture --tello --fpv`

See [`TELLO_SETUP_HE.md`](TELLO_SETUP_HE.md) for troubleshooting.

## Suggested screen recording (30–60 s)

1. Launcher → preflight green → start gesture mode (sim or Tello).
2. Show open palm calibration, then 2–3 gestures (POINT forward, OPEN hover, FIST land on ground only).
3. If flying: short clip of FPV + HUD (battery, intent line).
4. End with **L** emergency land or landed state.

**Safety:** fly outdoors or large indoor space; keep people clear; this is a hobby project, not a commercial product.

## Assets folder (optional)

Add screenshots or clips here for the repo (not required for the app to run):

- `assets/launcher.png`
- `assets/gesture-hud.png`
- `assets/demo.mp4` (keep under ~20 MB for GitHub, or host on LinkedIn only)
