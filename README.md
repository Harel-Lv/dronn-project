# Drone Gesture Control

A computer vision project that detects hand gestures from a webcam
and maps them into movement commands for future drone control.

## Stage 1
- Webcam input
- Hand landmark detection
- Gesture recognition
- Command output

## DJI Tello (רחפן אמיתי)

1. חבר את המחשב ל־Wi‑Fi **`Tello-XXXXXX`**.
2. `pip install djitellopy av`
3. הרצה: `python main.py --mode gesture --tello`  
   עם מצלמת רחפן: הוסף `--fpv`

מדריך מפורט בעברית: [`docs/TELLO_SETUP_HE.md`](docs/TELLO_SETUP_HE.md)  
שליטה במקלדת (מצב `manual`): [`docs/MANUAL_KEYBOARD_HE.md`](docs/MANUAL_KEYBOARD_HE.md)

## Future stages
- Command smoothing
- Keyboard simulation