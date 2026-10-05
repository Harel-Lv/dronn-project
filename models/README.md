# מודלים (MediaPipe + InsightFace)

הפרויקט צריך קבצי `.task` של MediaPipe בתיקייה הזו.  
בלי המודלים — מצבי `gesture`, `identity`, `tracking` וכו' ייכשלו בהפעלה.

## קבצים נדרשים

| קובץ | חובה למצב | גודל משוער |
|------|-----------|------------|
| `hand_landmarker.task` | `gesture`, `webcam_faces`, `manual` (מחוות) | ~7.5 MB |
| `face_landmarker.task` | `identity`, `webcam_faces` (fallback ללא InsightFace) | ~3.8 MB |
| `pose_landmarker_lite.task` | `tracking`, `manual` (FOLLOW בלחיצה ארוכה על Tab) | ~5 MB |

> **שם קובץ פנים:** אם יש לך `face_landmarker (2).task` — שנה ל־`face_landmarker.task`  
> (הקוד תומך גם ב־`face_landmarker*.task`, אבל השם הסטנדרטי נקי יותר).

## הורדה (PowerShell)

הרץ מתוך שורש הפרויקט (`dronn_project`):

```powershell
New-Item -ItemType Directory -Force -Path models | Out-Null
cd models

# יד
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "hand_landmarker.task"

# פנים
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "face_landmarker.task"

# pose (lite)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "pose_landmarker_lite.task"

cd ..
```

## זיהוי פנים (ArcFace) — מומלץ

ברירת המחדל ב־`config.yaml` היא `embedding_backend: insightface`.  
זה **לא** קובץ בתיקייה הזו — המודל `buffalo_l` יורד אוטומטית בפעם הראשונה דרך InsightFace.

```powershell
pip install insightface onnxruntime
```

- **GPU:** `pip install onnxruntime-gpu` במקום `onnxruntime` (אם יש CUDA מתאים).
- אחרי התקנת insightface, אם `cv2.imshow` נשבר:
  ```powershell
  pip uninstall opencv-python-headless -y
  pip install -U "opencv-python>=4.8,<5"
  ```

## בדיקה מהירה

```powershell
# מחוות בסימולציה (רק יד)
python main.py --mode gesture

# זיהוי פנים
python main.py --mode identity

# מעקב גוף (סימולציה)
python main.py --mode tracking
```

## Git

קבצי `.task` כבדים. אם לא רוצים אותם ב־GitHub — בטל הערה ב־`.gitignore`:

```gitignore
models/*.task
```

והשאר את `models/README.md` tracked כדי שמי שמ-clone יידע להוריד.
