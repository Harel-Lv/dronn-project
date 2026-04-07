# פרופיל זהות יחיד (`--mode identity` / `webcam_faces`)

## למה כולם היו "מזוהים" כאותו אדם?

**MediaPipe Face Landmarks** (מפת פרצוף) נועדו לעקוב אחרי תנועות פנים — **לא** לזהות זהות בין אנשים שונים.  
לכן השתמשנו ב־**ArcFace דרך InsightFace** — embedding שמבדיל בין אנשים.

## התקנה (ArcFace)

```bash
pip install insightface onnxruntime
```

(או `onnxruntime-gpu` אם יש GPU מתאים.)

## `config.yaml` — מקטע `identity`

| מפתח | משמעות |
|------|--------|
| `embedding_backend` | `insightface` (ברירת מחדל, מומלץ) או `landmarks` (חלש לזיהוי אנשים) |
| `match_threshold` | סף ל־landmarks בלבד (~0.88) |
| `match_threshold_insightface` | סף ל־ArcFace — טיפוסי **0.35–0.45**; **גבוה יותר = רק מי שדומה מאוד** |
| `profile_path` | נתיב ל־`profile.npz` |
| `display_name` | שם ברישום ראשון |
| `enroll_frame_count` | כמה פריימים לאסוף ברישום |

## מעבר מ־landmarks ל־InsightFace

אם כבר נשמר `profile.npz` עם landmarks, **מחק** את הקובץ (או לחץ **N** במצב identity ורשום מחדש) אחרי שמפעילים `embedding_backend: insightface`.

- **`python main.py --mode identity`** — תצוגה + מלבן + שם רק בהתאמה.
- **`python main.py --mode webcam_faces`** — אותו פרופיל + מחוות.

הקובץ ב־`.gitignore` — לא לשתף בפומבי.
