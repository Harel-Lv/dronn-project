# גלריית פרצופים (`fpv_faces`)

בדיקת גלריה על מצלמת המחשב (מספר אנשים): הרץ `run_face_gallery_webcam_loop` מ־`app/run_face_gallery_webcam.py` או דרך קוד מותאם (לא `--mode identity` — זה פרופיל יחיד).

תיקייה זו מכילה קבצי **`.npz`** — כל קובץ הוא פרופיל אחד (וקטור embedding + שם תצוגה).

## איך נוצרים קבצים

- דרך תכונות הפרויקט ששומרות embeddings (למשל אחרי רישום), או
- העתקה ידנית של `.npz` תואם למבנה:

  - שדה `embedding` — וקטור `float64` מנורמל
  - שדה `name` (אופציונלי) — שם שיוצג בזיהוי

## הגדרות ב־`config.yaml`

במקטע `fpv_faces`:

| מפתח | משמעות |
|------|--------|
| `gallery_dir` | נתיב לתיקייה זו (ברירת מחדל: `data/faces_gallery`) |
| `embedding_backend` | `landmarks` (MediaPipe) או `insightface` (ArcFace) |
| `match_threshold` | סף דמיון ל־landmarks (~0.93 נפוץ) |
| `match_threshold_insightface` | סף ל־InsightFace (~0.42 נפוץ) |
| `allowed_names` | רשימת שמות מורשים בלבד, או `null` לכולם |

## הערות

- קבצי `.npz` מוזנים ב־`.gitignore` כדי שלא יעלו בטעות ל־Git (פרטיות).
- ל־InsightFace נדרשות חבילות נוספות; ראו `requirements.txt`.
