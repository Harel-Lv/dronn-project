# חיבור DJI / Ryze Tello לפרויקט

## 1. רשת Wi‑Fi

1. הפעל את ה-Tello (לחצן כוח ארוך).
2. במחשב: התחבר לרשת **`Tello-XXXXXX`** (הסיסמה מופיעה לעיתים על גוף הרחפן או במדריך).
3. **אין אינטרנט** בזמן החיבור — זה נורמלי (מצב "רק רחפן").

כתובת ברירת המחדל של הרחפן ברשת הזו: **`192.168.10.1`**.  
אם משתמשים ב־Wi‑Fi Extender / רשת מורכבת, אולי תצטרך `tello_host` ב־`config.yaml`.

## 2. חבילות Python

```bash
pip install djitellopy av
```

- **`djitellopy`** — פקודות ו־RC.
- **`av`** — נדרש לזרם וידאו (FPV) אם משתמשים ב־`--fpv`.

## 3. איך מפעילים מהפרויקט

### אופציה א׳ — שורת פקודה (מומלץ)

מחוות מהמצלמה **+** רחפן אמיתי:

```bash
python main.py --mode gesture --tello
```

עם תצוגת מצלמת הרחפן (חלון נוסף; **המחוות נשארות ממצלמת המחשב**):

```bash
python main.py --mode gesture --tello --fpv
```

- חלון אחד: ידיים / מחוות (webcam).
- חלון שני: **Tello FPV** — מה שהרחפן רואה.
- דורש `pip install av`.

מצבים נתמכים עם רחפן: `gesture`, `webcam_faces`, `tracking` (מעקב גוף), `manual` (מקלדת).  
כולם משתמשים ב־`DroneController` — הוסף `--tello` (ובמעקב/ידני גם FPV).

```bash
# מעקב גוף אחרי המראה
python main.py --mode tracking --tello
```

### אופציה ב׳ — `config.yaml`

במקטע `drone`:

```yaml
drone:
  enabled: true
  backend: tello
  show_tello_fpv: true   # אופציונלי — דורש av
  tello_host: null       # או "192.168.10.1" אם צריך
```

אז אפשר בלי `--tello`:

```bash
python main.py --mode gesture
```

## 4. בדיקה שהחיבור עובד

בהפעלה מוצלחת אמור להופיע בטרמינל משהו כמו:

```text
[drone] Tello: battery=XX%
```

אם יש כשל — הפרויקט מנסה שוב לפי `tello_connect_retries` / `tello_connect_retry_delay_sec` ב־`config.yaml`.

## 5. בעיות נפוצות

| תסמין | מה לבדוק |
|--------|-----------|
| `str, bytes or bytearray expected, not NoneType` ב־`sendto` | תוקן בפרויקט: `tello_host: null` לא מעביר יותר `host=None` ל־djitellopy. עדכן קוד / השאר `null` ל־192.168.10.1 |
| `Did not receive a state packet` | Wi‑Fi מחובר ל־Tello-XX? חומת אש / VPN כבויים? |
| FPV לא נפתח | `pip install av`; הרץ עם `--fpv` או `show_tello_fpv: true` |
| פקודות לא מגיעות | ודא `--tello` או `enabled: true` + `backend: tello` |
| Tello EDU ברשת מורכבת | הגדר `tello_host` לכתובת ה-IP של הרחפן |

## 6. בטיחות

- טוס רק במקום פתוח, רחוק מאנשים.
- התחל ב**סימולציה** (`בלי --tello`) עד שהמחוות יציבות.
- מחוות **TAKEOFF** / **LAND** שולחות פקודות אמיתיות לרחפן כש־`--tello` פעיל.
