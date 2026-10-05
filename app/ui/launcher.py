"""Minimal launcher — cute, simple preflight screen before a session."""
from __future__ import annotations

import customtkinter as ctk

from app.config import load_config
from app.session import SessionOptions, run_session
from app.ui.preflight import CheckResult, preflight_ready, run_preflight

# Segmented control label → CLI mode
_MODE_OPTIONS: list[tuple[str, str]] = [
    ("ידני 🎮", "manual"),
    ("מחוות ✋", "gesture"),
    ("זיהוי 🙂", "identity"),
    ("שער 🔒", "webcam_faces"),
    ("מעקב 🎯", "tracking"),
    ("רדיפה 🔴", "chase"),
]
_MODE_LABEL_BY_VALUE = {value: label for label, value in _MODE_OPTIONS}

_ACCENT = "#2dd4bf"
_ACCENT_HOVER = "#14b8a6"
_CARD = ("#f4f4f8", "#1a1b26")
_MUTED = ("#6b7280", "#9ca3af")


class LauncherApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("Dronn")
        self.geometry("440x640")
        self.minsize(400, 600)

        self._mode_var = ctk.StringVar(value="gesture")
        self._live_var = ctk.BooleanVar(value=False)
        self._fpv_var = ctk.BooleanVar(value=False)
        self._camera_var = ctk.StringVar(value="0")
        self._check_labels: list[ctk.CTkLabel] = []
        self._running = False
        self._preflight_after_id: str | None = None

        self._build()
        self._schedule_preflight()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=20, pady=16)

        hero = ctk.CTkFrame(outer, corner_radius=20, fg_color=_CARD)
        hero.pack(fill="x", pady=(0, 14))

        ctk.CTkLabel(
            hero,
            text="🛸",
            font=ctk.CTkFont(size=44),
        ).pack(pady=(18, 0))

        ctk.CTkLabel(
            hero,
            text="Dronn",
            font=ctk.CTkFont(size=28, weight="bold"),
        ).pack(pady=(2, 0))

        ctk.CTkLabel(
            hero,
            text="שליטה ברחפן — מחוות, מעקב וזיהוי",
            font=ctk.CTkFont(size=13),
            text_color=_MUTED,
        ).pack(pady=(0, 16))

        card = ctk.CTkFrame(outer, corner_radius=16, fg_color=_CARD)
        card.pack(fill="both", expand=True)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=18, pady=16)

        ctk.CTkLabel(
            inner,
            text="מה נעשה היום?",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        seg = ctk.CTkSegmentedButton(
            inner,
            values=[label for label, _ in _MODE_OPTIONS],
            command=self._on_mode_segment,
            height=36,
            font=ctk.CTkFont(size=13),
        )
        seg.set(_MODE_LABEL_BY_VALUE["gesture"])
        seg.pack(fill="x", pady=(0, 14))
        self._mode_segment = seg

        toggles = ctk.CTkFrame(inner, fg_color="transparent")
        toggles.pack(fill="x", pady=(0, 10))

        ctk.CTkSwitch(
            toggles,
            text="רחפן אמיתי (Tello)",
            variable=self._live_var,
            command=self._on_live_toggle,
            progress_color=_ACCENT,
        ).pack(anchor="w", pady=3)

        self._fpv_switch = ctk.CTkSwitch(
            toggles,
            text="מצלמת רחפן (FPV)",
            variable=self._fpv_var,
            command=self._schedule_preflight,
            progress_color=_ACCENT,
            state="disabled",
        )
        self._fpv_switch.pack(anchor="w", pady=3)

        cam_row = ctk.CTkFrame(inner, fg_color="transparent")
        cam_row.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(cam_row, text="מצלמת מחשב", text_color=_MUTED).pack(side="left")
        ctk.CTkOptionMenu(
            cam_row,
            values=["0", "1", "2"],
            variable=self._camera_var,
            width=72,
            command=lambda _v: self._schedule_preflight(),
        ).pack(side="right")

        ctk.CTkLabel(
            inner,
            text="בדיקות מוכנות",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        self._checks_frame = ctk.CTkFrame(inner, corner_radius=12, fg_color=("white", "#12131a"))
        self._checks_frame.pack(fill="x", pady=(0, 14))

        self._fly_btn = ctk.CTkButton(
            inner,
            text="🚀  הטס!",
            height=50,
            corner_radius=14,
            font=ctk.CTkFont(size=20, weight="bold"),
            fg_color=_ACCENT,
            hover_color=_ACCENT_HOVER,
            command=self._on_fly,
        )
        self._fly_btn.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            inner,
            text="L = נחיתת חירום   ·   q / Esc = יציאה",
            font=ctk.CTkFont(size=11),
            text_color=_MUTED,
        ).pack()

        self._status = ctk.CTkLabel(
            inner,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=_MUTED,
        )
        self._status.pack(pady=(8, 0))

    def _on_mode_segment(self, label: str) -> None:
        for seg_label, value in _MODE_OPTIONS:
            if seg_label == label:
                self._mode_var.set(value)
                break
        self._schedule_preflight()

    def _on_live_toggle(self) -> None:
        live = self._live_var.get()
        if live:
            self._fpv_var.set(True)
        else:
            self._fpv_var.set(False)
        self._fpv_switch.configure(state="normal" if live else "disabled")
        self._schedule_preflight()

    def _schedule_preflight(self) -> None:
        if self._preflight_after_id is not None:
            self.after_cancel(self._preflight_after_id)
        self._status.configure(text="בודק מוכנות…")
        self._preflight_after_id = self.after(650, self._refresh_preflight)

    def _current_mode(self) -> str:
        return self._mode_var.get()

    def _refresh_preflight(self) -> None:
        self._preflight_after_id = None
        for w in self._check_labels:
            w.destroy()
        self._check_labels.clear()

        mode = self._current_mode()
        live = self._live_var.get()
        camera = int(self._camera_var.get())
        results = run_preflight(
            mode=mode,
            live_tello=live,
            camera_index=camera,
            config=load_config(),
        )

        for r in results:
            lbl = ctk.CTkLabel(
                self._checks_frame,
                text=self._format_check(r),
                anchor="w",
                font=ctk.CTkFont(size=12),
                text_color=self._check_color(r),
            )
            lbl.pack(fill="x", padx=12, pady=3)
            self._check_labels.append(lbl)

        ready = preflight_ready(results, mode=mode, live_tello=live)
        if self._running:
            return

        if ready:
            self._fly_btn.configure(
                state="normal",
                text="🚀  הטס!",
                fg_color=_ACCENT,
            )
            hint = "מוכן לטוס! ✨" if not live else "מוכן — ודא Wi‑Fi ל-Tello ✨"
            for r in results:
                if r.label == "סוללה Tello" and r.ok and "אזהרה" in r.detail:
                    hint = "מוכן — אזהרה: סוללה נמוכה ⚠️"
                    break
            self._status.configure(text=hint)
        else:
            self._fly_btn.configure(
                state="disabled",
                text="עוד לא מוכן",
                fg_color=("#9ca3af", "#374151"),
            )
            self._status.configure(text="תקן את הסעיפים האדומים למעלה")

    @staticmethod
    def _check_color(r: CheckResult) -> tuple[str, str]:
        if not r.ok:
            return ("#b91c1c", "#f87171")
        if "אזהרה" in r.detail:
            return ("#b45309", "#fbbf24")
        return ("#15803d", "#4ade80")

    @staticmethod
    def _format_check(r: CheckResult) -> str:
        mark = "✓" if r.ok else "✗"
        detail = f" — {r.detail}" if r.detail else ""
        return f"{mark}  {r.label}{detail}"

    def _on_fly(self) -> None:
        if self._running:
            return
        self._running = True
        self._fly_btn.configure(state="disabled", text="ממריא… 🛫")
        self.update_idletasks()

        opts = SessionOptions(
            mode=self._current_mode(),
            tello=self._live_var.get(),
            fpv=self._fpv_var.get() and self._live_var.get(),
            camera=int(self._camera_var.get()),
        )

        self.withdraw()
        try:
            run_session(opts)
        except Exception as exc:
            print(f"[launcher] session error: {exc}")
        finally:
            destroy_launcher_cv2_windows()
            self.deiconify()
            self._running = False
            self._refresh_preflight()

    def _on_close(self) -> None:
        if self._preflight_after_id is not None:
            self.after_cancel(self._preflight_after_id)
        self.destroy()


def destroy_launcher_cv2_windows() -> None:
    try:
        from app.cv2_gui import destroy_all_windows_safe

        destroy_all_windows_safe()
    except Exception:
        pass


def run_launcher() -> None:
    app = LauncherApp()
    app.mainloop()
