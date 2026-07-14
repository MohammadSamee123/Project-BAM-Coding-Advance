"""
gui.py
The desktop interface: a dark, modern-looking Tkinter window with a live
camera feed, a smooth caption overlay ("<name>, you are holding a <object>"),
and a collapsible AI chat sidebar for help.
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, font as tkfont

import cv2
from PIL import Image, ImageTk

from chatbot import ChatBot
from detector import ObjectDetector

# ---- palette -----------------------------------------------------------
BG = "#0f1115"
PANEL = "#171a21"
ACCENT = "#5b8cff"
ACCENT_SOFT = "#2a3550"
TEXT = "#e8eaf0"
SUBTEXT = "#8b92a5"
GOOD = "#5be49b"


class NamePrompt(tk.Toplevel):
    """First-run dialog asking the user's name before the main app opens."""

    def __init__(self, master, on_submit):
        super().__init__(master)
        self.on_submit = on_submit
        self.title("Welcome")
        self.configure(bg=BG)
        self.geometry("420x260")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        title_font = tkfont.Font(family="Helvetica", size=20, weight="bold")
        sub_font = tkfont.Font(family="Helvetica", size=11)

        tk.Label(
            self, text="Smart Object Detector", font=title_font, fg=TEXT, bg=BG
        ).pack(pady=(36, 6))
        tk.Label(
            self,
            text="What should I call you?",
            font=sub_font,
            fg=SUBTEXT,
            bg=BG,
        ).pack(pady=(0, 18))

        self.entry = tk.Entry(
            self,
            font=("Helvetica", 13),
            justify="center",
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=ACCENT_SOFT,
            highlightcolor=ACCENT,
        )
        self.entry.pack(ipady=8, padx=60, fill="x")
        self.entry.focus_set()
        self.entry.bind("<Return>", lambda _e: self._submit())

        submit_btn = tk.Button(
            self,
            text="Start",
            font=("Helvetica", 12, "bold"),
            bg=ACCENT,
            fg="white",
            activebackground=ACCENT,
            relief="flat",
            cursor="hand2",
            command=self._submit,
        )
        submit_btn.pack(pady=24, ipadx=20, ipady=6)

    def _submit(self):
        name = self.entry.get().strip() or "Friend"
        self.destroy()
        self.on_submit(name)

    def _on_close(self):
        self.destroy()
        self.on_submit("Friend")


class MainWindow(tk.Tk):
    def __init__(self, camera, user_name: str):
        super().__init__()
        self.camera = camera
        self.user_name = user_name
        self.title("Smart Object Detector")
        self.configure(bg=BG)
        self.geometry("1180x680")
        self.minsize(900, 560)

        self.detector = ObjectDetector()
        self.chatbot = ChatBot(user_name=user_name)

        self._latest_label = None
        self._latest_conf = 0.0
        self._detect_lock = threading.Lock()
        self._running = True

        self._build_layout()
        self._start_detection_thread()
        self._update_video()

    # ------------------------------------------------------------------
    def _build_layout(self):
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True, padx=16, pady=16)

        # Left: video + caption
        left = tk.Frame(container, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        header = tk.Frame(left, bg=BG)
        header.pack(fill="x", pady=(0, 10))
        tk.Label(
            header,
            text=f"Hi {self.user_name} \u2014 show me something!",
            font=("Helvetica", 16, "bold"),
            fg=TEXT,
            bg=BG,
        ).pack(side="left")

        self.chat_toggle_btn = tk.Button(
            header,
            text="\U0001F4AC  Help / Chat",
            font=("Helvetica", 10, "bold"),
            bg=PANEL,
            fg=TEXT,
            activebackground=ACCENT_SOFT,
            relief="flat",
            cursor="hand2",
            command=self._toggle_chat,
        )
        self.chat_toggle_btn.pack(side="right")

        video_frame = tk.Frame(left, bg=PANEL, highlightthickness=1, highlightbackground=ACCENT_SOFT)
        video_frame.pack(fill="both", expand=True)

        self.video_label = tk.Label(video_frame, bg=PANEL)
        self.video_label.pack(fill="both", expand=True, padx=4, pady=4)

        self.caption_var = tk.StringVar(value=f"{self.user_name}, show me an object...")
        self.caption_label = tk.Label(
            left,
            textvariable=self.caption_var,
            font=("Helvetica", 18, "bold"),
            fg=TEXT,
            bg=PANEL,
            anchor="w",
            padx=18,
            pady=14,
        )
        self.caption_label.pack(fill="x", pady=(10, 0))

        # Right: chat sidebar (hidden until toggled)
        self.chat_frame = tk.Frame(container, bg=PANEL, width=320)
        self._chat_visible = False
        self._build_chat_panel(self.chat_frame)

    def _build_chat_panel(self, parent):
        tk.Label(
            parent,
            text="Ask for help",
            font=("Helvetica", 13, "bold"),
            fg=TEXT,
            bg=PANEL,
        ).pack(anchor="w", padx=14, pady=(14, 4))

        self.chat_log = tk.Text(
            parent,
            bg="#11141a",
            fg=TEXT,
            relief="flat",
            wrap="word",
            state="disabled",
            font=("Helvetica", 10),
            padx=10,
            pady=10,
        )
        self.chat_log.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        self._append_chat("Assistant", "Hi! Ask me anything about how this app works.")

        suggestions = tk.Frame(parent, bg=PANEL)
        suggestions.pack(fill="x", padx=14, pady=(0, 8))
        for text in ["How does this work?", "Why isn't it detecting anything?"]:
            tk.Button(
                suggestions,
                text=text,
                font=("Helvetica", 9),
                bg=ACCENT_SOFT,
                fg=TEXT,
                relief="flat",
                cursor="hand2",
                command=lambda t=text: self._send_chat(t),
            ).pack(side="left", padx=(0, 6))

        input_row = tk.Frame(parent, bg=PANEL)
        input_row.pack(fill="x", padx=14, pady=(0, 14))

        self.chat_entry = tk.Entry(
            input_row,
            bg="#11141a",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Helvetica", 10),
        )
        self.chat_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 6))
        self.chat_entry.bind("<Return>", lambda _e: self._send_chat())

        tk.Button(
            input_row,
            text="Send",
            font=("Helvetica", 10, "bold"),
            bg=ACCENT,
            fg="white",
            relief="flat",
            cursor="hand2",
            command=lambda: self._send_chat(),
        ).pack(side="left")

    def _toggle_chat(self):
        self._chat_visible = not self._chat_visible
        if self._chat_visible:
            self.chat_frame.pack(side="right", fill="y", padx=(16, 0))
        else:
            self.chat_frame.pack_forget()

    def _append_chat(self, who: str, text: str):
        self.chat_log.configure(state="normal")
        tag = "user" if who == "You" else "assistant"
        self.chat_log.tag_config("user", foreground=ACCENT)
        self.chat_log.tag_config("assistant", foreground=GOOD)
        self.chat_log.insert("end", f"{who}: ", tag)
        self.chat_log.insert("end", f"{text}\n\n")
        self.chat_log.configure(state="disabled")
        self.chat_log.see("end")

    def _send_chat(self, preset_text: str = None):
        message = preset_text or self.chat_entry.get().strip()
        if not message:
            return
        self.chat_entry.delete(0, "end")
        self._append_chat("You", message)

        def worker():
            reply = self.chatbot.ask(message)
            self.after(0, lambda: self._append_chat("Assistant", reply))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    def _start_detection_thread(self):
        def loop():
            while self._running:
                frame = self.camera.read()
                if frame is not None:
                    result = self.detector.detect(frame)
                    with self._detect_lock:
                        self._latest_label = result["label"]
                        self._latest_conf = result["confidence"]
                    self._update_caption(result["label"])
                time.sleep(0.3)  # detection cadence; video stays smooth independently

        threading.Thread(target=loop, daemon=True).start()

    def _update_caption(self, label):
        if label is None:
            text = f"{self.user_name}, you are holding nothing."
        else:
            text = f"{self.user_name}, you are holding a {label}."
        self.after(0, lambda: self.caption_var.set(text))

    def _update_video(self):
        frame = self.camera.read()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # fit to label size while keeping aspect ratio
            label_w = max(self.video_label.winfo_width(), 640)
            label_h = max(self.video_label.winfo_height(), 480)
            img = Image.fromarray(rgb)
            img.thumbnail((label_w, label_h))
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # keep reference

        if self._running:
            self.after(15, self._update_video)  # ~60fps target for a smooth feed

    def destroy(self):
        self._running = False
        super().destroy()
