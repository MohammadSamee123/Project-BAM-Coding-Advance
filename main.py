"""
main.py
Entry point for Smart Object Detector.

Run with:  python main.py
"""

import sys
import tkinter as tk

from dotenv import load_dotenv

from camera import Camera
from gui import NamePrompt, MainWindow

load_dotenv()  # pulls ANTHROPIC_API_KEY / CLAUDE_MODEL / YOLO_MODEL etc. from .env


def launch_main_app(camera, user_name):
    app = MainWindow(camera, user_name)
    app.mainloop()


def main():
    try:
        camera = Camera(source=0).start()
    except RuntimeError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()  # hidden root just so NamePrompt has a valid master

    def on_name_submitted(name):
        root.destroy()
        launch_main_app(camera, name)

    NamePrompt(root, on_submit=on_name_submitted)
    root.mainloop()

    camera.stop()


if __name__ == "__main__":
    main()
