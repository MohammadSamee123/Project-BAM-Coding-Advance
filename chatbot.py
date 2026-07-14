"""
chatbot.py
Thin wrapper around the Anthropic API to power the in-app "Help" chat.
Reads ANTHROPIC_API_KEY (and optionally CLAUDE_MODEL) from the environment,
which main.py loads from a local .env file via python-dotenv.
"""

import os
import anthropic

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are the friendly built-in help assistant for a desktop
app called "Smart Object Detector". Keep answers short (2-4 sentences unless
asked for more) and practical.

What the app does:
- It opens the user's webcam and shows a live video feed in a window.
- It uses an AI object detector (YOLOv8, refined in some cases by CLIP) to
  figure out what object the user is holding up to the camera and displays
  a message like "<name>, you are holding a <object>."
- If nothing recognizable is in frame, it says "<name>, you are holding
  nothing."
- There is a chat panel (this one) where users can ask questions about how
  the app works, why detection might be failing, or how to configure it.

Common troubleshooting you can help with:
- Poor lighting or the object being too far/close/blurry reduces accuracy.
- Only common everyday objects and a curated list of gadgets are recognized
  well; very unusual objects may be mislabeled or missed.
- If the camera doesn't open, another app may be using it, or camera
  permissions may need to be granted to the terminal/Python.
- The YOLO_MODEL and confidence thresholds can be changed in the .env file
  for a speed/accuracy tradeoff (e.g. yolov8n.pt is faster, yolov8m.pt/yolov8l.pt
  are more accurate but slower).
"""


class ChatBot:
    def __init__(self, user_name: str = "there", model: str = None):
        self.user_name = user_name
        self.model = model or DEFAULT_MODEL
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
        self.history = []

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def ask(self, message: str) -> str:
        if not self.is_configured:
            return (
                "I can't chat yet — no ANTHROPIC_API_KEY was found.\n"
                "Add one to your .env file (see .env.example) and restart the app."
            )

        self.history.append({"role": "user", "content": message})
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=self.history,
            )
            reply = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except anthropic.AuthenticationError:
            return "Your Anthropic API key looks invalid. Please check it in your .env file."
        except Exception as exc:  # noqa: BLE001 - surface any API/network issue to the user
            return f"Sorry, I hit an error talking to the AI service: {exc}"
