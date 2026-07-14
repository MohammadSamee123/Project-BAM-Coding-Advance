# Smart Object Detector 🎥🔍

A native desktop app that watches your webcam, tells you exactly what you're
holding up to the camera, and greets you by name. Includes a built-in AI
chat assistant (powered by Claude) to help you use the app.

> "Mohammad, you are holding a mouse."
> "Mohammad, you are holding nothing."

## How it works

- **Camera** — OpenCV grabs frames from your webcam on a background thread
  so the video stays smooth.
- **Detection (accuracy-first, two stages)**
  1. **YOLOv8** finds and classifies the object filling most of the frame
     (the thing you're "holding up") from 80 everyday categories — mouse,
     fork, cup, cell phone, keyboard, and more.
  2. **CLIP** (zero-shot) kicks in only for broad "device" categories
     (phone, laptop, TV, remote, book, keyboard) to sharpen the label
     further — e.g. telling an **iPad** apart from a MacBook or a Kindle.
- **GUI** — a clean, dark, native Tkinter window: live video, a smooth
  caption bar, and a collapsible chat sidebar. No browser required.
- **Chatbot** — a "Help / Chat" panel backed by the Anthropic API that can
  answer questions about how the app works or why detection might be off.

## Requirements

- Python 3.9–3.11
- A webcam
- An [Anthropic API key](https://console.anthropic.com/) (only needed for
  the chat assistant — object detection works without it)

## Setup

```bash
git clone <this-repo-url>
cd smart-object-detector

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # then edit .env and add your ANTHROPIC_API_KEY
```

The first run will automatically download the YOLOv8 weights (~20–50 MB)
and the CLIP model (~600 MB) — this only happens once.

## Run

```bash
python main.py
```

1. Type your name in the welcome dialog.
2. Hold an object up to your webcam.
3. Watch the caption update with what the app sees.
4. Click **Help / Chat** any time to ask the built-in assistant a question.

## Tuning accuracy vs. speed

Edit these values in your `.env` file:

| Variable              | Effect                                                             |
|-----------------------|---------------------------------------------------------------------|
| `YOLO_MODEL`          | `yolov8n.pt` (fastest) → `yolov8x.pt` (most accurate, slower)       |
| `YOLO_CONF_THRESHOLD` | Higher = fewer false positives, but may miss less-obvious objects   |
| `CLIP_CONF_THRESHOLD` | Higher = CLIP only overrides YOLO's label when very confident       |

If you have an NVIDIA GPU with CUDA installed, both models will
automatically run on it for a big speed boost — no code changes needed.

## Project structure

```
smart-object-detector/
├── main.py        # entry point
├── camera.py       # threaded webcam capture
├── detector.py     # YOLOv8 + CLIP object recognition
├── chatbot.py      # Anthropic API chat wrapper
├── gui.py          # Tkinter UI (name prompt, video, caption, chat sidebar)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Troubleshooting

- **"Could not open camera"** — another app may be using the webcam, or a
  different camera index is needed. Try `Camera(source=1)` in `main.py`.
- **Slow first launch** — model downloads happen once; subsequent launches
  are fast.
- **Chat says "no ANTHROPIC_API_KEY was found"** — make sure `.env` exists
  (copied from `.env.example`) and contains a valid key, then restart the app.
- **Object detected incorrectly** — try better lighting, hold the object
  closer/steadier, or bump up `YOLO_MODEL` to a larger variant for more
  accuracy.

## License

MIT — do whatever you'd like with this.
