"""
detector.py
Two-stage object recognition:

  Stage 1 (YOLOv8):  fast, accurate detection + bounding box for anything in
                      the COCO 80-class list (mouse, fork, cell phone, cup, ...).

  Stage 2 (CLIP):     when YOLO's label is a broad "device" category (cell
                      phone, laptop, tv, remote, book, keyboard), we crop that
                      region and ask CLIP to pick the closest match from a
                      richer, more specific vocabulary (iPad, iPhone, Kindle,
                      MacBook, game controller, ...). This lets the app say
                      "iPad" instead of just "laptop" when it's confident.

The two stages are combined so you get YOLO's speed/accuracy for the general
case, with CLIP sharpening the label only where it adds real value.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor

# Confidence thresholds (tuned to favor accuracy over chattiness)
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.55"))
CLIP_CONF_THRESHOLD = float(os.getenv("CLIP_CONF_THRESHOLD", "0.35"))

# YOLO classes broad enough that CLIP refinement is worth running
REFINABLE_CLASSES = {"cell phone", "laptop", "tv", "remote", "book", "keyboard"}

# Specific candidates CLIP is allowed to choose between when refining.
# Feel free to extend this list with more specific products/objects.
CLIP_CANDIDATES = [
    "an iPad tablet", "an iPhone", "an Android smartphone", "a Kindle e-reader",
    "a MacBook laptop", "a Windows laptop", "a Nintendo Switch", "a game controller",
    "a TV remote control", "a television screen", "a paperback book", "a notebook",
    "a computer keyboard", "a calculator",
]

# Human-friendly names for the raw COCO labels ultralytics returns
FRIENDLY_NAMES = {
    "cell phone": "phone",
    "tv": "TV",
    "remote": "remote control",
    "diningtable": "table",
    "pottedplant": "plant",
    "tvmonitor": "monitor",
}


def _friendly(label: str) -> str:
    return FRIENDLY_NAMES.get(label, label)


class ObjectDetector:
    def __init__(self, yolo_weights: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        yolo_weights = yolo_weights or os.getenv("YOLO_MODEL", "yolov8s.pt")
        self.yolo = YOLO(yolo_weights)

        self._clip_model = None
        self._clip_processor = None
        self._clip_text_features = None  # cached embeddings for CLIP_CANDIDATES

    # -- lazy CLIP loading so app startup (and the "no CLIP needed" path) is fast --
    def _ensure_clip(self):
        if self._clip_model is not None:
            return
        self._clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self._clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        with torch.no_grad():
            text_inputs = self._clip_processor(
                text=CLIP_CANDIDATES, return_tensors="pt", padding=True
            ).to(self.device)
            features = self._clip_model.get_text_features(**text_inputs)
            self._clip_text_features = features / features.norm(dim=-1, keepdim=True)

    def _refine_with_clip(self, crop_bgr: np.ndarray):
        """Return (label, confidence) from CLIP, or None if not confident enough."""
        self._ensure_clip()
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        with torch.no_grad():
            image_inputs = self._clip_processor(images=image, return_tensors="pt").to(
                self.device
            )
            image_features = self._clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ self._clip_text_features.T).softmax(dim=-1)
            best_idx = int(similarity.argmax())
            best_score = float(similarity[0, best_idx])

        if best_score >= CLIP_CONF_THRESHOLD:
            label = CLIP_CANDIDATES[best_idx]
            # strip leading article for a cleaner label, e.g. "an iPad tablet" -> "iPad"
            label = label.replace("a ", "", 1).replace("an ", "", 1)
            label = label.split(" tablet")[0].split(" laptop")[0].split(" e-reader")[0]
            return label, best_score
        return None

    def detect(self, frame: np.ndarray):
        """
        Run detection on a single BGR frame.
        Returns dict: {"label": str or None, "confidence": float, "box": (x1,y1,x2,y2) or None}
        "label" is None when nothing is confidently detected ("holding nothing").
        The largest confident detection is treated as "what's being held up"
        since that's typically the object closest to / filling the camera view.
        """
        results = self.yolo.predict(
            frame, conf=YOLO_CONF_THRESHOLD, verbose=False, device=self.device
        )[0]

        best = None
        best_area = 0
        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                cls_id = int(box.cls[0])
                raw_label = self.yolo.names[cls_id]
                best = {"label": raw_label, "confidence": conf, "box": (x1, y1, x2, y2)}

        if best is None:
            return {"label": None, "confidence": 0.0, "box": None}

        # Stage 2: refine broad "device" labels into something more specific
        if best["label"] in REFINABLE_CLASSES:
            x1, y1, x2, y2 = best["box"]
            crop = frame[max(0, y1): y2, max(0, x1): x2]
            if crop.size > 0:
                refined = self._refine_with_clip(crop)
                if refined is not None:
                    label, score = refined
                    best["label"] = label
                    best["confidence"] = max(best["confidence"], score)
                    return best

        best["label"] = _friendly(best["label"])
        return best
