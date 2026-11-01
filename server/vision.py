"""
vision.py – Server-side OpenCV Haar face detection.
Processes JPEG base64 frames, never stores image bytes.
Episodes: 2 consecutive frames trigger +1 count.
"""
import base64
from typing import Dict, Tuple

import cv2
import numpy as np

_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _cascade


def check_image_quality(image_b64: str) -> Dict:
    """Return brightness and blur metrics for pre-exam camera quality check."""
    try:
        img_data = base64.b64decode(image_b64)
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return {"ok": False, "reason": "Could not decode image"}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        warnings = []
        if brightness < 40:
            warnings.append("Too dark – please improve lighting")
        elif brightness > 220:
            warnings.append("Too bright – reduce glare")
        if blur < 50:
            warnings.append("Image appears blurry – check camera focus")
        return {"ok": len(warnings) == 0, "brightness": round(brightness, 1),
                "blur": round(blur, 1), "warnings": warnings}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def process_frame(image_b64: str, face_state: Dict) -> Tuple[Dict, int]:
    """
    Process one frame, update face_state in-place, return (updated_state, n_faces).
    face_state must have keys: face_not_detected_count, multiple_faces_count,
                                consecutive_no_face, consecutive_multi.
    """
    try:
        img_data = base64.b64decode(image_b64)
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return face_state, -1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        n = len(faces)

        if n == 0:
            face_state["consecutive_no_face"] += 1
            face_state["consecutive_multi"] = 0
            if face_state["consecutive_no_face"] >= 2:
                face_state["face_not_detected_count"] += 1
                face_state["consecutive_no_face"] = 0
        elif n > 1:
            face_state["consecutive_multi"] += 1
            face_state["consecutive_no_face"] = 0
            if face_state["consecutive_multi"] >= 2:
                face_state["multiple_faces_count"] += 1
                face_state["consecutive_multi"] = 0
        else:
            face_state["consecutive_no_face"] = 0
            face_state["consecutive_multi"] = 0

        return face_state, n
    except Exception:
        return face_state, -1

