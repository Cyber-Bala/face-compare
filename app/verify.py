"""Face check, scoring, thresholds. Quality gate rejects blurry, shaky, dark, tiny or tilted photos before matching."""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .detect_align import FaceDetector, align, quality
from .embed import Embedder, l2_normalize

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "config.json"

DEFAULTS = {
    # Calibrated with tools/calibrate_single.py on ~3500 student ID photos (6M wrong-person pairs):
    # 0.50 accepts a wrong person ~2 in 100,000 per attempt (0.25 accepted ~6 in 100).
    # Re-tune with data/scores.csv (real live scores) after a week of use.
    "t_accept": 0.50,
    "min_good_frames": 2,    # live frames that must pass the quality gate (capped at frames sent)
    "max_abs_yaw": 0.6,      # only reject extreme side views
    "max_attempts": 3,       # 1 try + 2 retries
    # Photo quality gate. Tune with real photos; override in data/config.json.
    "min_face_width": 60,    # px in the (resized) image: face too small / far
    "min_blur": 30.0,        # Laplacian variance of face region: below = blurry / shaky
    "min_brightness": 40.0,  # mean gray of aligned face: too dark
    "max_brightness": 225.0, # too bright / washed out
    "min_contrast": 20.0,    # gray std of aligned face: flat / washed out
    "min_det_score": 0.5,    # detector confidence
    "max_abs_roll": 35.0,    # degrees of head tilt
}


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    if os.environ.get("T_ACCEPT"):
        cfg["t_accept"] = float(os.environ["T_ACCEPT"])
    return cfg


class QualityError(Exception):
    pass


def decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise QualityError("Could not read the image. Upload a JPG or PNG.")
    return img


def _is_ghost(main, other):
    """Blurry faces make the detector fire twice (chin/neck, hair). A second box that
    overlaps the main face, or scores clearly lower, is the same face, not a person."""
    mx, my, mw, mh = main.box
    ox, oy, ow, oh = other.box
    iw = max(0.0, min(mx + mw, ox + ow) - max(mx, ox))
    ih = max(0.0, min(my + mh, oy + oh) - max(my, oy))
    overlap = iw * ih / (min(mw * mh, ow * oh) + 1e-6)
    return overlap > 0.15 or other.score < 0.85


class Engine:
    def __init__(self):
        self.det = FaceDetector()
        self.emb = Embedder()
        self.cfg = load_config()

    def _one_face(self, img):
        img, faces = self.det.detect(img)
        if not faces:
            raise QualityError("No face found in the photo. Use a clearer photo showing the face.")
        f = faces[0]
        faces = [f] + [g for g in faces[1:] if not _is_ghost(f, g)]
        if len(faces) > 1 and faces[1].box[2] * faces[1].box[3] > 0.25 * f.box[2] * f.box[3]:
            raise QualityError("More than one face in the photo.")
        crop = align(img, f.landmarks)
        return crop, quality(img, f, crop)

    def embed_photo(self, data: bytes):
        """Photo bytes -> unit 512-d embedding. Raises QualityError."""
        img = decode(data)
        crop, q = self._one_face(img)
        self._check_quality(q)
        return self.emb.embed(crop)

    def _check_quality(self, q):
        """Reject unusable photos before matching. Raises QualityError."""
        c = self.cfg
        if abs(q["yaw"]) > c["max_abs_yaw"]:
            raise QualityError("Face turned too far sideways. Use a front-facing photo.")
        if abs(q["roll"]) > c["max_abs_roll"]:
            raise QualityError("Head is tilted too much. Keep the head straight.")
        if q["face_width"] < c["min_face_width"]:
            raise QualityError("Face too small or too far. Move closer to the camera.")
        if q["det_score"] < c["min_det_score"]:
            raise QualityError("Face not clear. Retake the photo facing the camera.")
        if q["blur"] < c["min_blur"]:
            raise QualityError("Photo is blurry or shaky. Hold still and retake in focus.")
        if q["brightness"] < c["min_brightness"]:
            raise QualityError("Photo too dark. Retake in better light.")
        if q["brightness"] > c["max_brightness"]:
            raise QualityError("Photo too bright. Avoid strong light behind or on the face.")
        if q["contrast"] < c["min_contrast"]:
            raise QualityError("Photo is washed out or flat. Retake in even, good light.")

    def embed_photos(self, datas):
        """Several live frames -> one averaged unit embedding. Frames without a usable face are skipped."""
        vs, last = [], None
        for d in datas:
            try:
                vs.append(self.embed_photo(d))
            except QualityError as e:
                last = e
        if not vs:
            raise last or QualityError("No photo received.")
        return l2_normalize(np.mean(vs, axis=0))

    def match_frames(self, datas, stored):
        """Score each usable live frame against the stored embedding.

        Every usable frame must reach t_accept, so one good frame cannot hide a bad one.
        Returns the LOWEST frame score. Raises QualityError if too few frames pass the quality gate.
        """
        scores, last = [], None
        for d in datas:
            try:
                scores.append(self.score(self.embed_photo(d), stored))
            except QualityError as e:
                last = e
        need = max(1, min(self.cfg["min_good_frames"], len(datas)))
        if len(scores) < need:
            msg = str(last) if last else "No photo received."
            if len(scores) and len(datas) > 1:
                msg = f"Only {len(scores)} of {len(datas)} photos were clear enough. {msg}"
            raise QualityError(msg)
        return min(scores)

    @staticmethod
    def score(a, b):
        return float(np.dot(l2_normalize(a), l2_normalize(b)))

    def accept(self, score):
        return score >= self.cfg["t_accept"]


def thumbnail(data: bytes, side=360) -> bytes:
    """Small JPEG of the uploaded photo, kept for the admin list."""
    img = decode(data)
    h, w = img.shape[:2]
    k = side / max(h, w)
    if k < 1:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
