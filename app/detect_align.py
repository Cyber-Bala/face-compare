"""YuNet face detection, 5-point alignment to 112x112, quality metrics."""
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"

# Standard ArcFace/AdaFace 112x112 landmark template:
# right eye, left eye, nose tip, right mouth corner, left mouth corner.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

MAX_SIDE = 1280  # downscale huge phone photos before detection
MIN_SIDE = 640   # upscale tiny / low-res images before detection


class Face:
    def __init__(self, box, landmarks, score):
        self.box = box  # x, y, w, h
        self.landmarks = landmarks  # (5, 2)
        self.score = score


def _resize(img):
    h, w = img.shape[:2]
    m = max(h, w)
    if m > MAX_SIDE:
        s = MAX_SIDE / m
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    elif m < MIN_SIDE:
        s = MIN_SIDE / m
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return img


class FaceDetector:
    def __init__(self, score_threshold=0.5):
        self.score_threshold = score_threshold
        self._det = cv2.FaceDetectorYN.create(
            str(MODEL_PATH), "", (320, 320), score_threshold, 0.3, 5000
        )

    def detect(self, img):
        """Return (resized_img, [Face, ...]) sorted by area, largest first."""
        img = _resize(img)
        faces = self._run(img)
        if not faces:  # tight crops / weak faces: pad the border, loosen the score
            pad = int(0.3 * max(img.shape[:2]))
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            for thr in (self.score_threshold, 0.3):
                self._det.setScoreThreshold(thr)
                faces = self._run(padded)
                if faces:
                    img = padded
                    break
            self._det.setScoreThreshold(self.score_threshold)
        faces.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
        return img, faces

    def _run(self, img):
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _, res = self._det.detect(img)
        faces = []
        if res is not None:
            for r in res:
                faces.append(Face(r[0:4].astype(float), r[4:14].reshape(5, 2).astype(np.float32), float(r[14])))
        return faces


def align(img, landmarks):
    """Similarity-transform the face to the 112x112 template."""
    M, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0)


def yaw_ratio(lm):
    """Signed head-turn estimate: nose offset from eye midpoint / eye distance. ~0 = frontal."""
    re, le, nose = lm[0], lm[1], lm[2]
    eye_dist = np.linalg.norm(le - re) + 1e-6
    return float((nose[0] - (re[0] + le[0]) / 2) / eye_dist)


def roll_deg(lm):
    d = lm[1] - lm[0]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def quality(img, face, crop112):
    """Metrics used by the quality gate."""
    gray = cv2.cvtColor(crop112, cv2.COLOR_BGR2GRAY)
    # Sharpness: Laplacian variance on the face box, normalised to a fixed size so it
    # does not depend on photo resolution. Motion blur / out-of-focus lowers it.
    x, y, w, h = [int(round(v)) for v in face.box]
    x0, y0 = max(x, 0), max(y, 0)
    region = img[y0:max(y0 + 1, y + h), x0:max(x0 + 1, x + w)]
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.size else gray
    region = cv2.resize(region, (160, 160), interpolation=cv2.INTER_AREA)
    return {
        "face_width": float(face.box[2]),
        "blur": float(cv2.Laplacian(region, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "yaw": yaw_ratio(face.landmarks),
        "roll": roll_deg(face.landmarks),
        "det_score": face.score,
    }
