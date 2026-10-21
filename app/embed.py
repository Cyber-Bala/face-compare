"""AdaFace IR-101 ONNX embedding."""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "adaface_ir101.onnx"


def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def preprocess(crop112_bgr):
    """BGR 112x112 uint8 -> (1,3,112,112) float32 in [-1, 1]."""
    x = (crop112_bgr.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.transpose(x, (2, 0, 1))[None]


class Embedder:
    def __init__(self):
        self._sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        self._in = self._sess.get_inputs()[0].name

    def embed(self, crop112_bgr):
        """Return unit-length 512-d vector."""
        batch = np.concatenate([preprocess(crop112_bgr), preprocess(crop112_bgr[:, ::-1])])
        out = self._sess.run(None, {self._in: batch})[0]
        return l2_normalize(l2_normalize(out[0]) + l2_normalize(out[1]))

    def embed_many(self, crops):
        """Embed several crops, average, re-normalize to length 1."""
        vs = np.stack([self.embed(c) for c in crops])
        return l2_normalize(vs.mean(axis=0))
