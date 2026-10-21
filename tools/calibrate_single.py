"""Pick T_ACCEPT from a folder of ONE photo per student (file stem = student id).

Impostor = every pair of different stems (the dangerous side, measured for real).
Genuine  = (a) same stem found in more than one folder, if any (real pairs)
           (b) augmented copy of a photo (blur, light, rotation, low-res, JPEG) vs its original (estimate)
Usage: python -m tools.calibrate_single <folder> [--far 1e-5] [--aug 500] [--write]
Embeddings are cached in --cache so a re-run is fast. Nothing is copied or uploaded.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.verify import CONFIG_PATH, Engine, QualityError  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png"}
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def embed_raw(eng, img):
    """Embedding without the quality gate, plus the metrics the gate would use."""
    crop, q = eng._one_face(img)
    return eng.emb.embed(crop), q


def gate_fails(eng, q):
    try:
        eng._check_quality(q)
        return None
    except QualityError as e:
        return str(e)


def augment(img, rng):
    """Make a photo look like a worse live capture."""
    h, w = img.shape[:2]
    out = img.copy()
    ang = rng.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, rng.uniform(0.9, 1.1))
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    s = rng.uniform(0.35, 0.8)
    out = cv2.resize(cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), (w, h))
    if rng.random() < 0.6:
        k = rng.uniform(0.5, 2.0)
        out = cv2.GaussianBlur(out, (0, 0), k)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.6, 1.3) + rng.uniform(-35, 35), 0, 255).astype(np.uint8)
    out = np.clip(out.astype(np.float32) + rng.normal(0, rng.uniform(0, 8), out.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 75))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def load_all(eng, folder, cache):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)
    done = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        done = {k: (e, q) for k, e, q in zip(z["paths"], z["embs"], z["quals"])}
    recs, skipped, t0 = [], [], time.time()
    for i, f in enumerate(files):
        key = str(f)
        if key in done:
            e, q = done[key]
            recs.append((key, f.stem, e, q))
            continue
        img = cv2.imread(key, cv2.IMREAD_COLOR)
        try:
            if img is None:
                raise QualityError("unreadable")
            e, q = embed_raw(eng, img)
            recs.append((key, f.stem, e, q))
        except QualityError as err:
            skipped.append((key, str(err)))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(files)}  {el:.0f}s", flush=True)
    np.savez(cache, paths=np.array([r[0] for r in recs], dtype=object),
             embs=np.array([r[2] for r in recs]), quals=np.array([r[3] for r in recs], dtype=object))
    return recs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--far", type=float, default=1e-5, help="target false-accept rate per attempt")
    ap.add_argument("--aug", type=int, default=500, help="photos to augment for the genuine estimate")
    ap.add_argument("--cache", default="calib_cache.npz")
    ap.add_argument("--write", action="store_true", help="write t_accept to data/config.json")
    a = ap.parse_args()

    eng = Engine()
    print("embedding photos ...")
    recs, skipped = load_all(eng, a.folder, Path(a.cache))
    print(f"usable {len(recs)}  skipped (no single face / unreadable) {len(skipped)}")
    for p, why in skipped[:15]:
        print("  skip", p, "-", why)

    # ---- quality gate impact on ID photos
    reasons = defaultdict(int)
    for _, _, _, q in recs:
        r = gate_fails(eng, q)
        if r:
            reasons[r] += 1
    print(f"\nquality gate would reject {sum(reasons.values())}/{len(recs)} of these photos:")
    for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {r}")

    # ---- impostor scores (all pairs of different stems)
    E = np.stack([r[2] for r in recs]).astype(np.float32)
    stems = np.array([r[1] for r in recs])
    S = E @ E.T
    same = stems[:, None] == stems[None, :]
    iu = np.triu_indices(len(recs), k=1)
    mask = ~same[iu]
    imp = S[iu][mask]
    real_gen = S[iu][~mask]
    print(f"\nimpostor pairs {len(imp):,}  mean {imp.mean():.3f}  p99.99 {np.quantile(imp, 0.9999):.3f}  max {imp.max():.3f}")
    print(f"real same-id pairs (same file stem in 2+ folders): {len(real_gen)}")
    if len(real_gen):
        print(f"  mean {real_gen.mean():.3f}  min {real_gen.min():.3f}")

    # ---- most similar different-id pairs: look-alikes or one person under two ids
    order = np.argsort(-np.where(same, -2, S)[iu])[:10]
    print("\nhighest-scoring DIFFERENT-id pairs (check these: look-alikes or duplicate people):")
    for k in order:
        i, j = iu[0][k], iu[1][k]
        print(f"  {S[i, j]:.3f}  {recs[i][1]}  vs  {recs[j][1]}")

    # ---- augmented genuine estimate
    rng = np.random.default_rng(0)
    idx = rng.choice(len(recs), size=min(a.aug, len(recs)), replace=False)
    aug_gen = []
    print(f"\naugmenting {len(idx)} photos for the genuine estimate ...")
    for n, i in enumerate(idx):
        img = cv2.imread(recs[i][0], cv2.IMREAD_COLOR)
        try:
            e, _ = embed_raw(eng, augment(img, rng))
            aug_gen.append(float(recs[i][2] @ e))
        except QualityError:
            aug_gen.append(None)  # detector lost the face: counts as a reject
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(idx)}", flush=True)
    lost = sum(v is None for v in aug_gen)
    ag = np.array([v for v in aug_gen if v is not None])

    print(f"\ngenuine (augmented) n={len(ag)} mean {ag.mean():.3f}  p5 {np.quantile(ag, 0.05):.3f}  min {ag.min():.3f}  face lost {lost}")
    print("\n thr    FAR/attempt   FAR 3 tries   false-reject (augmented estimate)")
    for t in THRESHOLDS:
        far = float((imp >= t).mean())
        frr = (float((ag < t).sum()) + lost) / len(aug_gen)
        print(f" {t:.2f}   {far:.2e}     {1 - (1 - far) ** 3:.2e}      {frr:.3f}")

    t = float(np.quantile(imp, 1 - a.far)) + 1e-4 if len(imp) * a.far >= 5 else float(imp.max()) + 1e-4
    t = max(t, 0.35)  # never below this, however kind the data looks
    frr = (float((ag < t).sum()) + lost) / len(aug_gen)
    print(f"\nRECOMMENDED T_ACCEPT = {t:.4f}  (impostor accept <= {a.far:g}, floor 0.35)  est. false-reject {frr:.3f}")
    print("impostor-only: this is a lower bound; re-tune with data/scores.csv after a real week.")
    if a.write:
        cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        cfg["t_accept"] = round(t, 4)
        CONFIG_PATH.parent.mkdir(exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        print("wrote", CONFIG_PATH)


if __name__ == "__main__":
    main()
