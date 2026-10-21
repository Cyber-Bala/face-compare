"""Pick T_ACCEPT from real pairs.

Layout:  pairs/<student_id>/id.jpg  (ID photo)
         pairs/<student_id>/live*.jpg  (phone photos of that student, one or more)
Genuine = id vs own live. Impostor = id vs every other student's live.
Usage: python -m tools.tune_threshold pairs --far 0.001
Writes data/config.json with t_accept.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.verify import CONFIG_PATH, Engine, QualityError  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--far", type=float, default=0.001, help="target false-accept rate")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    eng = Engine()
    ids, lives = {}, {}
    for d in sorted(p for p in Path(a.folder).iterdir() if p.is_dir()):
        try:
            ids[d.name] = eng.embed_photo((d / "id.jpg").read_bytes())
        except (QualityError, FileNotFoundError) as e:
            print("skip id", d.name, e)
            continue
        for f in sorted(d.glob("live*")):
            try:
                lives.setdefault(d.name, []).append(eng.embed_photo(f.read_bytes()))
            except QualityError as e:
                print("skip live", f, e)
    gen, imp = [], []
    for s, e in ids.items():
        for o, vs in lives.items():
            for v in vs:
                (gen if o == s else imp).append(float(e @ v))
    if not gen or not imp:
        sys.exit("Need genuine and impostor pairs (>= 2 students).")
    gen, imp = np.array(gen), np.array(imp)
    # Smallest threshold whose impostor accept-rate <= FAR.
    t = float(np.quantile(imp, 1 - a.far)) + 1e-4 if a.far > 0 else float(imp.max()) + 1e-4
    frr = float((gen < t).mean())
    print(f"genuine n={len(gen)} mean={gen.mean():.3f} min={gen.min():.3f}")
    print(f"impostor n={len(imp)} mean={imp.mean():.3f} max={imp.max():.3f}")
    print(f"T_ACCEPT={t:.4f} at FAR<={a.far}  -> false-reject rate {frr:.3f}")
    if len(imp) * a.far < 5:
        print("WARNING: too few impostor pairs to trust this FAR. Collect more students.")
    if not a.no_write:
        cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        cfg["t_accept"] = round(t, 4)
        CONFIG_PATH.parent.mkdir(exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        print("wrote", CONFIG_PATH)


if __name__ == "__main__":
    main()
