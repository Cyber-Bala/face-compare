"""Bulk-register ID photos.

Usage: python -m tools.bulk_register photos [students.csv]
Photos: <student_id>.jpg|png  (file name = student ID).
Names: optional CSV with header  student_id,name  (else name = student ID).
Existing IDs are skipped (delete in Admin first). Failed photos are listed at the end.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import db  # noqa: E402
from app.verify import Engine, QualityError, thumbnail  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1])
    names = {}
    if len(sys.argv) > 2:
        with open(sys.argv[2], newline="", encoding="utf-8-sig") as f:
            names = {r["student_id"].strip(): r["name"].strip() for r in csv.DictReader(f)}
    db.init()
    eng = Engine()
    ok, bad = 0, []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        sid = p.stem
        try:
            data = p.read_bytes()
            emb = eng.embed_photo(data)
            added = db.add_student(sid, names.get(sid, sid), emb, thumbnail(data))
        except QualityError as e:
            bad.append((p.name, str(e)))
            continue
        if added:
            ok += 1
        else:
            bad.append((p.name, "ID already registered, skipped"))
    print(f"registered {ok}")
    for n, m in bad:
        print("FAILED", n, "-", m)


if __name__ == "__main__":
    main()
