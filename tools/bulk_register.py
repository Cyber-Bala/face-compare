"""Extract face vectors from ID photos in bulk. Images are read, never stored.

Usage: python -m tools.bulk_register manifest.csv report.csv
manifest.csv header: roll,name,path
report.csv columns: roll,status,detail   (status: ok, enhanced, exists, failed)
Safe to run again: IDs already in the database are skipped.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import db  # noqa: E402
from app.verify import Engine, QualityError  # noqa: E402


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    db.init()
    eng = Engine()
    with open(sys.argv[1], newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    done = {s["student_id"] for s in db.list_students()}
    counts = {"ok": 0, "enhanced": 0, "exists": 0, "failed": 0}
    with open(sys.argv[2], "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["roll", "status", "detail"])
        for i, r in enumerate(rows, 1):
            roll = r["roll"].strip()
            if roll in done:
                status, detail = "exists", ""
            else:
                try:
                    emb, enhanced = eng.embed_id_photo(Path(r["path"]).read_bytes())
                    db.add_student(roll, r["name"].strip() or roll, emb)
                    status, detail = ("enhanced" if enhanced else "ok"), ""
                except (QualityError, OSError) as e:
                    status, detail = "failed", str(e)
            counts[status] += 1
            w.writerow([roll, status, detail])
            if i % 250 == 0:
                out.flush()
                print(i, "of", len(rows), counts, flush=True)
    print("finished", counts, flush=True)


if __name__ == "__main__":
    main()
