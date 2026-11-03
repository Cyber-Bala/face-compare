"""Security checks for the face service. Run: venv\Scripts\python.exe tests\security_test.py
Uses an in-process client and wrong-face photos only, so nothing is registered or verified for real."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402

import app.main as m  # noqa: E402
from app import db  # noqa: E402

fail = 0


def check(name, ok, extra=""):
    global fail
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {extra}"))
    fail += 0 if ok else 1


img = Path("tests/a.jpg").read_bytes()
photo = lambda: [("photos", ("a.jpg", img, "image/jpeg"))]  # noqa: E731

with TestClient(m.app, client=("203.0.113.10", 5000)) as attacker, TestClient(m.app, client=("198.51.100.7", 5000)) as student:
    sid = db.list_students()[0]["student_id"]
    tok = m._admin_token()

    for p in ("/api/admin/students", "/api/admin/exists/x", "/api/admin/verifications", "/api/admin/students/x/photo"):
        check(f"admin route {p} without token -> 401", attacker.get(p).status_code == 401)
        check(f"admin route {p} wrong token -> 401", attacker.get(p, headers={"X-Admin-Token": "nope"}).status_code == 401)
    check("admin register without token -> 401", attacker.post("/api/admin/register", data={"student_id": "x", "name": "y"}, files={"photo": ("a.jpg", img)}).status_code == 401)
    check("admin delete without token -> 401", attacker.delete(f"/api/admin/students/{sid}").status_code == 401)
    check("unknown student id -> 404", attacker.post("/api/verify", data={"student_id": "NOPE-ID"}, files=photo()).status_code == 404)

    # an attacker's wrong faces lock only the attacker's device, not the real student
    m.attempts.clear(); m.ip_failures.clear()
    seq = [attacker.post("/api/verify", data={"student_id": sid}, files=photo()).json()["status"] for _ in range(4)]
    check("attacker is locked after 3 wrong faces", seq[-1] == "GO_TO_ADMIN", seq)
    real = student.post("/api/verify", data={"student_id": sid}, files=photo()).json()
    check("the real student on another device is NOT locked out", real["status"] == "RETRY", real)

    # lock-out expires by itself
    old = m.LOCK_SECONDS
    m.LOCK_SECONDS = 1
    time.sleep(1.2)
    again = attacker.post("/api/verify", data={"student_id": sid}, files=photo()).json()
    check("lock-out expires after the lock time", again["status"] == "RETRY", again)
    m.LOCK_SECONDS = old

    # one device cannot keep guessing across many students
    m.attempts.clear(); m.ip_failures.clear()
    m.IP_FAIL_LIMIT = 4
    codes = []
    for s in db.list_students()[:6]:
        codes.append(attacker.post("/api/verify", data={"student_id": s["student_id"]}, files=photo()).status_code)
    check("one device is rate limited after too many failed matches (429)", 429 in codes, codes)
    m.IP_FAIL_LIMIT = 20
    m.attempts.clear(); m.ip_failures.clear()

    # upload limits
    big = b"\xff\xd8" + b"0" * (9 * 1024 * 1024)
    check("9 MB photo is refused (413)", attacker.post("/api/verify", data={"student_id": sid}, files=[("photos", ("b.jpg", big, "image/jpeg"))]).status_code == 413)
    many = [("photos", (f"{i}.jpg", img, "image/jpeg")) for i in range(11)]
    check("11 photos in one request is refused (413)", attacker.post("/api/verify", data={"student_id": sid}, files=many).status_code == 413)
    check("admin upload over 8 MB is refused (413)", attacker.post("/api/admin/register", headers={"X-Admin-Token": tok}, data={"student_id": "ZZ-NEW", "name": "t"}, files={"photo": ("b.jpg", big, "image/jpeg")}).status_code == 413)
    check("bad student id is refused (422)", attacker.post("/api/admin/register", headers={"X-Admin-Token": tok}, data={"student_id": "../etc", "name": "t"}, files={"photo": ("a.jpg", img)}).status_code == 422)
    check("not an image is refused with a message, no crash", attacker.post("/api/verify", data={"student_id": sid}, files=[("photos", ("x.jpg", b"not an image", "image/jpeg"))]).json()["status"] == "RETRY")

    r = attacker.get("/api/health")
    check("security headers present", r.headers.get("x-content-type-options") == "nosniff" and r.headers.get("x-frame-options") == "DENY" and r.headers.get("cache-control") == "no-store", dict(r.headers))
    check("no CORS headers for another website", "access-control-allow-origin" not in attacker.get("/api/health", headers={"Origin": "https://evil.example"}).headers)
    check("errors do not leak details", "Traceback" not in attacker.post("/api/verify", data={}).text)

print("FAILED" if fail else "ALL PASSED", fail)
sys.exit(1 if fail else 0)
