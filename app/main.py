"""FastAPI backend.

Run: python run.py
Admin endpoints need header  X-Admin-Token: <ADMIN_TOKEN env or data/admin.token>.
"""
import os
import re
import secrets
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import db
from .verify import Engine, QualityError, thumbnail

ROOT = Path(__file__).resolve().parent.parent
FAIL_MSG = "Verification failed. Please meet the admin to check your photo and re-register your face."
ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")

app = FastAPI(title="Face Verification Portal")
engine = None

MAX_PHOTO_BYTES = 8 * 1024 * 1024  # one photo
MAX_PHOTOS = 10                    # photos per request
LOCK_SECONDS = 30 * 60             # how long a student+device lock-out lasts
IP_FAIL_LIMIT, IP_FAIL_WINDOW = 20, 15 * 60  # failed matches allowed per device per 15 minutes

# Failed matches are counted per student AND per client address, and they expire. Counting per student only
# let anyone on the internet lock any student out just by sending wrong faces for their ID.
attempts: dict[tuple[str, str], tuple[int, float]] = {}  # (student, ip) -> (failures, first failure time)
ip_failures: dict[str, list[float]] = {}                  # ip -> failure times


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _fails(student_id: str, ip: str) -> int:
    n, first = attempts.get((student_id, ip), (0, 0.0))
    if n and time.time() - first > LOCK_SECONDS:
        attempts.pop((student_id, ip), None)
        return 0
    return n


def _record_fail(student_id: str, ip: str) -> int:
    now = time.time()
    n, first = attempts.get((student_id, ip), (0, now))
    if n and now - first > LOCK_SECONDS:
        n, first = 0, now
    attempts[(student_id, ip)] = (n + 1, first if n else now)
    recent = [t for t in ip_failures.get(ip, []) if now - t < IP_FAIL_WINDOW]
    recent.append(now)
    ip_failures[ip] = recent
    if len(attempts) > 5000:  # keep memory bounded
        for k in [k for k, (_, t) in attempts.items() if now - t > LOCK_SECONDS]:
            attempts.pop(k, None)
    return n + 1


def _ip_blocked(ip: str) -> bool:
    now = time.time()
    return len([t for t in ip_failures.get(ip, []) if now - t < IP_FAIL_WINDOW]) >= IP_FAIL_LIMIT


def _clear_attempts(student_id: str) -> None:
    for k in [k for k in attempts if k[0] == student_id]:
        attempts.pop(k, None)


async def _read_photo(f: UploadFile) -> bytes:
    data = await f.read(MAX_PHOTO_BYTES + 1)
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(413, "Photo is too large (max 8 MB).")
    return data


@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


def _admin_token():
    t = os.environ.get("ADMIN_TOKEN")
    if t:
        return t
    p = ROOT / "data" / "admin.token"
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        p.write_text(secrets.token_urlsafe(16))
    return p.read_text().strip()


def require_admin(x_admin_token: str = Header(default="")):
    if not secrets.compare_digest(x_admin_token.encode(), _admin_token().encode()):
        raise HTTPException(401, "Wrong admin token.")


@app.on_event("startup")
def _startup():
    global engine
    db.init()
    if engine is None:  # shared when two servers run in one process
        engine = Engine()
    print("ADMIN TOKEN:", _admin_token(), "(file data/admin.token)")


def _log(student_id, score):
    """Every match score (not images) goes to data/scores.csv, to tune the threshold later."""
    p = ROOT / "data" / "scores.csv"
    new = not p.exists()
    with open(p, "a", encoding="utf-8") as f:
        if new:
            f.write("time,student_id,score,accepted\n")
        f.write(f"{db._now()},{student_id},{score:.4f},{int(engine.accept(score))}\n")


@app.get("/api/health")
def health():
    return {"ok": True, "students": len(db.list_students())}


# ---------------- photo check: is this capture usable? (no matching, nothing stored) ----------------
CHECK_PER_MINUTE = 90
_check_hits: dict[str, list[float]] = {}


@app.post("/api/check")
async def check_photo(request: Request, photo: UploadFile = File(...)):
    ip = _client_ip(request)
    now = time.time()
    recent = [t for t in _check_hits.get(ip, []) if now - t < 60]
    if len(recent) >= CHECK_PER_MINUTE:
        raise HTTPException(429, "Too many photo checks. Wait a moment.")
    recent.append(now)
    _check_hits[ip] = recent
    result = engine.check_photo(await _read_photo(photo))
    if not result["ok"]:
        result["message"] = "Please retake the photo. " + result["message"]
    return result


# ---------------- verification: upload a photo, get the result ----------------
@app.post("/api/verify")
async def verify(request: Request, student_id: str = Form(...), photos: list[UploadFile] = File(...)):
    ip = _client_ip(request)
    if _ip_blocked(ip):
        raise HTTPException(429, "Too many failed attempts from this device. Try again later or meet the admin.")
    if len(photos) > MAX_PHOTOS:
        raise HTTPException(413, f"Send at most {MAX_PHOTOS} photos.")
    student_id = student_id.strip()
    s = db.get_student(student_id)
    if not s:
        raise HTTPException(404, "Student ID not registered.")
    max_a = engine.cfg["max_attempts"]
    if _fails(student_id, ip) >= max_a:
        return {"status": "GO_TO_ADMIN", "message": FAIL_MSG}
    try:
        score = engine.match_frames([await _read_photo(f) for f in photos], s["embedding"])
    except QualityError as e:
        return {"status": "RETRY", "message": str(e)}  # unusable photo: not counted as a failed match
    _log(student_id, score)
    if engine.accept(score):
        attempts.pop((student_id, ip), None)
        db.add_verification(student_id, score)
        return {"status": "VERIFIED", "name": s["name"], "score": round(score, 4)}
    failures = _record_fail(student_id, ip)
    if failures >= max_a:
        return {"status": "GO_TO_ADMIN", "message": FAIL_MSG, "score": round(score, 3)}
    return {"status": "RETRY", "message": "Face did not match. Try again in good light, facing the camera.",
            "score": round(score, 3), "threshold": engine.cfg["t_accept"],
            "attempts_left": max_a - failures}


# ---------------- admin ----------------
@app.post("/api/admin/register", dependencies=[Depends(require_admin)])
async def register(student_id: str = Form(...), name: str = Form(...), photo: UploadFile = File(...)):
    student_id, name = student_id.strip(), name.strip()
    if not ID_RE.match(student_id):
        raise HTTPException(422, "Student ID: letters, digits, _ . - only.")
    if not name:
        raise HTTPException(422, "Name required.")
    if db.get_student(student_id):
        raise HTTPException(409, "Student ID already registered. Delete it first to register again.")
    data = await _read_photo(photo)
    try:
        emb = engine.embed_photo(data)
        thumb = thumbnail(data)
    except QualityError as e:
        raise HTTPException(422, str(e))
    if not db.add_student(student_id, name, emb, thumb):
        raise HTTPException(409, "Student ID already registered.")
    return {"student_id": student_id}


@app.get("/api/admin/exists/{student_id}", dependencies=[Depends(require_admin)])
def exists(student_id: str):
    s = db.get_student(student_id)
    return {"registered": bool(s), "name": s["name"] if s else None}


@app.post("/api/admin/match", dependencies=[Depends(require_admin)])
async def match(student_id: str = Form(...), photos: list[UploadFile] = File(...)):
    """Admin camera capture vs the stored photo. Nothing is saved."""
    s = db.get_student(student_id)
    if not s:
        raise HTTPException(404, "No photo registered for this student.")
    try:
        score = engine.match_frames([await _read_photo(f) for f in photos], s["embedding"])
    except QualityError as e:
        raise HTTPException(422, str(e))
    return {"match": engine.accept(score), "score": round(score, 4), "threshold": engine.cfg["t_accept"]}


@app.get("/api/admin/students", dependencies=[Depends(require_admin)])
def students():
    return db.list_students()


@app.get("/api/admin/students/{student_id}/photo", dependencies=[Depends(require_admin)])
def student_photo(student_id: str):
    p = db.get_photo(student_id)
    if not p:
        raise HTTPException(404, "No photo.")
    return Response(p, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.delete("/api/admin/students/{student_id}", dependencies=[Depends(require_admin)])
def delete_student(student_id: str):
    if not db.delete_student(student_id):
        raise HTTPException(404, "Student ID not found.")
    _clear_attempts(student_id)
    return {"deleted": student_id}


@app.get("/api/admin/verifications", dependencies=[Depends(require_admin)])
def verifications(student_id: str | None = None):
    return db.list_verifications(student_id)


@app.exception_handler(Exception)
async def _err(_, exc):
    return JSONResponse({"detail": f"Server error: {type(exc).__name__}"}, status_code=500)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
