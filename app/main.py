"""FastAPI backend.

Run: python run.py
Admin endpoints need header  X-Admin-Token: <ADMIN_TOKEN env or data/admin.token>.
"""
import os
import re
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import db
from .verify import Engine, QualityError, thumbnail

ROOT = Path(__file__).resolve().parent.parent
FAIL_MSG = "Verification failed. Please meet the admin to check your photo and re-register your face."
ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")

app = FastAPI(title="Face Verification Portal")
engine = None
attempts: dict[str, int] = {}  # failed attempts per student (in memory)


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


# ---------------- verification: upload a photo, get the result ----------------
@app.post("/api/verify")
async def verify(student_id: str = Form(...), photos: list[UploadFile] = File(...)):
    student_id = student_id.strip()
    s = db.get_student(student_id)
    if not s:
        raise HTTPException(404, "Student ID not registered.")
    max_a = engine.cfg["max_attempts"]
    if attempts.get(student_id, 0) >= max_a:
        return {"status": "GO_TO_ADMIN", "message": FAIL_MSG}
    try:
        score = engine.match_frames([await f.read() for f in photos], s["embedding"])
    except QualityError as e:
        return {"status": "RETRY", "message": str(e)}  # unusable photo: not counted as a failed match
    _log(student_id, score)
    if engine.accept(score):
        attempts.pop(student_id, None)
        db.add_verification(student_id, score)
        return {"status": "VERIFIED", "name": s["name"], "score": round(score, 4)}
    attempts[student_id] = attempts.get(student_id, 0) + 1
    if attempts[student_id] >= max_a:
        return {"status": "GO_TO_ADMIN", "message": FAIL_MSG, "score": round(score, 3)}
    return {"status": "RETRY", "message": "Face did not match. Try again in good light, facing the camera.",
            "score": round(score, 3), "threshold": engine.cfg["t_accept"],
            "attempts_left": max_a - attempts[student_id]}


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
    data = await photo.read()
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
        score = engine.match_frames([await f.read() for f in photos], s["embedding"])
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
    attempts.pop(student_id, None)
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
