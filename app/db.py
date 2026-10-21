"""SQLite storage. Embeddings are Fernet-encrypted at rest."""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "faces.db"
KEY_PATH = DATA_DIR / "secret.key"


def _fernet():
    key = os.environ.get("EMBEDDING_KEY")
    if key:
        return Fernet(key.encode())
    DATA_DIR.mkdir(exist_ok=True)
    if not KEY_PATH.exists():
        KEY_PATH.write_bytes(Fernet.generate_key())
    return Fernet(KEY_PATH.read_bytes())


_F = None


def _f():
    global _F
    if _F is None:
        _F = _fernet()
    return _F


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    DATA_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init():
    with connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id  TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                enrolled_at TEXT NOT NULL,
                photo       BLOB
            );
            CREATE TABLE IF NOT EXISTS verifications (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT NOT NULL REFERENCES students(student_id),
                verified_at TEXT NOT NULL,
                score       REAL NOT NULL,
                status      TEXT NOT NULL
            );
            """
        )
        cols = [r["name"] for r in con.execute("PRAGMA table_info(students)")]
        if "photo" not in cols:
            con.execute("ALTER TABLE students ADD COLUMN photo BLOB")


def _enc(emb):
    return _f().encrypt(np.asarray(emb, dtype=np.float32).tobytes())


def _dec(blob):
    return np.frombuffer(_f().decrypt(blob), dtype=np.float32)


def add_student(student_id, name, emb, photo_jpeg):
    """Insert a new student. Returns False if the ID already exists (no replacement)."""
    with connect() as con:
        try:
            con.execute(
                "INSERT INTO students(student_id,name,embedding,enrolled_at,photo) VALUES(?,?,?,?,?)",
                (student_id, name, _enc(emb), _now(), _f().encrypt(photo_jpeg)),
            )
        except sqlite3.IntegrityError:
            return False
    return True


def delete_student(student_id):
    with connect() as con:
        con.execute("DELETE FROM verifications WHERE student_id=?", (student_id,))
        n = con.execute("DELETE FROM students WHERE student_id=?", (student_id,)).rowcount
    return n > 0


def get_photo(student_id):
    with connect() as con:
        r = con.execute("SELECT photo FROM students WHERE student_id=?", (student_id,)).fetchone()
    return _f().decrypt(r["photo"]) if r and r["photo"] else None


def get_student(student_id):
    with connect() as con:
        r = con.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
    if not r:
        return None
    return {"student_id": r["student_id"], "name": r["name"], "enrolled_at": r["enrolled_at"],
            "embedding": _dec(r["embedding"])}


def list_students():
    with connect() as con:
        rows = con.execute("SELECT student_id,name,enrolled_at FROM students ORDER BY student_id").fetchall()
    return [dict(r) for r in rows]


def add_verification(student_id, score):
    with connect() as con:
        con.execute(
            "INSERT INTO verifications(student_id,verified_at,score,status) VALUES(?,?,?,?)",
            (student_id, _now(), float(score), "VERIFIED"),
        )


def list_verifications(student_id=None, limit=200):
    q = "SELECT * FROM verifications"
    args = ()
    if student_id:
        q += " WHERE student_id=?"
        args = (student_id,)
    q += " ORDER BY id DESC LIMIT ?"
    with connect() as con:
        rows = con.execute(q, args + (limit,)).fetchall()
    return [dict(r) for r in rows]
