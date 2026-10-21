# Face Compare

Face verification service for the REC college gym app. Matches a live face against the student's registered ID photo.

- Detection and alignment: YuNet (OpenCV), 5-point alignment to 112x112
- Embedding: AdaFace IR-101 (ONNX), 512-d, cosine similarity
- API: FastAPI. Gym backend service on `http://127.0.0.1:8001`, phone page on `https://<LAN-IP>:8443`
- Details: see [FACE_VERIFICATION_PORTAL.md](FACE_VERIFICATION_PORTAL.md)

## Run (Windows)

```bat
start.bat           :: first time: creates venv, installs packages, downloads models (~260 MB), starts the service
start_service.bat   :: later runs: service only on http://127.0.0.1:8001
```

On first start `data/admin.token` is created. Put it in the gym app `.env` as `FACE_ML_TOKEN`
(and `FACE_ML_URL=http://127.0.0.1:8001`).

## Match settings

Defaults are in `app/verify.py`. Override in `data/config.json`.

| Setting | Default | Meaning |
|---------|---------|---------|
| `t_accept` | `0.50` | Minimum cosine score to accept. Calibrated on about 3,500 ID photos: a wrong person is accepted about 2 times in 100,000 attempts. Do not go below 0.45. |
| `min_good_frames` | `2` | Live frames that must pass the quality gate. Every usable frame must reach `t_accept`. |
| `max_attempts` | `3` | Failed matches before the student is sent to the admin. |
| quality gate | see file | Blurry, shaky, dark, tiny or tilted photos are rejected before matching. |

## Tools (run from this folder)

```bat
venv\Scripts\python.exe -m tools.bulk_register photos
venv\Scripts\python.exe -m tools.calibrate_single photos
```

## Not in this repository

`venv/`, `models/*.onnx*`, and everything in `data/` (student face database, admin token, keys, scores) are git-ignored.
Never commit student photos or `data/`.
