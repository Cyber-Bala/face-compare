# Face Verification Portal

Verifies that a live phone-captured face matches the student's registered ID photo.
Two outcomes only: **VERIFIED** or **GO TO ADMIN**.

## 1. How it works

A face is converted into 512 numbers (a face vector). The same person gives vectors that point in nearly the same direction, and different people give vectors that point differently. Verification measures that angle (cosine similarity).

## 2. Pipeline

1. **Detect**: YuNet finds the face box and 5 landmarks (eyes, nose tip, mouth corners).
2. **Align**: rotate and scale so the landmarks sit at fixed positions, crop to 112x112.
3. **Normalize pixels**: `(pixel/255 - 0.5) / 0.5`, BGR order, giving values in -1 to 1.
4. **Embed**: AdaFace IR-101 outputs 512 numbers.
5. **Normalize vector**: divide by its length so it has length 1.
6. **Compare**: `score = A . B` (cosine similarity, 1 = same direction, 0 = unrelated).
7. **Decide**: `score >= T_ACCEPT` -> VERIFIED, else -> GO TO ADMIN.

Averaging 5 to 8 live frames reduces noise. **Re-normalize the averaged vector to length 1.**

## 3. Workflow

### Registration (once per student)
ID photo -> YuNet -> align -> preprocess -> AdaFace -> normalize -> store embedding with student ID

### Verification
1. Student enters / selects their ID (1:1 match, safer than searching all students)
2. Phone camera capture (camera only, no gallery upload)
3. Quality gate (one face, big enough, not blurry, good lighting, frontal pose)
4. Liveness check (anti-spoofing model or blink / head-turn challenge)
5. YuNet -> align -> preprocess -> AdaFace -> average frames -> normalize
6. Cosine score against the stored embedding
7. `score >= T_ACCEPT`: save a row in `verifications` with status VERIFIED
8. Otherwise: save nothing and show
   *"Verification failed. Please meet the admin to check your photo and re-register your face."*
   (allow 1 or 2 retries first)

### Admin re-registration
Admin captures a fresh photo, and checks with the old images and tell it is same person or different person 

## 4. Database

`students`
- student_id (primary key)
- name
- embedding (512 floats, stored as a BLOB, encrypted at rest)
- enrolled_at

`verifications`
- id
- student_id
- verified_at
- score
- status (VERIFIED)

## 5. Thresholds

Do not trust generic numbers. ID photo vs live phone photo is a harder comparison, so scores will be lower than same-camera pairs.
Collect genuine pairs and impostor pairs from your own students, then choose `T_ACCEPT` at a target false-accept rate.

## 6. Requirements and caveats

- **Liveness:** AdaFace does not detect spoofing. Add a separate anti-spoofing step.
- **Privacy:** embeddings are biometric data. Encrypt them, restrict access, and get consent (India DPDP Act 2023).
- **Camera on phone:** browsers allow camera access only on HTTPS or localhost, so use HTTPS when testing from a phone.
- **Licensing:** check the license of the pretrained AdaFace weights before commercial use.

## 7. Tech stack

- Python 3.10 to 3.12
- OpenCV (YuNet via `cv2.FaceDetectorYN`)
- ONNX Runtime (inference)
- NumPy
- FastAPI + Uvicorn (backend)
- SQLite (database)
- Simple web page for phone camera capture

## 8. Windows setup

### 8.1 Download and install
1. **Python 3.11 (64-bit)** from https://www.python.org/downloads/windows/ . Tick **"Add python.exe to PATH"** during install.
2. **Git for Windows** from https://git-scm.com/download/win
3. **Antigravity** (already installed)

### 8.2 Project folder and virtual environment (PowerShell)

```powershell
mkdir $HOME\face_verification
cd $HOME\face_verification
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install opencv-python onnxruntime numpy fastapi uvicorn python-multipart
```

If activation is blocked, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 8.3 Verify install

```powershell
python -c "import cv2, onnxruntime, numpy, fastapi; print(cv2.__version__, onnxruntime.__version__, numpy.__version__, fastapi.__version__)"
```

### 8.4 Model files

Create the folder and download YuNet:

```powershell
mkdir models
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models\face_detection_yunet_2023mar.onnx"
```

AdaFace IR-101: the pretrained checkpoints are linked from the official repository README (https://github.com/mk-minchul/AdaFace). They are PyTorch `.ckpt` files, so they must be converted to ONNX once (needs PyTorch temporarily). Put the final file at `models\adaface_ir101.onnx`.

## 9. Folder structure

```
face_verification/
  venv/
  models/
    face_detection_yunet_2023mar.onnx
    adaface_ir101.onnx
  app/
    detect_align.py
    embed.py
    verify.py
    db.py
    main.py
  static/
    index.html
  data/
    faces.db
  FACE_VERIFICATION_PORTAL.md
```

## 10. Build order

1. Environment check (8.3)
2. YuNet detection on one image
3. Alignment to 112x112
4. AdaFace ONNX conversion and embedding
5. Cosine score on one ID photo + one live photo pair
6. SQLite storage and registration
7. FastAPI endpoints (register, verify)
8. Phone camera page
9. Quality gate and liveness
10. Threshold tuning with real pairs
