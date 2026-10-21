import cv2, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from app import main
P = r"C:\Users\CODERM~1\AppData\Local\Temp\claude\d--Clg-Project-REC-Gym-face-ML\1833aca7-6edd-40cc-9f5d-06c47d65c99c\images\2.jpg"
idp = Path(P).read_bytes(); lena = Path("tests/a.jpg").read_bytes()
def cam(data, i):  # simulate webcam: smaller, noisy, shifted light
    im = cv2.imdecode(np.frombuffer(data, np.uint8), 1); im = cv2.resize(im, None, fx=.4, fy=.4)
    im = np.clip(im.astype(np.float32) * (0.8 + .05*i) + np.random.randn(*im.shape) * 8, 0, 255).astype(np.uint8)
    return cv2.imencode(".jpg", im, [1, 45])[1].tobytes()
with TestClient(main.app) as c:
    T = {"X-Admin-Token": main._admin_token()}
    c.delete("/api/admin/students/TMP", headers=T)
    print(c.post("/api/admin/register", headers=T, data={"student_id": "TMP", "name": "T"}, files={"photo": ("a.jpg", idp)}).json())
    f = lambda d: [("photos", (f"{i}.jpg", cam(d, i))) for i in range(5)]
    print("genuine ", c.post("/api/verify", data={"student_id": "TMP"}, files=f(idp)).json())
    print("impostor", c.post("/api/verify", data={"student_id": "TMP"}, files=f(lena)).json())
    c.delete("/api/admin/students/TMP", headers=T)
