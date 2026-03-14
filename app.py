from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
from utils import calculate_EAR, calculate_MAR
import config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

frame_counter = 0
drowsy_score = 0

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global frame_counter, drowsy_score

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "invalid image"}

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {
            "ear": 0.0,
            "mar": 0.0,
            "frame_counter": frame_counter,
            "drowsy_score": drowsy_score,
            "status": "NO FACE"
        }

    face = results.multi_face_landmarks[0]
    h, w, _ = img.shape

    left_eye_idx  = [33, 160, 158, 133, 153, 144]
    right_eye_idx = [362, 385, 387, 263, 373, 380]
    mouth_idx     = [78, 308, 14, 13, 82, 312, 317, 87]

    left_eye  = [(int(face.landmark[i].x * w),
                  int(face.landmark[i].y * h))
                  for i in left_eye_idx]
    right_eye = [(int(face.landmark[i].x * w),
                  int(face.landmark[i].y * h))
                  for i in right_eye_idx]
    mouth     = [(int(face.landmark[i].x * w),
                  int(face.landmark[i].y * h))
                  for i in mouth_idx]

    left_EAR  = calculate_EAR(left_eye)
    right_EAR = calculate_EAR(right_eye)
    EAR = (left_EAR + right_EAR) / 2.0
    MAR = calculate_MAR(mouth)

    if EAR < config.EAR_THRESHOLD:
        frame_counter += 1
        drowsy_score += 1
    else:
        frame_counter = 0
        drowsy_score = max(0, drowsy_score - 1)

    if MAR > config.MAR_THRESHOLD:
        drowsy_score += 2

    if (frame_counter > config.FRAME_THRESHOLD or
            drowsy_score > config.DROWSY_SCORE_LIMIT):
        status = "DROWSY"
    elif drowsy_score > config.WARNING_SCORE_LIMIT:
        status = "WARNING"
    else:
        status = "SAFE"

    return {
        "ear": round(EAR, 2),
        "mar": round(MAR, 2),
        "frame_counter": frame_counter,
        "drowsy_score": drowsy_score,
        "status": status
    }

