import cv2
import mediapipe as mp
import threading
import pygame
import config
from utils import calculate_EAR, calculate_MAR

# Initialize pygame mixer
pygame.mixer.init()

def play_alarm():
    pygame.mixer.music.load("alarm.wav")
    pygame.mixer.music.play(-1)   # Loop continuously

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

frame_counter = 0
drowsy_score = 0
status = "SAFE"

startup_frames = 30   # Prevent false detection at start

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if startup_frames > 0:
        startup_frames -= 1
        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Eye landmarks
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            left_eye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]

            right_eye = [(int(face_landmarks.landmark[i].x * w),
                          int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]

            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)

            EAR = (left_EAR + right_EAR) / 2.0

            # Mouth landmarks
            mouth_indices = [78, 308, 14, 13, 82, 312, 317, 87]
            mouth = [(int(face_landmarks.landmark[i].x * w),
                      int(face_landmarks.landmark[i].y * h)) for i in mouth_indices]

            MAR = calculate_MAR(mouth)

            # ----------- Detection Logic -----------

            if EAR < config.EAR_THRESHOLD:
                frame_counter += 1
                drowsy_score += 1
            else:
                frame_counter = 0
                drowsy_score = max(0, drowsy_score - 1)

            if MAR > config.MAR_THRESHOLD:
                drowsy_score += 2

            # ----------- Status Decision -----------

            if frame_counter > config.FRAME_THRESHOLD or drowsy_score > config.DROWSY_SCORE_LIMIT:
                status = "DROWSY"

                if not pygame.mixer.music.get_busy():
                    play_alarm()

            elif drowsy_score > config.WARNING_SCORE_LIMIT:
                status = "WARNING"
                pygame.mixer.music.stop()

            else:
                status = "SAFE"
                pygame.mixer.music.stop()

            # ----------- Display -----------

            cv2.putText(frame, f"EAR: {EAR:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"MAR: {MAR:.2f}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Score: {drowsy_score}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if status == "SAFE":
                color = (0, 255, 0)
            elif status == "WARNING":
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.putText(frame, status, (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()