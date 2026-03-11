import cv2
import mediapipe as mp
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import config
from utils import calculate_EAR, calculate_MAR

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

st.title("Driver Drowsiness Detection")
st.write("This app detects drowsiness in real-time using your webcam.")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_counter = 0
        self.drowsy_score = 0
        self.status = "SAFE"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = img.shape
                
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                mouth_indices = [78, 308, 14, 13, 82, 312, 317, 87]
                
                left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
                right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]
                mouth = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in mouth_indices]
                
                left_EAR = calculate_EAR(left_eye)
                right_EAR = calculate_EAR(right_eye)
                EAR = (left_EAR + right_EAR) / 2.0
                MAR = calculate_MAR(mouth)

                # Drowsiness Logic
                if EAR < config.EAR_THRESHOLD:
                    self.frame_counter += 1
                    self.drowsy_score += 1
                else:
                    self.frame_counter = 0
                    self.drowsy_score = max(0, self.drowsy_score - 1)

                if MAR > config.MAR_THRESHOLD:
                    self.drowsy_score += 2

                # Status Decision
                if self.frame_counter > config.FRAME_THRESHOLD or self.drowsy_score > config.DROWSY_SCORE_LIMIT:
                    self.status = "DROWSY"
                elif self.drowsy_score > config.WARNING_SCORE_LIMIT:
                    self.status = "WARNING"
                else:
                    self.status = "SAFE"
                    
                # Drawing details on frame
                cv2.putText(img, f"EAR: {EAR:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f"MAR: {MAR:.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f"Score: {self.drowsy_score}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                color = (0, 255, 0) if self.status == "SAFE" else ((0, 255, 255) if self.status == "WARNING" else (0, 0, 255))
                cv2.putText(img, self.status, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
