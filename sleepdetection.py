import cv2
import time
import numpy as np
from collections import deque
import mediapipe as mp

# =============================
# Utility
# =============================
def l2(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

def ema(x, prev, a):
    return x if prev is None else a * x + (1 - a) * prev

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# =============================
# FaceMesh Detector
# =============================
class FaceMeshDetector:
    def __init__(self):
        self.fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def find(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)
        faces = []
        if res.multi_face_landmarks:
            for f in res.multi_face_landmarks:
                faces.append([(lm.x, lm.y, lm.z) for lm in f.landmark])
        return img, faces

# =============================
# Minimal Eye + Iris Overlay
# =============================
# =============================
# Minimal Eye + Iris Overlay
# =============================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]  # ✅ Fixed: 5 landmarks, not 4
RIGHT_IRIS = [473, 474, 475, 476, 477]  # ✅ Fixed: 5 landmarks, not 4

def draw_eye_overlay(img, lm):
    h, w = img.shape[:2]

    def px(i):
        return int(lm[i][0] * w), int(lm[i][1] * h)

    for eye in (LEFT_EYE, RIGHT_EYE):
        for i in range(len(eye)):
            cv2.line(img, px(eye[i]), px(eye[(i + 1) % len(eye)]), (0, 255, 0), 1)

    for iris in (LEFT_IRIS, RIGHT_IRIS):
        pts = [px(i) for i in iris]
        for p in pts:
            cv2.circle(img, p, 2, (255, 0, 255), -1)
        cx = int(sum(p[0] for p in pts) / 5)  # ✅ Fixed: divide by 5, not 4
        cy = int(sum(p[1] for p in pts) / 5)  # ✅ Fixed: divide by 5, not 4
        cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)

# =============================
# Sleep Detector (Binary)
# =============================
class SleepDetector:
    def __init__(self, fps=20):
        self.fps = fps
        self.alpha = 0.12
        self.sleep_frames = int(3.5 * fps)
        self.wake_frames = int(2.0 * fps)
        self.reset()

    def reset(self):
        self.state = "CALIBRATING"
        self.ear_ema = None
        self.consec_closed = 0
        self.consec_open = 0
        self.calib_ear = []
        self.calibrated = False

    def ear(self, lm):
        def one(eye):
            p1, p2, p3, p4, p5, p6 = [lm[i] for i in eye]
            return (l2(p2, p6) + l2(p3, p5)) / (2 * l2(p1, p4) + 1e-6)
        return (one(LEFT_EYE) + one(RIGHT_EYE)) / 2

    def process(self, lm):
        ear = self.ear(lm)
        self.ear_ema = ema(ear, self.ear_ema, self.alpha)

        # ===== Calibration =====
        if not self.calibrated:
            if self.ear_ema > 0.18:
                self.calib_ear.append(self.ear_ema)

            if len(self.calib_ear) > self.fps * 5:
                mu = np.mean(self.calib_ear)
                sd = np.std(self.calib_ear)
                self.ear_th = max(0.15, mu - 1.2 * sd)
                self.calibrated = True
                self.state = "AWAKE"

            return self.state

        # ===== Eye closure =====
        closed = self.ear_ema < self.ear_th

        if closed:
            self.consec_closed += 1
            self.consec_open = 0
        else:
            self.consec_open += 1
            self.consec_closed = 0

        # ===== Binary FSM =====
        if self.state == "AWAKE":
            if self.consec_closed >= self.sleep_frames:
                self.state = "ASLEEP"

        elif self.state == "ASLEEP":
            if self.consec_open >= self.wake_frames:
                self.state = "AWAKE"

        return self.state

# =============================
# MAIN
# =============================
def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    sleep = SleepDetector(fps=20)

    while True:
        ok, img = cap.read()
        if not ok:
            break

        img, faces = detector.find(img)

        if faces:
            draw_eye_overlay(img, faces[0])
            state = sleep.process(faces[0])
        else:
            state = "NO FACE"

        color = (0, 255, 0) if state == "AWAKE" else (0, 0, 255)

        cv2.putText(img, f"STATE: {state}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow("Binary Sleep Detection (AWAKE / ASLEEP)", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
