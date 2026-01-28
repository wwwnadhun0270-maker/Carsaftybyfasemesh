import cv2
import time
import numpy as np
from collections import deque
import mediapipe as mp

# -------------------------
# Utilities
# -------------------------
def l2(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def ema(val, prev, alpha=0.15):
    return val if prev is None else alpha * val + (1 - alpha) * prev

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# -------------------------
# Landmarks (FIXED)
# p1-left, p2-upper1, p3-upper2, p4-right, p5-lower2, p6-lower1
# -------------------------
LEFT_EYE  = [33, 159, 158, 133, 153, 145]
RIGHT_EYE = [362, 386, 385, 263, 373, 374]  # fixed: correct corner is 263, not 282

LEFT_IRIS  = [468, 469, 470, 471]
RIGHT_IRIS = [472, 473, 474, 475]

LEFT_EYE_L, LEFT_EYE_R = 33, 133
RIGHT_EYE_L, RIGHT_EYE_R = 362, 263  # fixed with RIGHT_EYE
NOSE = 1

# -------------------------
# FaceMesh
# -------------------------
class FaceMeshDetector:
    def __init__(self):
        self.fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)
        if res.multi_face_landmarks:
            return [(lm.x, lm.y, lm.z) for lm in res.multi_face_landmarks[0].landmark]
        return None

# -------------------------
# Overlays
# -------------------------
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
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)

def draw_eye_gaze_arrows(img, lm, gaze_l, gaze_r, scale=50):
    h, w = img.shape[:2]

    def iris_center(idx):
        xs = [lm[i][0] * w for i in idx]
        ys = [lm[i][1] * h for i in idx]
        return int(np.mean(xs)), int(np.mean(ys))

    # Left eye arrow
    cx_l, cy_l = iris_center(LEFT_IRIS)
    dx_l = int((gaze_l - 0.5) * 2 * scale)
    cv2.arrowedLine(img, (cx_l, cy_l), (cx_l + dx_l, cy_l),
                    (0, 255, 255), 2, tipLength=0.3)

    # Right eye arrow
    cx_r, cy_r = iris_center(RIGHT_IRIS)
    dx_r = int((gaze_r - 0.5) * 2 * scale)
    cv2.arrowedLine(img, (cx_r, cy_r), (cx_r + dx_r, cy_r),
                    (0, 255, 255), 2, tipLength=0.3)

def draw_ratio_bar(img, value, pos, size, color, label):
    x, y = pos
    w, h = size

    value = clamp(value, 0.0, 1.0)
    fill_w = int(w * value)

    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)

    cv2.putText(
        img, label,
        (x, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

# -------------------------
# Driver Monitor
# -------------------------
class DriverMonitor:
    def __init__(self, fps=20):
        self.fps = fps
        self.sleep_frames = int(3 * fps)
        self.wake_frames = int(2 * fps)

        self.ear_ema = None
        self.gaze_ema = None

        self.ear_samples = []
        self.ear_enter = None
        self.ear_exit = None
        self.calib_frames = int(5 * fps)

        self.closed = 0
        self.opened = 0
        self.sleep_state = "CALIBRATING"

        self.dist_frames = 0
        self.dist_limit = int(2 * fps)
        self.focus_state = "UNKNOWN"

    def _ear_one(self, lm, eye):
        p1, p2, p3, p4, p5, p6 = [lm[i] for i in eye]
        v = l2(p2, p6) + l2(p3, p5)
        h = l2(p1, p4) + 1e-6
        return v / (2 * h)

    def compute_ear(self, lm):
        return (self._ear_one(lm, LEFT_EYE) + self._ear_one(lm, RIGHT_EYE)) / 2

    def gaze_ratio(self, lm, iris, l, r):
        ix = np.mean([lm[i][0] for i in iris])
        return clamp((ix - lm[l][0]) / (lm[r][0] - lm[l][0] + 1e-6), 0, 1)

    def update(self, lm):
        ear = self.compute_ear(lm)

        gaze_l_raw = self.gaze_ratio(lm, LEFT_IRIS, LEFT_EYE_L, LEFT_EYE_R)
        gaze_r_raw = self.gaze_ratio(lm, RIGHT_IRIS, RIGHT_EYE_L, RIGHT_EYE_R)
        gaze = (gaze_l_raw + gaze_r_raw) / 2

        self.ear_ema = ema(ear, self.ear_ema)
        self.gaze_ema = ema(gaze, self.gaze_ema)

        if self.sleep_state == "CALIBRATING":
            if self.ear_ema > 0.18:
                self.ear_samples.append(self.ear_ema)
            if len(self.ear_samples) >= self.calib_frames:
                mu, sd = np.mean(self.ear_samples), np.std(self.ear_samples)
                self.ear_enter = max(0.14, mu - 1.2 * sd)
                self.ear_exit = self.ear_enter + 0.02
                self.sleep_state = "AWAKE"
            return self._pack(gaze_l_raw, gaze_r_raw, 0.0, 0.0)

        closed = self.ear_ema < self.ear_enter if self.closed == 0 else self.ear_ema < self.ear_exit

        if closed:
            self.closed += 1
            self.opened = 0
        else:
            self.opened += 1
            self.closed = 0

        if self.sleep_state == "AWAKE" and self.closed >= self.sleep_frames:
            self.sleep_state = "ASLEEP"
        if self.sleep_state == "ASLEEP" and self.opened >= self.wake_frames:
            self.sleep_state = "AWAKE"

        dev = abs(self.gaze_ema - 0.5) * 2
        self.dist_frames = self.dist_frames + 1 if dev > 0.45 else 0
        self.focus_state = "DISTRACTED" if self.dist_frames >= self.dist_limit else "FOCUSED"

        eye_open_ratio = clamp(
            (self.ear_ema - self.ear_enter) /
            (self.ear_exit - self.ear_enter + 1e-6),
            0.0, 1.0
        )

        distraction_ratio = min(1.0, self.dist_frames / self.dist_limit)

        return self._pack(gaze_l_raw, gaze_r_raw, distraction_ratio, eye_open_ratio)

    def _pack(self, gaze_l, gaze_r, distraction_ratio, eye_open_ratio):
        return {
            "ear": self.ear_ema,
            "gaze": self.gaze_ema,
            "gaze_l": float(gaze_l),
            "gaze_r": float(gaze_r),
            "sleep_state": self.sleep_state,
            "focus_state": self.focus_state,
            "ear_enter": self.ear_enter,
            "ear_exit": self.ear_exit,
            "distraction_ratio": float(distraction_ratio),
            "eye_open_ratio": float(eye_open_ratio),
        }

# -------------------------
# Main
# -------------------------
def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    monitor = DriverMonitor()

    while True:
        ok, img = cap.read()
        if not ok:
            break

        img = cv2.resize(img, (640, int(img.shape[0] * 640 / img.shape[1])))
        lm = detector.detect(img)

        if lm:
            draw_eye_overlay(img, lm)
            info = monitor.update(lm)

            # per-eye arrows (fixes left-eye arrow error)
            draw_eye_gaze_arrows(img, lm, info["gaze_l"], info["gaze_r"])

            # centered distraction bar
            bar_w, bar_h = 220, 14
            bar_x = (img.shape[1] - bar_w) // 2
            bar_y = 140
            draw_ratio_bar(
                img,
                info["distraction_ratio"],
                pos=(bar_x, bar_y),
                size=(bar_w, bar_h),
                color=(0, 0, 255) if info["focus_state"] == "DISTRACTED" else (0, 180, 0),
                label="Distraction Level"
            )

            # centered eye-open bar
            if info["ear_enter"] is not None and info["ear_exit"] is not None:
                draw_ratio_bar(
                    img,
                    info["eye_open_ratio"],
                    pos=(bar_x, bar_y + 25),
                    size=(bar_w, bar_h),
                    color=(0, 255, 0),
                    label="Eye Open Ratio"
                )

            cv2.putText(img, f"SLEEP: {info['sleep_state']}", (420, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"FOCUS: {info['focus_state']}", (420, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Driver Monitor", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
