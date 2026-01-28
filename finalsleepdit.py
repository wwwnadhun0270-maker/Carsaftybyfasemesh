"""
Driver Monitor (Production-style minimal)
- Binary sleep: AWAKE / ASLEEP (no drowsy state)
- Distraction: FOCUSED / DISTRACTED (gaze + head yaw proxy)
- Minimal eye + iris overlay
- Improved, more anatomically accurate EAR landmark sets
- EAR hysteresis (enter/exit) + calibration
- Optional gaze arrow overlay
Controls:
  q = quit
  r = recalibrate
  d = toggle landmark IDs
  a = toggle gaze arrow

Install:
  pip install opencv-python mediapipe numpy
"""

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
# Landmarks (improved EAR sets)
# EAR indexing order: [p1, p2, p3, p4, p5, p6]
# where p1-p4 are horizontal corners, (p2,p6) and (p3,p5) are vertical pairs.
# -------------------------
LEFT_EYE  = [33, 159, 158, 133, 153, 145]          # improved eyelid-centered set
RIGHT_EYE = [362, 386, 385, 263, 373, 374]         # improved eyelid-centered set (more accurate than 362,385,387,263,373,380)

# Iris (requires refine_landmarks=True)
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [472, 473, 474, 475]

# For gaze ratio (horizontal)
LEFT_EYE_L = 33
LEFT_EYE_R = 133
RIGHT_EYE_L = 362
RIGHT_EYE_R = 263

# Head yaw proxy points
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
# Minimal overlay
# -------------------------
def draw_eye_overlay(img, lm, show_ids=False):
    h, w = img.shape[:2]

    def px(i):
        return int(lm[i][0] * w), int(lm[i][1] * h)

    # Eye contour from our 6 points (connect them in order for a quick shape)
    for eye in (LEFT_EYE, RIGHT_EYE):
        for i in range(len(eye)):
            cv2.line(img, px(eye[i]), px(eye[(i + 1) % len(eye)]), (0, 255, 0), 1)

    # Iris points + center
    for iris in (LEFT_IRIS, RIGHT_IRIS):
        pts = [px(i) for i in iris]
        for p in pts:
            cv2.circle(img, p, 2, (255, 0, 255), -1)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)

    if show_ids:
        ids = sorted(set(LEFT_EYE + RIGHT_EYE + LEFT_IRIS + RIGHT_IRIS + [LEFT_EYE_L, LEFT_EYE_R, RIGHT_EYE_L, RIGHT_EYE_R, NOSE]))
        for i in ids:
            x, y = px(i)
            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)

def iris_center_px(img, lm, iris_idx):
    h, w = img.shape[:2]
    xs = [lm[i][0] for i in iris_idx]
    ys = [lm[i][1] for i in iris_idx]
    return int(np.mean(xs) * w), int(np.mean(ys) * h)

def draw_gaze_arrow(img, lm, gaze_ratio, scale_px=80):
    # arrow from average iris center pointing left/right based on gaze_ratio
    lc = iris_center_px(img, lm, LEFT_IRIS)
    rc = iris_center_px(img, lm, RIGHT_IRIS)
    cx = int((lc[0] + rc[0]) / 2)
    cy = int((lc[1] + rc[1]) / 2)
    dx = int((gaze_ratio - 0.5) * 2.0 * scale_px)  # gaze_ratio: 0..1
    cv2.arrowedLine(img, (cx, cy), (cx + dx, cy), (0, 255, 255), 2, tipLength=0.25)

# -------------------------
# Driver Monitor
# -------------------------
class DriverMonitor:
    def __init__(self, fps=20):
        self.fps = int(max(10, fps))

        # Sleep FSM (binary)
        self.sleep_frames = int(3.0 * self.fps)
        self.wake_frames = int(2.0 * self.fps)

        # Smoothing
        self.ear_ema = None
        self.gaze_ema = None
        self.yaw_ema = None
        self.alpha = 0.15

        # Calibration
        self.ear_samples = []
        self.calib_frames = int(5 * self.fps)
        self.ear_enter = None
        self.ear_exit = None
        self.sleep_state = "CALIBRATING"

        # Counters
        self.closed_cnt = 0
        self.open_cnt = 0

        # Distraction
        self.dist_frames = 0
        self.dist_limit = int(2 * self.fps)
        self.focus_state = "UNKNOWN"

        # 10s PERCLOS (for display only)
        self.perclos = deque(maxlen=int(10 * self.fps))

        # UI toggles
        self.debug_ids = False
        self.show_gaze_arrow = True

    # -------------------------
    # Features
    # -------------------------
    def _ear_one(self, lm, eye):
        p1, p2, p3, p4, p5, p6 = [lm[i] for i in eye]
        v = l2(p2, p6) + l2(p3, p5)
        h = l2(p1, p4) + 1e-6
        return v / (2.0 * h)

    def compute_ear(self, lm):
        return (self._ear_one(lm, LEFT_EYE) + self._ear_one(lm, RIGHT_EYE)) / 2.0

    def _gaze_ratio(self, lm, iris, l, r):
        ix = float(np.mean([lm[i][0] for i in iris]))
        denom = (lm[r][0] - lm[l][0]) + 1e-6
        return clamp((ix - lm[l][0]) / denom, 0.0, 1.0)

    def compute_gaze(self, lm):
        return (
            self._gaze_ratio(lm, LEFT_IRIS, LEFT_EYE_L, LEFT_EYE_R) +
            self._gaze_ratio(lm, RIGHT_IRIS, RIGHT_EYE_L, RIGHT_EYE_R)
        ) / 2.0

    def compute_head_yaw_proxy(self, lm):
        # nose x relative to eye-corner midpoint, normalized by eye-corner width
        lx = lm[LEFT_EYE_L][0]
        rx = lm[RIGHT_EYE_R][0]
        cx = (lx + rx) / 2.0
        denom = (rx - lx) + 1e-6
        return clamp((lm[NOSE][0] - cx) / denom, -0.5, 0.5)

    # -------------------------
    # Reset / recalibrate
    # -------------------------
    def reset(self):
        self.ear_ema = None
        self.gaze_ema = None
        self.yaw_ema = None

        self.ear_samples = []
        self.ear_enter = None
        self.ear_exit = None

        self.sleep_state = "CALIBRATING"
        self.closed_cnt = 0
        self.open_cnt = 0

        self.dist_frames = 0
        self.focus_state = "UNKNOWN"

        self.perclos.clear()

    # -------------------------
    # Update
    # -------------------------
    def update(self, lm):
        ear_raw = self.compute_ear(lm)
        gaze_raw = self.compute_gaze(lm)
        yaw_raw = self.compute_head_yaw_proxy(lm)

        self.ear_ema = ema(ear_raw, self.ear_ema, self.alpha)
        self.gaze_ema = ema(gaze_raw, self.gaze_ema, self.alpha)
        self.yaw_ema = ema(yaw_raw, self.yaw_ema, self.alpha)

        # ---------- calibration ----------
        if self.sleep_state == "CALIBRATING":
            # Accept stable open-eye frames: ear high and gaze not extreme
            if self.ear_ema is not None and self.ear_ema > 0.18 and 0.25 < self.gaze_ema < 0.75:
                self.ear_samples.append(self.ear_ema)

            if len(self.ear_samples) >= self.calib_frames:
                mu = float(np.mean(self.ear_samples))
                sd = float(np.std(self.ear_samples) + 1e-6)

                # Hysteresis thresholds
                self.ear_enter = max(0.14, mu - 1.2 * sd)                 # close when below this
                self.ear_exit = max(self.ear_enter + 0.02, mu - 0.6 * sd) # reopen when above this
                self.sleep_state = "AWAKE"

            return self._pack(perclos_10s=0.0)

        # ---------- sleep (binary) with hysteresis ----------
        if self.closed_cnt > 0:
            closed = self.ear_ema < self.ear_exit
        else:
            closed = self.ear_ema < self.ear_enter

        self.perclos.append(1 if closed else 0)

        if closed:
            self.closed_cnt += 1
            self.open_cnt = 0
        else:
            self.open_cnt += 1
            self.closed_cnt = 0

        if self.sleep_state == "AWAKE" and self.closed_cnt >= self.sleep_frames:
            self.sleep_state = "ASLEEP"
        elif self.sleep_state == "ASLEEP" and self.open_cnt >= self.wake_frames:
            self.sleep_state = "AWAKE"

        perclos_10s = float(sum(self.perclos)) / float(len(self.perclos)) if self.perclos else 0.0

        # ---------- distraction (only when awake) ----------
        if self.sleep_state == "ASLEEP":
            self.focus_state = "UNKNOWN"
            self.dist_frames = 0
        else:
            gaze_dev = abs(self.gaze_ema - 0.5) * 2.0   # 0..1
            yaw_dev = abs(self.yaw_ema) * 2.0           # 0..1
            score = 0.7 * gaze_dev + 0.3 * yaw_dev      # 0..1-ish

            if score > 0.45:
                self.dist_frames += 1
            else:
                self.dist_frames = 0

            self.focus_state = "DISTRACTED" if self.dist_frames >= self.dist_limit else "FOCUSED"

        return self._pack(perclos_10s=perclos_10s)

    def _pack(self, perclos_10s):
        return {
            "ear": float(self.ear_ema) if self.ear_ema is not None else 0.0,
            "gaze": float(self.gaze_ema) if self.gaze_ema is not None else 0.5,
            "yaw": float(self.yaw_ema) if self.yaw_ema is not None else 0.0,
            "perclos_10s": float(perclos_10s),
            "sleep_state": self.sleep_state,
            "focus_state": self.focus_state,
            "closed_cnt": int(self.closed_cnt),
            "open_cnt": int(self.open_cnt),
            "ear_enter": self.ear_enter,
            "ear_exit": self.ear_exit,
        }

# -------------------------
# Main
# -------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available")
        return

    detector = FaceMeshDetector()
    monitor = DriverMonitor(fps=20)

    ptime = 0.0
    fps_smooth = None

    print("Controls: q=quit, r=recalibrate, d=toggle debug ids, a=toggle gaze arrow")

    while True:
        ok, img = cap.read()
        if not ok:
            break

        # resize for consistent processing
        h0, w0 = img.shape[:2]
        target_w = 640
        if w0 != target_w:
            img = cv2.resize(img, (target_w, int(h0 * target_w / w0)), interpolation=cv2.INTER_LINEAR)

        lm = detector.detect(img)

        if lm is not None:
            draw_eye_overlay(img, lm, show_ids=monitor.debug_ids)

            info = monitor.update(lm)

            # optional gaze arrow
            if monitor.show_gaze_arrow and info["sleep_state"] != "CALIBRATING":
                draw_gaze_arrow(img, lm, info["gaze"])

            # UI
            h, w = img.shape[:2]
            cv2.putText(img, f"EAR: {info['ear']:.3f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Gaze: {info['gaze']:.2f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Yaw: {info['yaw']:.2f}", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"PERCLOS(10s): {info['perclos_10s']*100:.0f}%", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # top-right states
            sleep_state = info["sleep_state"]
            focus_state = info["focus_state"]
            sleep_color = (0, 255, 0) if sleep_state == "AWAKE" else (0, 0, 255)
            focus_color = (0, 255, 0) if focus_state == "FOCUSED" else (0, 0, 255)

            cv2.putText(img, f"SLEEP: {sleep_state}", (w - 260, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sleep_color, 2)
            cv2.putText(img, f"FOCUS: {focus_state}", (w - 260, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, focus_color, 2)

            # banners
            if sleep_state == "CALIBRATING":
                cv2.rectangle(img, (0, h - 80), (w, h), (255, 120, 0), -1)
                cv2.putText(img, "CALIBRATING — KEEP EYES OPEN", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            elif sleep_state == "ASLEEP":
                cv2.rectangle(img, (0, h - 80), (w, h), (0, 0, 255), -1)
                cv2.putText(img, "SLEEP DETECTED — ALERT!", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            elif focus_state == "DISTRACTED":
                cv2.rectangle(img, (0, h - 80), (w, h), (0, 0, 180), -1)
                cv2.putText(img, "DISTRACTION — LOOK AHEAD", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        else:
            cv2.putText(img, "NO FACE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # FPS
        ctime = time.time()
        dt = (ctime - ptime) if ptime else 0.0
        inst = (1.0 / dt) if dt > 1e-6 else 0.0
        fps_smooth = inst if fps_smooth is None else 0.2 * inst + 0.8 * fps_smooth
        ptime = ctime
        cv2.putText(img, f"FPS: {int(fps_smooth)}", (img.shape[1] - 110, 28), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)

        cv2.imshow("Driver Monitor — Sleep + Distraction", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            monitor.reset()
            print("Recalibrating...")
        elif key == ord("d"):
            monitor.debug_ids = not monitor.debug_ids
            print("Debug IDs:", monitor.debug_ids)
        elif key == ord("a"):
            monitor.show_gaze_arrow = not monitor.show_gaze_arrow
            print("Gaze arrow:", monitor.show_gaze_arrow)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
