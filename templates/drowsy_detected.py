"""
Eye / Drowsiness detection using MediaPipe FaceLandmarker (tasks API)
+ Eye Aspect Ratio (EAR).

Why this is better than Haar Cascades:
  - MediaPipe provides 478 precise facial landmarks (sub-pixel accuracy).
  - EAR is a proven metric: when eyes close, the ratio drops sharply.
  - Works across head angles, lighting, and skin tones that Haar fails on.
"""

import cv2
import time
import threading
import numpy as np
import mediapipe as mp
from camera_config import get_camera_index

# ── MediaPipe FaceLandmarker setup (tasks API) ──────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

# ── Eye landmark indices (from the 478-point mesh) ──────────────────
# Right eye (viewer's left)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Left eye (viewer's right)
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# EAR threshold — below this value, the eye is considered closed.
# Typical open-eye EAR ≈ 0.25–0.30;  closed ≈ 0.15 or lower.
EAR_THRESHOLD = 0.22

# ── Shared state (thread-safe) ──────────────────────────────────────
_eye_lock = threading.Lock()
_eyes_closed_since = None   # timestamp when eyes first closed
_eyes_are_closed = False
_alarm_active = False
CLOSED_THRESHOLD = 4        # seconds before alarm fires


def _eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Compute the Eye Aspect Ratio for one eye.

    EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)

    Points are ordered:  p1(corner) p2(top-outer) p3(top-inner)
                         p4(corner) p5(bottom-inner) p6(bottom-outer)
    """
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices],
        dtype=np.float64,
    )
    # vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # horizontal distance
    h1 = np.linalg.norm(pts[0] - pts[3])
    if h1 == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h1)


def get_eye_status():
    """Return whether the alarm should be active (eyes closed > threshold)."""
    with _eye_lock:
        return {"alarm": _alarm_active, "eyes_closed": _eyes_are_closed}


def eye_frames():
    global _eyes_closed_since, _eyes_are_closed, _alarm_active

    cap = cv2.VideoCapture(get_camera_index())

    # Create a new landmarker instance for this generator
    landmarker = FaceLandmarker.create_from_options(_landmarker_options)
    frame_timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_timestamp_ms += 33  # ~30 fps

        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect face landmarks
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        face_found = False
        eyes_open = False

        if results.face_landmarks:
            for face_lm in results.face_landmarks:
                face_found = True

                # Calculate EAR for both eyes
                left_ear = _eye_aspect_ratio(face_lm, LEFT_EYE, w, h)
                right_ear = _eye_aspect_ratio(face_lm, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                # ── Draw eye landmarks ──────────────────────────────
                for idx_list, color in [(LEFT_EYE, (0, 255, 128)),
                                        (RIGHT_EYE, (0, 255, 128))]:
                    pts = [(int(face_lm[i].x * w), int(face_lm[i].y * h))
                           for i in idx_list]
                    for p in pts:
                        cv2.circle(frame, p, 2, color, -1)
                    # Connect the landmarks with lines
                    for i in range(len(pts)):
                        cv2.line(frame, pts[i], pts[(i + 1) % len(pts)],
                                 color, 1)

                # ── Draw face bounding box (from mesh extents) ──────
                xs = [lm.x for lm in face_lm]
                ys = [lm.y for lm in face_lm]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                x2, y2 = int(max(xs) * w), int(max(ys) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 1)

                # ── Decide open / closed ────────────────────────────
                if avg_ear >= EAR_THRESHOLD:
                    eyes_open = True
                    cv2.putText(frame, f"Eyes Open  (EAR {avg_ear:.2f})",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Eyes Closed (EAR {avg_ear:.2f})",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                # Show EAR bar in top-right
                bar_w = int(avg_ear * 300)
                bar_color = (0, 255, 0) if avg_ear >= EAR_THRESHOLD else (0, 0, 255)
                cv2.rectangle(frame, (w - 160, 10), (w - 160 + bar_w, 30),
                              bar_color, -1)
                cv2.rectangle(frame, (w - 160, 10),
                              (w - 160 + int(EAR_THRESHOLD * 300), 30),
                              (255, 255, 255), 1)
                cv2.putText(frame, "EAR", (w - 200, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)

        # ── Update shared alarm state ───────────────────────────────
        with _eye_lock:
            if face_found and not eyes_open:
                _eyes_are_closed = True
                if _eyes_closed_since is None:
                    _eyes_closed_since = time.time()
                elif time.time() - _eyes_closed_since >= CLOSED_THRESHOLD:
                    _alarm_active = True
                    cv2.putText(frame, "!! DROWSINESS ALERT !!",
                                (30, h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 0, 255), 3)
            else:
                _eyes_are_closed = False
                _eyes_closed_since = None
                _alarm_active = False

        # ── Show timer while eyes are closing ───────────────────────
        with _eye_lock:
            if _eyes_closed_since is not None:
                elapsed = time.time() - _eyes_closed_since
                cv2.putText(frame, f"Closed: {elapsed:.1f}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 165, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    landmarker.close()