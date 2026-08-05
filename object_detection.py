import cv2
import threading
from ultralytics import YOLO
from camera_config import get_camera_index

model = YOLO("yolov8l.pt")

# -- Shared state for weapon detection --
_weapon_lock = threading.Lock()
_weapon_detected = False

def get_weapon_status():
    """Return whether a weapon was detected in the last frame."""
    with _weapon_lock:
        return {"weapon_detected": _weapon_detected}

def object_frames():
    global _weapon_detected

    cap = cv2.VideoCapture(get_camera_index())

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Inference with "Large" model
        results = model(frame, conf=0.15, iou=0.45, imgsz=960)
        
        # Check for weapons (knife: 43, scissors: 76)
        detected_classes = results[0].boxes.cls.tolist()
        has_weapon = any(cls in [43, 76] for cls in detected_classes)
        
        with _weapon_lock:
            _weapon_detected = has_weapon

        frame = results[0].plot()

        # Add visual warning if weapon detected
        if has_weapon:
            cv2.putText(frame, "!! WEAPON DETECTED !!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')