from ultralytics import YOLO
import cv2
import threading
from datetime import datetime
from camera_config import get_camera_index

model = YOLO("yolov8n.pt")

# vehicle classes from COCO dataset
vehicle_classes = [1, 2, 3, 5, 7]
# bicycle, car, motorcycle, bus, truck

vehicle_class_names = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Shared state for vehicle tracking
_vehicle_lock = threading.Lock()
_current_counts = {}  # live counts per frame
_capture_log = []     # list of captured snapshots


def get_current_counts():
    """Return current frame vehicle counts."""
    with _vehicle_lock:
        return dict(_current_counts)


def capture_snapshot():
    """Capture current vehicle counts with timestamp."""
    with _vehicle_lock:
        snapshot = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "counts": dict(_current_counts)
        }
        _capture_log.append(snapshot)
        return snapshot


def generate_report():
    """Generate a text report of all captured snapshots."""
    with _vehicle_lock:
        if not _capture_log:
            return "No captures recorded yet.\n"

        lines = []
        lines.append("=" * 55)
        lines.append("       VEHICLE DETECTION REPORT")
        lines.append("=" * 55)
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Total Captures: {len(_capture_log)}")
        lines.append("=" * 55)
        lines.append("")

        grand_total = {}

        for i, snap in enumerate(_capture_log, 1):
            lines.append(f"  Capture #{i}  |  {snap['timestamp']}")
            lines.append("-" * 40)
            total = 0
            for cls_name, count in snap["counts"].items():
                lines.append(f"    {cls_name:<15} : {count}")
                total += count
                grand_total[cls_name] = grand_total.get(cls_name, 0) + count
            lines.append(f"    {'TOTAL':<15} : {total}")
            lines.append("")

        lines.append("=" * 55)
        lines.append("  GRAND TOTAL ACROSS ALL CAPTURES")
        lines.append("-" * 40)
        overall = 0
        for cls_name, count in sorted(grand_total.items()):
            lines.append(f"    {cls_name:<15} : {count}")
            overall += count
        lines.append(f"    {'TOTAL':<15} : {overall}")
        lines.append("=" * 55)

        return "\n".join(lines) + "\n"


def clear_captures():
    """Clear all captured data."""
    with _vehicle_lock:
        _capture_log.clear()


def vehicle_frames():
    global _current_counts

    cap = cv2.VideoCapture(get_camera_index())

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model(frame)

        frame_counts = {}

        for r in results:

            boxes = r.boxes

            for box in boxes:

                cls = int(box.cls[0])

                if cls in vehicle_classes:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    label = model.names[cls]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)

                    frame_counts[label] = frame_counts.get(label, 0) + 1

        # Update shared counts
        with _vehicle_lock:
            _current_counts = frame_counts

        # Draw count overlay on frame
        y_offset = 30
        total = 0
        for name, count in frame_counts.items():
            cv2.putText(frame, f"{name}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            total += count
        cv2.putText(frame, f"Total vehicles: {total}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')