import time
import math
from ultralytics import YOLO
import cv2
import threading
from datetime import datetime
from camera_config import get_camera_index

model = YOLO("yolov8l.pt")

# Expanded classes for more comprehensive detection
vehicle_classes = [1, 2, 3, 5, 6, 7, 8, 9, 11]
vehicle_class_names = {
    1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 11: "stop sign"
}

# Shared state
_vehicle_lock = threading.Lock()
_current_counts = {}  
_capture_log = []     
_truck_restriction_enabled = False

# Tracking & Speed Estimation state
_track_history = {}  # {track_id: (center_pos, timestamp, last_speed)}
_PIXELS_PER_METER = 15  

# Traffic Intelligence Metrics
_entry_ids = set()      # IDs seen in current 60s window
_vpm = 0                # Vehicles Per Minute
_avg_speed = 0          # Average speed in current window
_last_reset = time.time()
_prediction_data = {
    "status": "Clear",
    "message": "Initializing...",
    "distance_km": 0,
    "time_mins": 0
}


def get_traffic_status():
    """Return comprehensive traffic intelligence data."""
    with _vehicle_lock:
        return {
            "vpm": round(_vpm, 1),
            "avg_speed": round(_avg_speed, 1),
            "prediction": _prediction_data
        }


def set_truck_restriction(enabled):
    global _truck_restriction_enabled
    with _vehicle_lock:
        _truck_restriction_enabled = enabled


def get_truck_restriction():
    with _vehicle_lock:
        return _truck_restriction_enabled


def get_current_counts():
    """Return current frame vehicle counts."""
    with _vehicle_lock:
        return dict(_current_counts)


def capture_snapshot():
    """Capture current vehicle counts with timestamp and restriction check."""
    with _vehicle_lock:
        has_truck = _current_counts.get("truck", 0) > 0
        violation = _truck_restriction_enabled and has_truck
        
        snapshot = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "counts": dict(_current_counts),
            "restriction_warning": "⚠ VIOLATION: TRUCK DETECTED" if violation else None,
            "traffic_prediction": _prediction_data["message"]
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
        lines.append("       VEHICLE DETECTION & TRAFFIC INTELLIGENCE")
        lines.append("=" * 55)
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Current Flow: {_vpm:.1f} vehicles/min")
        lines.append(f"  Avg Speed: {_avg_speed:.1f} km/h")
        lines.append("-" * 55)
        lines.append("")

        grand_total = {}

        for i, snap in enumerate(_capture_log, 1):
            lines.append(f"  Capture #{i}  |  {snap['timestamp']}")
            if snap.get("restriction_warning"):
                lines.append(f"    [!] {snap['restriction_warning']}")
            lines.append(f"    Prediction: {snap.get('traffic_prediction', 'N/A')}")
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
    global _current_counts, _track_history, _entry_ids, _vpm, _avg_speed, _last_reset, _prediction_data

    cap = cv2.VideoCapture(get_camera_index())

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Inference with tracking enabled
        results = model.track(frame, persist=True, conf=0.15, iou=0.45, imgsz=960)

        frame_counts = {}
        current_time = time.time()
        speeds_this_frame = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if int(cls) not in vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                label = model.names[int(cls)]
                
                # Flow tracking
                _entry_ids.add(track_id)
                
                # Speed Estimation
                speed_text = ""
                if track_id in _track_history:
                    prev_pos, prev_time, prev_speed = _track_history[track_id]
                    dt = current_time - prev_time
                    
                    if dt > 0.1:  # Update speed every 100ms
                        dist_px = math.sqrt((cx - prev_pos[0])**2 + (cy - prev_pos[1])**2)
                        # Speed = (Pixels / PixelsPerMeter) / Seconds * 3.6 (m/s to km/h)
                        speed_kmh = (dist_px / _PIXELS_PER_METER) / dt * 3.6
                        
                        # Simple smoothing (moving average)
                        if prev_speed > 0:
                            speed_kmh = (prev_speed * 0.7) + (speed_kmh * 0.3)
                        
                        _track_history[track_id] = ((cx, cy), current_time, speed_kmh)
                        speed_text = f"{speed_kmh:.1f} km/h"
                        speeds_this_frame.append(speed_kmh)
                    else:
                        speed_text = f"{prev_speed:.1f} km/h" if prev_speed > 0 else ""
                        if prev_speed > 0: speeds_this_frame.append(prev_speed)
                else:
                    _track_history[track_id] = ((cx, cy), current_time, 0)

                # Drawing
                color = (0, 255, 0)
                if label == "truck" and _truck_restriction_enabled:
                    color = (0, 0, 255) # Red for restricted trucks
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                display_label = f"ID:{track_id} {label}"
                if speed_text:
                    display_label += f" | {speed_text}"
                
                cv2.putText(frame, display_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                frame_counts[label] = frame_counts.get(label, 0) + 1

        # Periodic cleanup of track history (prevent memory leak)
        if len(_track_history) > 100:
            _track_history = {tid: val for tid, val in _track_history.items() 
                              if current_time - val[1] < 5}

        # Traffic Intelligence Calculation (every 5 seconds)
        elapsed = current_time - _last_reset
        if elapsed >= 5:
            with _vehicle_lock:
                _vpm = (len(_entry_ids) / elapsed) * 60
                if speeds_this_frame:
                    new_avg = sum(speeds_this_frame) / len(speeds_this_frame)
                    _avg_speed = (_avg_speed * 0.5) + (new_avg * 0.5) if _avg_speed > 0 else new_avg
                
                # Traffic Prediction Logic
                if _vpm < 5:
                    _prediction_data = {"status": "Clear", "message": "Smooth flow; road clear for 5km+", "distance_km": 5, "time_mins": 10}
                elif _vpm < 15:
                    _prediction_data = {"status": "Moderate", "message": "Moderate traffic; slight delay in 2.5km", "distance_km": 2.5, "time_mins": 5}
                elif _vpm >= 15 and _avg_speed < 20:
                    _prediction_data = {"status": "Congested", "message": "HEAVY TRAFFIC; gridlock likely in 500m", "distance_km": 0.5, "time_mins": 2}
                else:
                    _prediction_data = {"status": "Busy", "message": "High volume; expect slowdown in 1.2km", "distance_km": 1.2, "time_mins": 3}
                
                # Reset window
                _entry_ids.clear()
                _last_reset = current_time

        # Update shared counts
        with _vehicle_lock:
            _current_counts = frame_counts

        # Draw UI Overlays
        y_offset = 30
        total = 0
        cv2.putText(frame, f"FLOW: {_vpm:.1f} VPM | SPEED: {_avg_speed:.1f} km/h", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        cv2.putText(frame, f"PREDICTION: {_prediction_data['status']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        y_offset += 25

        if _truck_restriction_enabled:
            cv2.putText(frame, "TRUCK RESTRICTION: ACTIVE", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25

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