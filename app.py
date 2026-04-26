from flask import Flask, render_template, Response, jsonify, request
from camera_config import get_camera_index, set_camera_index

from object_detection import object_frames
from eye_detection import eye_frames, get_eye_status
from vehicle_detect import (vehicle_frames, capture_snapshot,
                            generate_report, get_current_counts,
                            clear_captures)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/object')
def object():
    return Response(object_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/eye')
def eye():
    return Response(eye_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/eye_status')
def eye_status():
    """Polled by frontend to check if buzzer should play."""
    return jsonify(get_eye_status())


@app.route('/vehicle')
def vehicle():
    return Response(vehicle_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/vehicle_counts')
def vehicle_counts():
    """Return current live vehicle counts."""
    return jsonify(get_current_counts())


@app.route('/capture_vehicles', methods=['POST'])
def capture_vehicles():
    """Capture current vehicle counts snapshot."""
    snapshot = capture_snapshot()
    return jsonify({"status": "captured", "data": snapshot})


@app.route('/vehicle_report')
def vehicle_report():
    """Download vehicle detection report as TXT."""
    report = generate_report()
    return Response(
        report,
        mimetype='text/plain',
        headers={"Content-Disposition": "attachment; filename=vehicle_report.txt"}
    )


@app.route('/clear_captures', methods=['POST'])
def clear_caps():
    """Clear all captured vehicle data."""
    clear_captures()
    return jsonify({"status": "cleared"})


@app.route('/set_camera', methods=['POST'])
def set_camera():
    """Switch the active camera index (0 = laptop, 1 = OBS, etc.)."""
    data = request.get_json(force=True)
    idx = int(data.get('index', 0))
    set_camera_index(idx)
    return jsonify({"status": "ok", "camera_index": idx})


@app.route('/get_camera')
def get_camera():
    """Return the current camera index."""
    return jsonify({"camera_index": get_camera_index()})


if __name__ == "__main__":
    app.run(debug=True)