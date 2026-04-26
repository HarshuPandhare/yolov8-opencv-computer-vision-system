# AI Real-Time Vision Detection System - Project Explanation

This document provides a detailed breakdown of the codebase, the project's architecture, and the flow of data within the system. 

## 1. Project Overview & Flow

The system is a multi-modal computer vision application built on **Flask** (Backend/Web Server) and **OpenCV** (Video Processing), utilizing deep learning models like **YOLOv8** (for object/vehicle detection) and **MediaPipe** (for facial landmarks/eye detection).

### Data Flow
1. **Frontend (Browser)**: The user accesses the web interface (`index.html`). They can select a camera source and choose a detection mode (Object, Vehicle, or Eye).
2. **Backend (Flask)**: When a mode is selected, the browser requests a video stream from a specific Flask route (e.g., `/object`, `/vehicle`, `/eye`).
3. **Computer Vision Processing**: 
   - Flask triggers a generator function (e.g., `object_frames()`) that captures frames from the selected camera.
   - The frames are passed through the respective AI model (YOLOv8 or MediaPipe).
   - Bounding boxes, labels, and overlays (like vehicle counts or EAR values) are drawn onto the frames using OpenCV.
4. **Streaming**: The processed frames are encoded to JPEG and streamed continuously back to the browser using Multipart HTTP responses (`multipart/x-mixed-replace`).
5. **Asynchronous Polling**: For features like the Drowsiness Alert, the frontend periodically polls a JSON endpoint (`/eye_status`) to check if the alarm should trigger and plays a buzzer if necessary.

---

## 2. File-by-File Breakdown

### `app.py`
This is the main entry point for the Flask web application.
- **Purpose**: It initializes the Flask server and defines all the HTTP routes used by the frontend.
- **Key Functions/Routes**:
  - `/`: Serves the `index.html` frontend.
  - `/object`, `/eye`, `/vehicle`: Stream the processed video frames for their respective modes using Flask's `Response` object.
  - `/eye_status`: An endpoint polled by the frontend to check if the user's eyes have been closed long enough to trigger an alarm.
  - `/vehicle_counts`, `/capture_vehicles`, `/vehicle_report`, `/clear_captures`: API endpoints for the vehicle detection mode to handle live counts, snapshots, and report generation.
  - `/set_camera`, `/get_camera`: Endpoints to globally switch the active webcam source.

### `object_detection.py`
Handles general object detection using the YOLOv8 neural network.
- **Purpose**: Detects 80+ different objects from the COCO dataset in real-time.
- **Working**: 
  - Loads the `yolov8n.pt` (nano) model.
  - `object_frames()`: Continuously reads frames from the active camera, passes them to YOLO, uses YOLO's built-in `.plot()` method to draw bounding boxes, and yields the encoded JPEG frames for the Flask stream.

### `vehicle_detect.py`
A specialized YOLOv8 implementation focused purely on tracking and counting vehicles.
- **Purpose**: Detects specific vehicle classes (bicycle, car, motorcycle, bus, truck), counts them, and allows saving snapshots and generating text reports.
- **Working**:
  - Filters YOLOv8 detections to only include vehicle class IDs.
  - Uses a shared thread lock (`_vehicle_lock`) to safely store the current frame's vehicle counts.
  - Draws custom bounding boxes and on-screen text for live counting.
  - `capture_snapshot()`: Saves the current count to a list (`_capture_log`) with a timestamp.
  - `generate_report()`: Formats all captured logs into a clean, downloadable text report.

### `eye_detection.py`
Implements Drowsiness Detection using MediaPipe's FaceLandmarker tasks API.
- **Purpose**: Tracks facial landmarks to calculate the Eye Aspect Ratio (EAR). If the EAR drops below a threshold (eyes closed) for a certain amount of time (4 seconds), it triggers an alarm state.
- **Working**:
  - Uses the `face_landmarker.task` model to get a precise 478-point mesh of the face.
  - `_eye_aspect_ratio()`: Calculates the vertical distance vs horizontal distance of the eye landmarks. 
  - Monitors the EAR. If `EAR < 0.22`, the eyes are considered closed.
  - Uses a thread lock (`_eye_lock`) to securely update shared state variables (`_eyes_are_closed`, `_alarm_active`) which are fetched by the `app.py` `/eye_status` endpoint.

### `camera_config.py`
A simple shared state module to handle the active camera index.
- **Purpose**: Solves the problem of needing to know which camera to use across multiple different Python files.
- **Working**: Maintains a thread-safe `_camera_index` variable (default `0`). It provides getter and setter functions (`get_camera_index`, `set_camera_index`) so the frontend can easily switch between a laptop webcam and external sources like OBS Virtual Camera.

### `templates/index.html`
The frontend user interface of the application.
- **Purpose**: Provides a dynamic, visually appealing interface to interact with the backend features.
- **Working**:
  - **Styling**: Uses custom CSS with animations, a futuristic grid background, and responsive layouts.
  - **Video Display**: The `<img>` tag acts as the video player by directly linking its `src` to the Flask video stream routes (`/object`, etc.).
  - **Interactivity**: Contains JavaScript functions (`startObject()`, `startEye()`, etc.) to dynamically change the `src` of the video element, essentially switching the AI mode without reloading the page.
  - **Asynchronous Fetching**: Uses JS `fetch()` to hit backend APIs for taking vehicle snapshots, downloading reports, and swapping the active camera.
  - **Drowsiness Alarm**: Contains an `<audio>` tag for the buzzer. In Eye Detection mode, a `setInterval` loop polls `/eye_status` every 500ms and plays the buzzer if the backend reports a state of drowsiness.

---

## 3. Summary of Execution
1. You run `python app.py`.
2. Flask starts a local web server (usually `http://127.0.0.1:5000`).
3. You open the browser. The browser downloads `index.html`.
4. You click "Object Detection". The JS updates the image `src` to `/object`.
5. Flask calls `object_frames()` in `object_detection.py`, which opens the webcam (using index from `camera_config.py`), runs YOLO, and streams the images back.
6. If you switch to "Eye Detection", the JS changes the `src` to `/eye`. The browser drops the `/object` connection (closing the camera in that script), and Flask starts `eye_frames()` in `eye_detection.py`, initializing MediaPipe and streaming the new feed.
