"""
Shared camera configuration.
All detection modules read the camera index from here so it can be
changed at runtime through the web UI.
"""
import threading

_lock = threading.Lock()
_camera_index = 0  # default: laptop built-in webcam


def get_camera_index():
    """Return the currently selected camera index."""
    with _lock:
        return _camera_index


def set_camera_index(index: int):
    """Set the camera index (0 = laptop cam, 1 = OBS virtual cam, etc.)."""
    with _lock:
        global _camera_index
        _camera_index = index
