from flask import Flask, render_template, Response

from object_detection import object_frames
from eye_detection import eye_frames
from vehicle_detect import vehicle_frames

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


@app.route('/vehicle')
def vehicle():
    return Response(vehicle_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)