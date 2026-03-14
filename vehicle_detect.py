from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

# vehicle classes from COCO dataset
vehicle_classes = [1,2,3,5,7]
# bicycle, car, motorcycle, bus, truck

def vehicle_frames():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model(frame)

        for r in results:

            boxes = r.boxes

            for box in boxes:

                cls = int(box.cls[0])

                if cls in vehicle_classes:

                    x1,y1,x2,y2 = map(int, box.xyxy[0])

                    label = model.names[cls]

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    cv2.putText(frame,label,(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,
                                (0,255,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')