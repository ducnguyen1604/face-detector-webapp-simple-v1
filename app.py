from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load your pre-trained model
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def gen_frames():  
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        if not ret:
            break  # If no frame is captured, exit the loop

        face = detector(frame)

        if face is not None:
            ret, buffer = cv2.imencode('.jpg', face)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def detector(img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.2:
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)

    return img


@app.route('/video')
def frame_gen():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True, threaded=True)



