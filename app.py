from flask import Flask, render_template, Response
import cv2
import os
import numpy as np

app = Flask(__name__)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

KNOWN_FACE_DIR = "known_face"
label_map = {}
faces = []
labels = []
label_id = 0

# Load training data
for person_name in os.listdir(KNOWN_FACE_DIR):
    person_path = os.path.join(KNOWN_FACE_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(label_id)

    label_id += 1

# Train model
recognizer.train(faces, np.array(labels))

# Start camera
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            face_roi = gray[y:y+h, x:x+w]

            try:
                predicted_id, confidence = recognizer.predict(face_roi)

                # Lower confidence = better match
                if confidence < 70:
                    name = label_map[predicted_id]
                else:
                    name = "Unknown"

            except:
                name = "Unknown"

            # Draw box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw name
            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
