import cv2
import face_recognition
import os
import numpy as np
import pyttsx3
import time

KNOWN_DIR = "known_face"
SCALE = 0.5
TOLERANCE = 0.45
RECOGNIZE_EVERY = 8       # run recognition once every N frames
NAME_HOLD_TIME = 2.0     # seconds to keep name locked

engine = pyttsx3.init()
engine.setProperty("rate", 160)

# ---------------- LOAD KNOWN FACES ----------------
known_encodings = []
known_names = []

for name in os.listdir(KNOWN_DIR):
    folder = os.path.join(KNOWN_DIR, name)
    if not os.path.isdir(folder):
        continue

    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        image = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(image)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(name)

print("[INFO] Known faces loaded:", len(known_names))

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(0)

last_name = "Unknown"
last_seen_time = 0
last_locations = []
frame_count = 0
spoken = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Always show last box (prevents blinking)
    for (top, right, bottom, left) in last_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, last_name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Run heavy recognition only every N frames
    if frame_count % RECOGNIZE_EVERY == 0:
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for (t, r, b, l), enc in zip(locations, encodings):
            distances = face_recognition.face_distance(known_encodings, enc)
            name = "Unknown"

            if len(distances) > 0:
                best = np.argmin(distances)
                if distances[best] < TOLERANCE:
                    name = known_names[best]

            # scale back
            t = int(t / SCALE)
            r = int(r / SCALE)
            b = int(b / SCALE)
            l = int(l / SCALE)

            last_locations = [(t, r, b, l)]
            last_name = name
            last_seen_time = time.time()

            if name != "Unknown" and name not in spoken:
                engine.say(name)
                engine.runAndWait()
                spoken.add(name)

    # Clear name if face gone for too long
    if time.time() - last_seen_time > NAME_HOLD_TIME:
        last_name = "Unknown"
        last_locations = []

    cv2.imshow("Smart Glass", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
