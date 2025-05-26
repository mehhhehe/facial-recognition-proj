import os
import cv2
import pickle
import numpy as np
import face_recognition
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Settings
ENCODING_FILE = "encodings.pkl"
TEST_DIR = "../lfw-deepfunneled"  # should have subfolders named after person labels
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MIN_SIZE = 130
ASPECT_MIN = 0.8
ASPECT_MAX = 1.35

# Load known encodings
with open(ENCODING_FILE, "rb") as f:
    data = pickle.load(f)
    known_encodings = np.array(data["encodings"])
    known_labels = np.array(data["names"])

# Load cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

y_true = []
y_pred = []
similarities = []

def detect_largest_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    clean = []
    for (x, y, w, h) in boxes:
        if w < MIN_SIZE or h < MIN_SIZE:
            continue
        ar = w / h
        if ASPECT_MIN <= ar <= ASPECT_MAX:
            clean.append((x, y, w, h))
    if not clean:
        return None
    return max(clean, key=lambda b: b[2] * b[3])

# Evaluation loop
for label in os.listdir(TEST_DIR):
    if 'boxed' in label:
        continue
    folder = os.path.join(TEST_DIR, label)
    if not os.path.isdir(folder):
        continue
    i = 0    
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        face_box = detect_largest_face(img)
        if face_box is None:
            continue

        x, y, w, h = face_box
        face_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(face_rgb, [(0, w, h, 0)], model='large', num_jitters=2)
        if not enc:
            continue

        query = enc[0]
        distances = face_recognition.face_distance(known_encodings, query)
        idx = np.argmin(distances)
        similarity = (1 - distances[idx]) * 100
        predicted_label = known_labels[idx]

        y_true.append(label)
        y_pred.append(predicted_label)
        similarities.append(similarity)
        i += 1
        if i > 7:
            break

# Results
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc*100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

confmat = confusion_matrix(y_true, y_pred, labels=list(set(known_labels)))
print("\nConfusion Matrix:\n")
print(confmat)
