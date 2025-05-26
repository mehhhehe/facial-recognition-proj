
import numpy as np
import face_recognition
import cv2
import base64
import os
import pickle 

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
MIN_SIZE = 130
ASPECT_MIN = 0.8    
ASPECT_MAX = 1.35
KEEP_ONLY_LARGEST = True

def encode_faces(image_folder='../lfw-deepfunneled'):
    name_folders = os.listdir(image_folder)
    for folder in name_folders:
        os.makedirs(f'{image_folder}/{folder}_boxed', exist_ok=True)

    encodings = []
    names = []

    for folder in name_folders:
        if 'boxed' not in folder:
            img_folder = os.path.join(image_folder, folder)
            print(img_folder)
            for img in os.listdir(img_folder):
                imagep = os.path.join(img_folder, img)
                frame = cv2.imread(imagep)
                if frame is None:
                    print(f"Warning: Failed to load image {imagep}")
                    continue

                image_gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = face_cascade.detectMultiScale(
                    image_gr,
                    scaleFactor=1.1,     
                    minSize=(MIN_SIZE, MIN_SIZE))

                clean_faces = []
                for (x, y, w, h) in face:
                    if w < MIN_SIZE or h < MIN_SIZE:
                        continue
                    ar = w / h
                    if ASPECT_MIN <= ar <= ASPECT_MAX:
                        clean_faces.append((x, y, w, h))

                if not clean_faces:
                    continue
                if KEEP_ONLY_LARGEST:
                    clean_faces = [max(clean_faces, key=lambda box: box[2] * box[3])]

                write_path = f'{image_folder}/{folder}_boxed/{img}'
                for (x, y, w, h) in clean_faces:
                    face_crop = image_gr[y:y + h, x:x + w].copy()
                    cv2.imwrite(write_path, face_crop)
                    print(write_path)

                    image = face_recognition.load_image_file(write_path)
                    encoding = face_recognition.face_encodings(image, num_jitters=2, model='large')
                    if encoding:
                        encodings.append(encoding[0])
                        names.append(folder)

    return encodings, names

def save_encodings_to_file(encodings, names, filename='encodings.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

if __name__ == "__main__":
    encodings, names = encode_faces()
    save_encodings_to_file(encodings, names)