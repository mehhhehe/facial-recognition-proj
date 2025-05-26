from flask import Flask, render_template, redirect, url_for, request, jsonify
import numpy as np
import face_recognition
import cv2
import base64
import pickle
import datetime
import os

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
MIN_SIZE = 80
ASPECT_MIN   = 0.8    
ASPECT_MAX   = 1.35
KEEP_ONLY_LARGEST = True

app = Flask(__name__)

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]


# Fake in‑memory “state” for now
AUTHORIZED = False
USERNAME=''
DOCS = [
    {"id": 1, "name": "Quarterly‑Report.pdf"},
    {"id": 2, "name": "Project‑Plan.docx"},
    {"id": 3, "name": "Onboarding‑Guide.pptx"},
]

@app.route("/")
def index():
    """
    Live‑camera face‑scan page.
    The JS on the page will call /authorize once it thinks the user is accepted.
    """
    return render_template("index.html")


@app.route("/authorize", methods=["POST"])
def authorize():
    """Dummy endpoint (POST from JS) that flags the session as authorized."""
    data = request.json
    global USERNAME
    USERNAME=data['person']
    global AUTHORIZED
    AUTHORIZED = True
    return ("", 204)


@app.route("/dashboard")
def dashboard():

    if not AUTHORIZED:
        # no access – go back
        return redirect(url_for("index"))
    
    return render_template("dashboard.html", docs=DOCS, username=USERNAME)


@app.route("/document/<int:doc_id>")
def document(doc_id):
    # Protect again
    if not AUTHORIZED:
        return redirect(url_for("index"))

    doc = next((d for d in DOCS if d["id"] == doc_id), None)
    if not doc:
        return "Document not found", 404
    return render_template("document.html", doc=doc)


@app.route("/authorized")
def authorized():
    return "<h1>Access Granted</h1>"

@app.route("/authenticate", methods=["POST"])
def authenticate():
    data = request.json
    image_data = base64.b64decode(data["image"].split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face = face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.1,     
            minSize=(MIN_SIZE, MIN_SIZE))
    

    if face == []:
        return jsonify({"match": False, "reason": "No face found", 
                        'person': '', 'similarity': 0})

    clean_faces = []
    for (x, y, w, h) in face:
        if w < MIN_SIZE or h < MIN_SIZE:
            continue
        ar = w / h
        if ASPECT_MIN <= ar <= ASPECT_MAX:
            clean_faces.append((x, y, w, h))

    if clean_faces == []:
        return jsonify({"match": False, "reason": "No face found", 
                        'person': '', 'similarity': 0})
    if KEEP_ONLY_LARGEST:
        clean_faces = [max(clean_faces, key=lambda box: box[2] * box[3])]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"static/snapshots/snapshot_{timestamp}.jpg"
    for (x, y, w, h) in clean_faces:
        face_crop = frame[y:y + h, x:x + w].copy()
        cv2.imwrite(filename, face_crop)
    image = face_recognition.load_image_file(filename)
    os.remove(filename)
    encoding = face_recognition.face_encodings(image, model='large')

    # faces = face_recognition.face_encodings(frame)
    
    if not encoding:
        print('Face not found')
        return jsonify({"match": False, "reason": "No face found"})
    


    face_encoding = encoding[0]
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(distances)
    similarity = (1 - distances[best_match_index]) * 100
    
    
    
    if similarity >= 50:
        
        return jsonify({
            "match": True,
            "person": known_names[best_match_index],
            "similarity": round(similarity, 2)
        }), 200
    

    return jsonify({
        "match": False,
        "person": known_names[best_match_index],
        "similarity": round(similarity, 2), 
        "reason": "No match found"
    }), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5000, ssl_context=('cert.pem', "key.pem"),debug=True)
