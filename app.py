from flask import Flask, render_template, request
import cv2 as cv
import mediapipe as mp
import numpy as np
import os
from insightface.app import FaceAnalysis

app = Flask(__name__)

# ---------------- InsightFace Setup ----------------
app_face = FaceAnalysis(name='buffalo_l')
app_face.prepare(ctx_id=-1)  # CPU (-1) | GPU (0)

# ---------------- MediaPipe Setup ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# ---------------- Recommendations ----------------
recommendations = {
    "Oval": ("Rectangular glasses", "Most hairstyles suit you"),
    "Round": ("Square glasses", "Layered hair"),
    "Square": ("Round glasses", "Soft textured hair"),
    "Unknown": ("Any glasses", "Any hairstyle"),
}

def classify_face_shape(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    face_width = max(xs) - min(xs)
    face_height = max(ys) - min(ys)
    ratio = face_height / face_width if face_width != 0 else 0

    if 1.3 < ratio < 1.55:
        return "Oval"
    elif ratio <= 1.3:
        return "Round"
    elif ratio >= 1.55:
        return "Square"
    else:
        return "Unknown"


@app.route('/', methods=['GET', 'POST'])
def index():
    shape = glasses = hair = None
    image_path = None
    no_face_detected = False
    msg = None
    error = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename != '':
            filepath = os.path.join("static/inputPhotos", file.filename)
            file.save(filepath)

            img = cv.imread(filepath)

            faces = app_face.get(img)

            print(f"No. of faces found = {len(faces)}")

            if len(faces) == 0:
                no_face_detected = True
                msg = "No face detected. Please upload a clear image."

            elif len(faces) > 1:
                error = "Multiple faces detected. Please upload a single face image."

            else:
                face = faces[0]
                x1, y1, x2, y2 = face.bbox.astype(int)

                cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                face_crop = img[y1:y2, x1:x2]

                if face_crop.size != 0:
                    rgb = cv.cvtColor(face_crop, cv.COLOR_BGR2RGB)

                    result = face_mesh.process(rgb)

                    if result.multi_face_landmarks:
                        shape = classify_face_shape(
                            result.multi_face_landmarks[0].landmark,
                            face_crop.shape[1],
                            face_crop.shape[0]
                        )

                        glasses, hair = recommendations.get(shape, recommendations["Unknown"])

            output_dir = "static/outputPhotos"
            os.makedirs(output_dir, exist_ok=True)

            image_path = os.path.join(output_dir, file.filename)
            cv.imwrite(image_path, img)

    return render_template(
        'index.html',
        shape=shape,
        glasses=glasses,
        hair=hair,
        image_path=image_path,
        no_face_detected=no_face_detected,
        msg=msg,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)